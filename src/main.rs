use std::{ops::Sub, time::Duration};

use bevy::{
    app::{App, FixedUpdate, PostUpdate, PreUpdate, Startup},
    asset::Assets,
    color::Color,
    math::Vec2,
    prelude::{
        Bundle, Camera2dBundle, Circle, Commands, Component, Entity, In, IntoSystem, Mesh,
        PluginGroup, Query, Res, ResMut, Resource, Transform, With, World,
    },
    sprite::{ColorMaterial, MaterialMesh2dBundle, Mesh2dHandle},
    time::{Fixed, Time},
    window::{Window, WindowPlugin, WindowResolution},
    DefaultPlugins,
};
use bevy_spatial::{kdtree::KDTree2, AutomaticUpdate, SpatialAccess, SpatialStructure};
use rand::{thread_rng, Rng};

const PARTICLE_SIZE: f32 = 1.5;
const NUMBER_OF_PARTICLES: u64 = 10000;
const WINDOW_WIDTH: f32 = 800.0;
const WINDOW_HEIGHT: f32 = 800.0;
const BETA_REPULSION_DISTANCE: f32 = 0.5;
const MAXIMUM_RADIUS_OF_EFFECT: f32 = 20.0;
const FRICTION_HALF_TIME: f32 = 0.4;
const NUMBER_OF_TYPES: usize = 5;

fn main() {
    App::new()
        .add_plugins(
            AutomaticUpdate::<Point>::new()
                .with_frequency(Duration::from_secs_f32(0.003))
                .with_spatial_ds(SpatialStructure::KDTree2),
        )
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                resolution: WindowResolution::new(WINDOW_WIDTH, WINDOW_HEIGHT),
                ..Default::default()
            }),
            ..Default::default()
        }))
        .insert_resource(Time::<Fixed>::from_hz(10.0))
        .init_resource::<AttractionFactors>()
        .add_systems(Startup, setup)
        .add_systems(PreUpdate, position_to_translation)
        .add_systems(
            FixedUpdate,
            get_particles
                .pipe(compute_acceleration)
                .pipe(apply_acceleration),
        )
        .add_systems(PostUpdate, wrap_particles)
        .run();
}

#[derive(Component, Default)]
struct Point;

type PointKDTree = KDTree2<Point>;

#[derive(Component, Default, Clone, Copy)]
struct Velocity(Vec2);

#[derive(Component, Default, Clone, Copy)]
struct Position(Vec2);

impl Sub<Position> for Position {
    type Output = Position;

    fn sub(self, rhs: Position) -> Self::Output {
        Position(self.0 - rhs.0)
    }
}

#[derive(Component, Default, Clone, Copy)]
struct Type(usize);

#[derive(Bundle)]
struct Particle {
    point: Point,
    velocity: Velocity,
    position: Position,
    ty: Type,
    mesh: MaterialMesh2dBundle<ColorMaterial>,
}

#[derive(Resource)]
struct AttractionFactors {
    number_of_types: usize,
    matrix: Vec<f32>,
}

impl AttractionFactors {
    fn get_factor(&self, type1: Type, type2: Type) -> f32 {
        self.matrix[type1.0 * self.number_of_types + type2.0]
    }
}

impl Default for AttractionFactors {
    fn default() -> Self {
        Self {
            number_of_types: NUMBER_OF_TYPES,
            matrix: (0..NUMBER_OF_TYPES * NUMBER_OF_TYPES)
                .map(|_| thread_rng().gen::<f32>() * 2.0 - 1.0)
                .collect(),
        }
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let mut rng = rand::thread_rng();
    commands.spawn(Camera2dBundle::default());

    let h = rng.gen::<f32>() * 360.0;
    let colors: Vec<Color> = (0..NUMBER_OF_TYPES)
        .map(|index| Color::hsv(h + 97.0 * index as f32, 0.6, 1.0))
        .collect();

    for _ in 0..NUMBER_OF_PARTICLES {
        let x: f32 = rng.gen();
        let y: f32 = rng.gen();
        let ty: usize = rng.gen_range(0..NUMBER_OF_TYPES);
        spawn_particle(
            &mut commands,
            &mut meshes,
            &mut materials,
            &colors,
            Position(Vec2::new(
                (x - 0.5) * WINDOW_WIDTH,
                (y - 0.5) * WINDOW_HEIGHT,
            )),
            Type(ty),
        );
    }
}

fn spawn_particle(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    colors: &[Color],
    location: Position,
    ty: Type,
) {
    let coordinates = location.0.extend(0.0);
    commands.spawn(Particle {
        point: Point,
        velocity: Velocity::default(),
        position: location,
        ty,
        mesh: MaterialMesh2dBundle {
            mesh: Mesh2dHandle(meshes.add(Circle {
                radius: PARTICLE_SIZE,
            })),
            material: materials.add(colors[ty.0]),
            transform: Transform::from_xyz(coordinates.x, coordinates.y, coordinates.z),
            ..Default::default()
        },
    });
}

fn position_to_translation(mut positions: Query<(&Position, &mut Transform)>) {
    for (position, mut transform) in positions.iter_mut() {
        transform.translation = position.0.extend(transform.translation.z);
    }
}

fn get_particles(
    particles: Query<(&Transform, Entity), With<Point>>,
    tree: Res<PointKDTree>,
) -> Vec<Vec<Entity>> {
    let mut list_of_nearest_entities = Vec::new();
    for (transform, this_entity) in particles.iter() {
        // We add it as the first entity.
        // This would make using `get_many_entities_dynamic` easier.
        let mut nearest_entities = vec![this_entity];
        for (_, other_entity) in
            tree.within_distance(transform.translation.truncate(), MAXIMUM_RADIUS_OF_EFFECT)
        {
            let other_entity = other_entity.unwrap();
            if this_entity == other_entity {
                continue;
            }
            nearest_entities.push(other_entity);
        }
        list_of_nearest_entities.push(nearest_entities);
    }
    list_of_nearest_entities
}

fn compute_acceleration(
    In(list_of_nearest_entites): In<Vec<Vec<Entity>>>,
    // We want to use `get_many_entities_dynamic`
    world: &mut World,
) -> Vec<(Entity, Vec2)> {
    let mut accelerations = Vec::new();
    let attraction_factors = world.get_resource::<AttractionFactors>().unwrap();
    for entities in list_of_nearest_entites {
        let entities = world.get_many_entities_dynamic(&entities).unwrap();
        let this_entity = entities[0];
        let this_type: Type = *this_entity.get().unwrap();
        let this_position: Position = *this_entity.get().unwrap();
        let mut acceleration = Vec2::default();
        for other_entity in &entities[1..] {
            let other_type: Type = *other_entity.get().unwrap();
            let other_position: Position = *other_entity.get().unwrap();
            let vector = other_position - this_position;
            let distance = vector.0.length();
            let force = force(
                attraction_factors.get_factor(this_type, other_type),
                distance / MAXIMUM_RADIUS_OF_EFFECT,
            );
            acceleration += (vector.0 / distance) * force;
        }
        acceleration *= MAXIMUM_RADIUS_OF_EFFECT;
        accelerations.push((this_entity.id(), acceleration));
    }
    accelerations
}

fn apply_acceleration(
    In(accelerations): In<Vec<(Entity, Vec2)>>,
    mut pos_vels: Query<(&mut Position, &mut Velocity)>,
    time: Res<Time<Fixed>>,
) {
    let delta_time = time.delta_seconds();
    for (this_entity, acceleration) in accelerations {
        let (mut position, mut velocity) = pos_vels.get_mut(this_entity).unwrap();
        let old_velocity = velocity.0;
        let friction = 0.5f32.powf(delta_time / FRICTION_HALF_TIME);
        velocity.0 = friction * old_velocity + acceleration * delta_time;
        let old_position = position.0;
        position.0 = old_position + velocity.0 * delta_time;
    }
}

fn wrap_particles(mut particles: Query<&mut Position, With<Point>>) {
    for mut position in particles.iter_mut() {
        position.0.x =
            (position.0.x + WINDOW_WIDTH / 2.0).rem_euclid(WINDOW_WIDTH) - WINDOW_WIDTH / 2.0;
        position.0.y =
            (position.0.y + WINDOW_HEIGHT / 2.0).rem_euclid(WINDOW_HEIGHT) - WINDOW_HEIGHT / 2.0;
    }
}

fn force(attraction_factor: f32, distance: f32) -> f32 {
    if distance < BETA_REPULSION_DISTANCE {
        distance / BETA_REPULSION_DISTANCE - 1.0
    } else if (BETA_REPULSION_DISTANCE..1.0).contains(&distance) {
        let numerator = (2.0 * distance - 1.0 - BETA_REPULSION_DISTANCE).abs();
        let denominator = 1.0 - BETA_REPULSION_DISTANCE;
        attraction_factor * (1.0 - (numerator / denominator))
    } else {
        0.0
    }
}
