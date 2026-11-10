"""
Adds the environment (floor, walls, roof, block) to a Drake MultibodyPlant.

Typical use:

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

    layout = make_two_segment_layout(EnvironmentParams())
    env = add_environment_to_plant(plant, layout)
    # ... later: add the iiwa and gripper to the same plant here ...
    plant.Finalize()

The floor, walls and roof are fixed to the world (they never move) and have
collision geometry, so the robot's collision checks will see them. The block
is a free body that starts at the start pose.
"""

from dataclasses import dataclass

import numpy as np
from pydrake.all import (
    AddCompliantHydroelasticProperties,
    AddContactMaterial,
    AddRigidHydroelasticProperties,
    Box,
    Convex,
    CoulombFriction,
    Cylinder,
    MultibodyPlant,
    ProximityProperties,
    RigidBody,
    RigidTransform,
    RotationMatrix,
    SpatialInertia,
)

from maze_layout import MazeLayout, Rect


# Colors as (red, green, blue, alpha). The light blue / green scheme matches
# the proposal figures.
FLOOR_COLOR = np.array([0.85, 0.85, 0.85, 1.0])
OPEN_AREA_COLOR = np.array([0.80, 0.88, 0.96, 1.0])    # light blue floor paint
WALL_COLOR = np.array([0.45, 0.50, 0.60, 1.0])
ROOF_COLOR = np.array([0.95, 0.60, 0.20, 0.5])         # see-through orange
BLOCK_COLOR = np.array([0.35, 0.65, 0.30, 1.0])        # green
GOAL_MARKER_COLOR = np.array([0.35, 0.65, 0.30, 0.3])  # see-through green
ROBOT_BASE_COLOR = np.array([0.30, 0.30, 0.30, 1.0])

# ----------------------------------------------------------------------
# Contact properties
# ----------------------------------------------------------------------
# We use Drake's "hydroelastic" contact: the block is slightly soft, the
# floor/walls/roof are rigid. Compared to the simpler point contact, this
# gives a contact *patch* (not a single point) when two flat faces touch,
# so the block rests stably on the floor and a gripper pinch resists twisting.
FRICTION = CoulombFriction(static_friction=0.8, dynamic_friction=0.6)
BLOCK_STIFFNESS = 1e7      # hydroelastic modulus in Pa (roughly a stiff plastic)
CONTACT_DAMPING = 5.0      # Hunt-Crossley dissipation, s/m
BLOCK_MESH_SIZE = 0.01     # m; how finely Drake meshes the soft block internally


def rigid_contact_properties() -> ProximityProperties:
    """For fixed scenery: floor, walls, roof."""
    properties = ProximityProperties()
    AddContactMaterial(dissipation=CONTACT_DAMPING, friction=FRICTION, properties=properties)
    AddRigidHydroelasticProperties(properties=properties)
    return properties


def soft_contact_properties() -> ProximityProperties:
    """For the block."""
    properties = ProximityProperties()
    AddContactMaterial(dissipation=CONTACT_DAMPING, friction=FRICTION, properties=properties)
    AddCompliantHydroelasticProperties(resolution_hint=BLOCK_MESH_SIZE,
                                       hydroelastic_modulus=BLOCK_STIFFNESS,
                                       properties=properties)
    return properties


@dataclass
class Environment:
    """Handles to what was added, for use by later code (planning, etc.)."""
    layout: MazeLayout
    block: RigidBody
    block_start_pose: RigidTransform  # pose of the block in the world (X_WB)
    block_goal_pose: RigidTransform


def add_environment_to_plant(plant: MultibodyPlant, layout: MazeLayout) -> Environment:
    """Add everything in `layout` to `plant`. Call before plant.Finalize()."""
    p = layout.params

    # --- Floor: a slab whose top surface is at z = 0 ---
    add_fixed_box(plant, "floor",
                  size=(p.floor_size, p.floor_size, p.floor_thickness),
                  pose=RigidTransform([0, 0, -p.floor_thickness / 2]),
                  color=FLOOR_COLOR)

    # --- Light blue paint on the floor marking the open areas ---
    # (Visual only: just makes the corridors and bay easy to see.)
    paint_thickness = 0.0005
    for name, rect in layout.open_areas.items():
        add_visual_only_box(plant, f"{name}_paint",
                            size=(rect.length, rect.width, paint_thickness),
                            pose=pose_of_rect(rect, z_center=paint_thickness / 2),
                            color=OPEN_AREA_COLOR)

    # --- Walls: triangular prisms standing on the floor ---
    for name, footprint in layout.walls.items():
        add_fixed_prism(plant, name, footprint, height=p.wall_height, color=WALL_COLOR)

    # --- Roofs: resting on top of the walls ---
    for name, rect in layout.roofs.items():
        add_fixed_box(plant, name,
                      size=(rect.length, rect.width, p.roof_thickness),
                      pose=pose_of_rect(rect, z_center=p.wall_height + p.roof_thickness / 2),
                      color=ROOF_COLOR)

    # --- The block: a free body, lying flat on the floor ---
    block_z = p.block_height / 2
    block_start_pose = pose_of_rect(layout.block_start, z_center=block_z)
    block_goal_pose = pose_of_rect(layout.block_goal, z_center=block_z)
    block = add_block(plant, p.block_length, p.block_width, p.block_height,
                      p.block_mass, block_start_pose)

    # --- A see-through "ghost" block showing the goal (visual only) ---
    add_visual_only_box(plant, "block_goal_marker",
                        size=(p.block_length, p.block_width, p.block_height),
                        pose=block_goal_pose,
                        color=GOAL_MARKER_COLOR)

    # --- A flat disk marking where the robot base will go (visual only) ---
    plant.RegisterVisualGeometry(plant.world_body(), RigidTransform([0, 0, 0.001]),
                                 Cylinder(radius=0.10, length=0.002),
                                 "robot_base_marker", ROBOT_BASE_COLOR)

    return Environment(layout=layout,
                       block=block,
                       block_start_pose=block_start_pose,
                       block_goal_pose=block_goal_pose)


# ======================================================================
# Small helpers
# ======================================================================

def pose_of_rect(rect: Rect, z_center: float) -> RigidTransform:
    """3D pose for a box whose footprint is `rect` and whose center is at
    height `z_center`. The box's x-axis points along rect.yaw."""
    x, y = rect.center
    return RigidTransform(RotationMatrix.MakeZRotation(rect.yaw), [x, y, z_center])


def add_fixed_box(plant, name, size, pose, color):
    """A box welded to the world, with both collision and visual geometry."""
    shape = Box(*size)
    world = plant.world_body()
    plant.RegisterCollisionGeometry(world, pose, shape, f"{name}_collision",
                                    rigid_contact_properties())
    plant.RegisterVisualGeometry(world, pose, shape, f"{name}_visual", color)


def add_fixed_prism(plant, name, footprint, height, color):
    """A straight-sided solid welded to the world: the 2D `footprint`
    (an (N, 2) array of corner points, in world x-y) extruded from the
    floor up to `height`. The footprint must be convex."""
    bottom = [[x, y, 0.0] for x, y in footprint]
    top = [[x, y, height] for x, y in footprint]
    corner_points = np.array(bottom + top).T  # Drake wants shape (3, N)
    shape = Convex(corner_points, name)

    world = plant.world_body()
    at_origin = RigidTransform()  # the points are already in world coordinates
    plant.RegisterCollisionGeometry(world, at_origin, shape, f"{name}_collision",
                                    rigid_contact_properties())
    plant.RegisterVisualGeometry(world, at_origin, shape, f"{name}_visual", color)


def add_visual_only_box(plant, name, size, pose, color):
    """A box welded to the world that is only drawn (nothing can hit it)."""
    plant.RegisterVisualGeometry(plant.world_body(), pose, Box(*size), name, color)


def add_block(plant, length, width, height, mass, start_pose) -> RigidBody:
    """The block, as a free-floating rigid body in its own model instance.

    Its body frame is at the block's center, with x along its length,
    y along its width and z along its height.
    """
    model = plant.AddModelInstance("block")
    block = plant.AddRigidBody("block", model,
                               SpatialInertia.SolidBoxWithMass(mass, length, width, height))
    shape = Box(length, width, height)
    plant.RegisterCollisionGeometry(block, RigidTransform(), shape, "block_collision",
                                    soft_contact_properties())
    plant.RegisterVisualGeometry(block, RigidTransform(), shape, "block_visual", BLOCK_COLOR)
    # Every new simulation context will start with the block here.
    plant.SetDefaultFreeBodyPose(block, start_pose)
    return block
