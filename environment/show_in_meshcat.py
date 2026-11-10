"""
Show the environment in Meshcat, Drake's 3D viewer (opens in a web browser).

Run from this folder:
    .venv/bin/python show_in_meshcat.py             # the default environment
    .venv/bin/python show_in_meshcat.py --seed 7    # a randomized environment

The script simulates a couple of seconds of gravity so you can check that the
block sits still on the floor (if it were stuck inside a wall it would jump).
"""

import argparse

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    MeshcatVisualizer,
    Simulator,
    StartMeshcat,
)

from drake_environment import add_environment_to_plant
from environment_params import EnvironmentParams, sample_random_params
from maze_layout import make_two_segment_layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="randomize the environment with this seed")
    parser.add_argument("--sim-seconds", type=float, default=2.0,
                        help="how long to simulate gravity for")
    parser.add_argument("--no-wait", action="store_true",
                        help="exit right away instead of keeping the viewer open")
    args = parser.parse_args()

    # 1. Choose the environment's numbers.
    if args.seed is None:
        params = EnvironmentParams()
    else:
        params = sample_random_params(seed=args.seed)
    layout = make_two_segment_layout(params)
    print_summary(layout)

    # 2. Build the Drake diagram: plant + scene graph + viewer.
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    env = add_environment_to_plant(plant, layout)
    plant.Finalize()

    meshcat = StartMeshcat()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()

    # 3. Simulate gravity for a moment and check that the block stayed put.
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(args.sim_seconds)

    plant_context = plant.GetMyContextFromRoot(simulator.get_context())
    final_pose = plant.GetFreeBodyPose(plant_context, env.block)
    moved = np.linalg.norm(final_pose.translation() - env.block_start_pose.translation())
    print(f"Block moved {moved * 1000:.2f} mm during {args.sim_seconds:.1f} s of simulation.")

    if not args.no_wait:
        input("Viewer is open at the Meshcat URL above. Press Enter to quit.")


def print_summary(layout):
    p = layout.params
    print("Environment")
    print(f"  corner angle      : {p.corner_angle_deg:.1f} deg")
    print(f"  bay size          : {p.bay_size:.3f} m")
    print(f"  corridor width    : {p.corridor_width:.3f} m")
    print(f"  start roof length : {p.start_roof_length:.3f} m")
    print(f"  block starts in   : corridor {p.start_corridor}, "
          f"goal is corridor {p.goal_corridor()}")
    print(f"  wall pieces       : {len(layout.walls)}")
    for label, rect in [("block start", layout.block_start), ("block goal", layout.block_goal)]:
        x, y = rect.center
        print(f"  {label:<17} : x={x:.3f}, y={y:.3f}, yaw={np.degrees(rect.yaw):.1f} deg")


if __name__ == "__main__":
    main()
