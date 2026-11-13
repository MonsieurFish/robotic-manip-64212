"""Show an iiwa above the maze environment in Meshcat."""

import sys
from pathlib import Path

from manipulation.station import LoadScenario, MakeHardwareStation
from pydrake.all import (
    ConstantVectorSource,
    DiagramBuilder,
    Simulator,
    StartMeshcat,
)

# The maze lives in ../environment, whose modules import each other by bare
# name (e.g. `from maze_layout import ...`), so put that folder on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment"))

from drake_environment import add_environment_to_plant  # noqa: E402
from environment_params import EnvironmentParams  # noqa: E402
from maze_layout import make_two_segment_layout  # noqa: E402

######################################################################
## The scene: an iiwa with a gripper; the maze is added in build_static_system
######################################################################


def scenario_yaml() -> str:
    return """directives:
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf
    default_joint_positions:
        iiwa_joint_1: [0]
        iiwa_joint_2: [0.1]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.2]
        iiwa_joint_5: [0]
        iiwa_joint_6: [1.6]
        iiwa_joint_7: [0]
- add_weld:
    parent: world
    child: iiwa::iiwa_link_0
# The maze environment assumes the iiwa base at the world origin, facing +x
# (the maze's bay is at x = 0.58 m).
    X_PC:
        translation: [0, 0, 0]
- add_model:
    name: wsg
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy { deg: [90, 0, 90] }
# No table: the maze environment brings its own floor, with its top at z = 0.

model_drivers:
    iiwa: !IiwaDriver
        control_mode: position_only
        hand_model_name: wsg
    wsg: !SchunkWsgDriver {}
"""

# How far apart the gripper fingers are held, in meters.
OPENED = 0.05

######################################################################
## Render the scene
######################################################################


def add_station(builder, meshcat, params=None):
    """Add the hardware station (iiwa + gripper + maze) to `builder`.

    Returns (station, env); `env` holds the maze's block body and its start
    and goal poses (see environment/drake_environment.py). The station's
    iiwa.position and wsg.position inputs are left for the caller to connect.
    """
    if params is None:
        params = EnvironmentParams()
    layout = make_two_segment_layout(params)
    env_holder = {}

    def add_maze(parser):
        # Runs inside MakeHardwareStation, just before the plant is finalized.
        env_holder["env"] = add_environment_to_plant(parser.plant(), layout)

    station = builder.AddSystem(
        MakeHardwareStation(
            LoadScenario(data=scenario_yaml()),
            meshcat=meshcat,
            parser_prefinalize_callback=add_maze,
        )
    )
    return station, env_holder["env"]


def build_static_system(meshcat, params=None):
    """The scene, with the arm held at its default joint positions.

    Returns (diagram, env).
    """
    builder = DiagramBuilder()
    station, env = add_station(builder, meshcat, params)
    plant = station.GetSubsystemByName("plant")

    # The iiwa driver needs a position command; hold the default pose.
    iiwa = plant.GetModelInstanceByName("iiwa")
    hold_arm = builder.AddSystem(ConstantVectorSource(plant.GetDefaultPositions(iiwa)))
    builder.Connect(hold_arm.get_output_port(), station.GetInputPort("iiwa.position"))

    # Start the fingers open and keep commanding them open.
    plant.SetDefaultPositions(
        plant.GetModelInstanceByName("wsg"), [-OPENED / 2, OPENED / 2]
    )
    hold_fingers = builder.AddSystem(ConstantVectorSource([OPENED]))
    builder.Connect(
        hold_fingers.get_output_port(), station.GetInputPort("wsg.position")
    )

    return builder.Build(), env


def show_scene(system, duration: float = 2.0):
    """Simulate a system in real time, so it can be watched in Meshcat."""
    simulator = Simulator(system)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(duration)
    return simulator


def main():
    meshcat = StartMeshcat()
    system, env = build_static_system(meshcat)
    show_scene(system)
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
