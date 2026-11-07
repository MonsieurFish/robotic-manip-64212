"""Show an iiwa next to a table in Meshcat."""

from manipulation.station import LoadScenario, MakeHardwareStation
from pydrake.all import (
    ConstantVectorSource,
    DiagramBuilder,
    Simulator,
    StartMeshcat,
)

######################################################################
## The scene: an iiwa, a table
######################################################################


def scenario_yaml() -> str:
    return """directives:
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf
    default_joint_positions:
        iiwa_joint_1: [-1.57]
        iiwa_joint_2: [0.1]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.2]
        iiwa_joint_5: [0]
        iiwa_joint_6: [1.6]
        iiwa_joint_7: [0]
- add_weld:
    parent: world
    child: iiwa::iiwa_link_0
    X_PC:
        translation: [0, -0.5, 0]
        rotation: !Rpy { deg: [0, 0, 180] }
- add_model:
    name: wsg
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy { deg: [90, 0, 90] }
- add_model:
    name: table
    file: package://manipulation/table.sdf
# Weld the table down so it doesn't fall. The table is a 0.1 m thick slab
# centered on its origin, so lower it by half that to put its top at z = 0,
# level with the iiwa's base.
- add_weld:
    parent: world
    child: table::table_link
    X_PC:
        translation: [0, 0, -0.05]

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


def build_static_system(meshcat):
    """The scene, with the arm held at its default joint positions."""
    builder = DiagramBuilder()
    station = builder.AddSystem(
        MakeHardwareStation(LoadScenario(data=scenario_yaml()), meshcat=meshcat)
    )
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

    return builder.Build()


def show_scene(system, duration: float = 2.0):
    """Simulate a system in real time, so it can be watched in Meshcat."""
    simulator = Simulator(system)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(duration)
    return simulator


def main():
    meshcat = StartMeshcat()
    show_scene(build_static_system(meshcat))
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
