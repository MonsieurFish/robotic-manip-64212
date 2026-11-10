"""
All the numbers that define one corridor environment, kept in one place.

Units: meters and degrees.
World frame: the robot base (the iiwa, added later) sits at the origin,
+z points up, and the top of the floor is at z = 0.

The default environment is the two-segment corridor from the proposal (Fig 1):
a start pocket and a goal pocket, each at the dead end of a corridor, joined
by a square "bay" at the corner. Seen from above:

                 corridor B
       dead end ┌──────────────────┬──────────┐
                │  (block goal)        bay    │
                └──────────────────┐          │
                                   └──┐    ┌──┘
                                      │    │
                                      │    │  corridor A
                                      │░░░░│  (block start, ░ = roof)
                             dead end └────┘

In the actual world frame the whole shape is rotated so that both corridors
open back toward the robot (see `bay_center_xy` below).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class EnvironmentParams:
    # ------------------------------------------------------------------
    # The block being carried
    # ------------------------------------------------------------------
    block_length: float = 0.30  # long axis; always lies along the corridor
    block_width: float = 0.05   # the gripper pinches across this dimension
    block_height: float = 0.05
    block_mass: float = 0.3     # kg

    # ------------------------------------------------------------------
    # Corridors (there are two: "A" and "B")
    # ------------------------------------------------------------------
    # Width: block (5 cm) + room for the two Schunk WSG fingers (1.2 cm thick
    # each) opened a little wider than the block -> about 9 cm minimum.
    # 10 cm leaves a bit of clearance and is still far too narrow for the
    # 30 cm block to turn around, so it must travel lengthwise.
    corridor_width: float = 0.10

    # Measured from the CENTER of the bay out to the corridor's dead end.
    corridor_length: float = 0.55

    # ------------------------------------------------------------------
    # The square bay at the corner
    # ------------------------------------------------------------------
    bay_size: float = 0.34  # side length of the square

    # Angle between the two corridors, as seen from the bay center.
    #   90  -> L-shaped corner (proposal Fig 1)
    #   180 -> the corridors form one straight line
    corner_angle_deg: float = 90.0

    # ------------------------------------------------------------------
    # Walls and roof
    # ------------------------------------------------------------------
    # Wall height: taller than the block, but short enough that the WSG
    # fingers (8.2 cm long) can reach down to the block while the gripper's
    # wide palm (14.6 cm, wider than the corridor) stays above the walls.
    wall_height: float = 0.08
    wall_thickness: float = 0.02

    # How much of the START corridor is covered by a roof, measured from its
    # dead end. The roof sits on top of the walls, so the gripper cannot
    # reach the part of the block underneath it. This limits which grasps
    # are possible at the start. 0 means no roof.
    start_roof_length: float = 0.12
    roof_thickness: float = 0.01

    # ------------------------------------------------------------------
    # Where the block starts and ends
    # ------------------------------------------------------------------
    start_corridor: str = "A"  # "A" or "B"; the goal is in the other corridor

    # Gap between the back of the block and the dead-end wall, at both the
    # start and the goal.
    block_gap_to_dead_end: float = 0.02

    # ------------------------------------------------------------------
    # Task rules (not physical objects, just limits the planner must obey)
    # ------------------------------------------------------------------
    # The bottom of the block may never rise more than this above the floor.
    # This keeps the block's top below the wall tops, so the robot cannot
    # just lift the block out of the corridor.
    max_block_lift: float = 0.02

    # The roof must leave at least this much of the block uncovered so that
    # some grasp is always possible. (The WSG palm is 7.25 cm long in this
    # direction, plus some margin.)
    min_uncovered_block_length: float = 0.10

    # ------------------------------------------------------------------
    # Placement in the world
    # ------------------------------------------------------------------
    # The bay center, in world coordinates. Both corridors open back toward
    # the robot at the origin, symmetric about the x-axis.
    bay_center_xy: tuple = (0.58, 0.0)

    # Floor: a square slab centered on the origin, top surface at z = 0.
    floor_size: float = 2.0
    floor_thickness: float = 0.02

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def goal_corridor(self) -> str:
        return "B" if self.start_corridor == "A" else "A"

    def max_start_roof_length(self) -> float:
        """Longest start roof that still leaves `min_uncovered_block_length`
        of the block exposed for grasping."""
        return (self.block_gap_to_dead_end
                + self.block_length
                - self.min_uncovered_block_length)

    def validate(self) -> None:
        """Raise a ValueError if the numbers don't make a sensible environment."""
        checks = [
            (self.corridor_width > self.block_width,
             "corridor_width must be larger than block_width"),
            (self.wall_height > self.block_height + self.max_block_lift,
             "walls must be taller than the block at its highest allowed lift"),
            (self.start_corridor in ("A", "B"),
             "start_corridor must be 'A' or 'B'"),
            (0 < self.corner_angle_deg <= 180,
             "corner_angle_deg must be between 0 and 180"),
            (0 <= self.start_roof_length <= self.max_start_roof_length(),
             f"start_roof_length must be between 0 and "
             f"{self.max_start_roof_length():.3f} m, so a grasp stays possible"),
        ]
        for is_ok, message in checks:
            if not is_ok:
                raise ValueError(message)


# ----------------------------------------------------------------------
# Randomized environments
# ----------------------------------------------------------------------
# Ranges for the randomized variables listed in the proposal:
#   1. roof cover at the start, and the bay size
#   2. the corner angle
#   3. which corridor the block starts in
BAY_SIZE_RANGE = (0.30, 0.42)       # meters
CORNER_ANGLE_RANGE = (60.0, 150.0)  # degrees; below ~60 the corridors start to merge


def sample_random_params(seed=None) -> EnvironmentParams:
    """Return default params with the randomized variables re-drawn."""
    rng = np.random.default_rng(seed)
    params = EnvironmentParams()

    params.bay_size = rng.uniform(*BAY_SIZE_RANGE)
    params.corner_angle_deg = rng.uniform(*CORNER_ANGLE_RANGE)
    params.start_corridor = str(rng.choice(["A", "B"]))
    params.start_roof_length = rng.uniform(0.0, params.max_start_roof_length())

    params.validate()
    return params
