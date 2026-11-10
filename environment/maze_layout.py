"""
The floor plan of the environment, seen from above (2D).

What's on the floor plan:
  - open areas : the corridors and the bay, i.e. where the block may travel
                 (rectangles, `Rect`)
  - walls      : built automatically around the outline of the open areas
                 (triangles, see `make_walls_around`)
  - roofs      : covered dead ends of the start and goal corridors (rectangles)
  - the block's footprint at the start and at the goal (rectangles)

Nothing here depends on Drake, so the layout can be checked with plain Python
(see plot_top_down.py). drake_environment.py turns these 2D shapes into 3D
solids by giving each one a height.
"""

from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

from environment_params import EnvironmentParams


# ======================================================================
# Basic shapes
# ======================================================================

@dataclass
class Rect:
    """A rectangle lying on the floor."""
    center: np.ndarray  # (x, y) in meters
    length: float       # size along the `yaw` direction
    width: float        # size perpendicular to the `yaw` direction
    yaw: float          # radians, counter-clockwise from the +x axis

    def corners(self) -> np.ndarray:
        """The four corners in counter-clockwise order, as a (4, 2) array."""
        along = unit_vector(self.yaw) * self.length / 2
        across = unit_vector(self.yaw + np.pi / 2) * self.width / 2
        c = np.asarray(self.center, dtype=float)
        return np.array([c - along - across,
                         c + along - across,
                         c + along + across,
                         c - along + across])

    def as_polygon(self) -> Polygon:
        return Polygon(self.corners())


def unit_vector(angle: float) -> np.ndarray:
    """2D unit vector pointing in direction `angle` (radians)."""
    return np.array([np.cos(angle), np.sin(angle)])


@dataclass
class MazeLayout:
    """Everything needed to build the environment."""
    params: EnvironmentParams

    open_areas: dict   # name -> Rect   (e.g. "bay", "corridor_A")
    walls: dict        # name -> (3, 2) array of triangle corners (e.g. "wall_00")
    roofs: dict        # name -> Rect   ("start_roof", "goal_roof")

    # All open areas merged into one shape. Handy for quick 2D checks,
    # e.g. `layout.open_space.contains(block_footprint.as_polygon())`.
    open_space: Polygon

    # Direction each corridor points, from the bay center toward its
    # dead end (radians).
    corridor_heading: dict  # "A" / "B" -> angle

    block_start: Rect  # footprint of the block at the start
    block_goal: Rect   # footprint of the block at the goal


# ======================================================================
# The two-segment corridor from the proposal
# ======================================================================

def make_two_segment_layout(params: EnvironmentParams) -> MazeLayout:
    """Build the floor plan for one bay with two dead-end corridors."""
    params.validate()
    bay_center = np.array(params.bay_center_xy, dtype=float)

    # Both corridors open back toward the robot (the -x direction), split
    # evenly on either side of it by the corner angle.
    half_angle = np.radians(params.corner_angle_deg) / 2
    corridor_heading = {
        "A": np.pi - half_angle,
        "B": np.pi + half_angle,
    }

    # --- Open areas: the bay plus the two corridors ---
    # The bay square is lined up with corridor A, as in the proposal figure.
    open_areas = {
        "bay": Rect(center=bay_center,
                    length=params.bay_size,
                    width=params.bay_size,
                    yaw=corridor_heading["A"]),
    }
    for name, heading in corridor_heading.items():
        # Each corridor starts at the bay center (that part is hidden inside
        # the bay) and runs out to its dead end.
        open_areas[f"corridor_{name}"] = Rect(
            center=bay_center + unit_vector(heading) * params.corridor_length / 2,
            length=params.corridor_length,
            width=params.corridor_width,
            yaw=heading,
        )

    open_space = merge_open_areas(open_areas.values())
    walls = make_walls_around(open_space, params.wall_thickness)

    # --- Block start and goal, each tucked into a dead end ---
    start_heading = corridor_heading[params.start_corridor]
    goal_heading = corridor_heading[params.goal_corridor()]
    block_start = block_in_pocket(bay_center, start_heading, params)
    block_goal = block_in_pocket(bay_center, goal_heading, params)

    for label, block in [("start", block_start), ("goal", block_goal)]:
        if not open_space.contains(block.as_polygon()):
            raise ValueError(f"The block's {label} position overlaps a wall.")

    # --- Roofs over the dead ends of the start and goal corridors ---
    roofs = {}
    if params.start_roof_length > 0:
        roofs["start_roof"] = roof_over_pocket(
            bay_center, start_heading, params.start_roof_length, params)
    if params.goal_roof_length > 0:
        roofs["goal_roof"] = roof_over_pocket(
            bay_center, goal_heading, params.goal_roof_length, params)

    return MazeLayout(
        params=params,
        open_areas=open_areas,
        walls=walls,
        roofs=roofs,
        open_space=open_space,
        corridor_heading=corridor_heading,
        block_start=block_start,
        block_goal=block_goal,
    )


def block_in_pocket(bay_center, heading, params: EnvironmentParams) -> Rect:
    """Footprint of the block lying lengthwise at a corridor's dead end."""
    distance_from_bay_center = (params.corridor_length
                                - params.block_gap_to_dead_end
                                - params.block_length / 2)
    return Rect(
        center=bay_center + unit_vector(heading) * distance_from_bay_center,
        length=params.block_length,
        width=params.block_width,
        yaw=heading,
    )


def roof_over_pocket(bay_center, heading, covered_length,
                     params: EnvironmentParams) -> Rect:
    """Roof covering the last `covered_length` of a corridor.

    It is wide enough to rest on both side walls, and long enough to also
    cover the dead-end wall.
    """
    roof_length = covered_length + params.wall_thickness
    # The roof's far edge lines up with the outside of the dead-end wall.
    far_edge = params.corridor_length + params.wall_thickness
    return Rect(
        center=bay_center + unit_vector(heading) * (far_edge - roof_length / 2),
        length=roof_length,
        width=params.corridor_width + 2 * params.wall_thickness,
        yaw=heading,
    )


# ======================================================================
# Walls
# ======================================================================

def merge_open_areas(rects) -> Polygon:
    """Merge rectangles into one outline shape."""
    merged = unary_union([r.as_polygon() for r in rects])
    if not isinstance(merged, Polygon):
        raise ValueError("The open areas do not form one connected region.")
    # Remove tiny leftover vertices that the merge sometimes creates.
    return merged.simplify(1e-6)


def make_walls_around(open_space: Polygon, thickness: float) -> dict:
    """Build walls of the given thickness that hug the outline of the open space.

    How it works:
      1. Grow the open space outward by `thickness`, keeping corners sharp.
      2. Subtract the open space itself. What's left is a ring of wall,
         exactly `thickness` wide, all the way around.
      3. Cut that ring into triangles. Drake's collision shapes have to be
         convex, and triangles always are. (Each triangle becomes a 3-sided
         prism standing on the floor; together they look like one wall.)

    This works for any corner angle and for any number of corridors, so
    composed mazes can reuse it unchanged.

    Returns: name -> (3, 2) array holding the triangle's corners (x, y).
    """
    grown = open_space.buffer(thickness, join_style="mitre")
    wall_ring = grown.difference(open_space)
    triangles = shapely.constrained_delaunay_triangles(wall_ring).geoms

    walls = {}
    for triangle in triangles:
        if triangle.area < 1e-9:
            continue  # skip slivers left over from rounding error
        corners = np.array(triangle.exterior.coords)[:3]  # last point repeats the first
        walls[f"wall_{len(walls):02d}"] = corners
    return walls
