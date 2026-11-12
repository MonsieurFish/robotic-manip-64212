"""
The regrasp map: a directed weighted graph the planners search over.

A node represents "the block is somewhere in the maze AND the robot is
holding it with some grasp (or has set it down)". The exact node
discretization (how finely we bucket grasp locations and block poses)
is Jinjou/Hailey's call; this file does NOT commit to one. All the
planner needs is the graph interface below.

Two kinds of edges:
  - "move":    the robot transports the block without releasing it.
               The grasp is the same at both endpoints.
  - "regrasp": the robot sets the block down, releases, and picks it up
               with a different grasp. Expensive.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RegraspNode:
    label: str  # free-form, for debugging/printing


@dataclass(frozen=True)
class RegraspEdge:
    src: int       # index into RegraspMap.nodes
    dst: int
    cost: float
    kind: str      # "move" or "regrasp"

    def __post_init__(self):
        if self.kind not in ("move", "regrasp"):
            raise ValueError(f"edge.kind must be 'move' or 'regrasp', got {self.kind!r}")


@dataclass
class RegraspMap:
    nodes: list = field(default_factory=list)   # RegraspNode
    edges: list = field(default_factory=list)   # RegraspEdge
    start: int = 0
    goals: frozenset = field(default_factory=frozenset)   # node indices

    # ---------- queries the planners use ----------
    def out_edges(self, node: int):
        return [e for e in self.edges if e.src == node]

    def move_out_edges(self, node: int):
        return [e for e in self.edges if e.src == node and e.kind == "move"]

    def is_goal(self, node: int) -> bool:
        return node in self.goals

    def node_label(self, node: int) -> str:
        return self.nodes[node].label


# ======================================================================
# A toy map, for testing the planners without the real maze geometry.
# ======================================================================

def toy_map() -> RegraspMap:
    """
    A 6-node map where:
      - the baseline (no regrasp) cannot reach the goal,
      - greedy finds a plan with 1 regrasp but costly moves (total 26),
      - global finds a plan with 2 regrasps but cheaper moves (total 12).

    Topology (M = move, R = regrasp):

           start -M-> A -M-> B
                      R (cost 5)
                      v
                      C -M (cost 1)-> D -R (cost 5)-> goal
                      C -M (cost 20)-------------------> goal

    Baseline: cannot reach the goal, since B has no outgoing move edge.
    Greedy: once at C, the one-move hop (cost 20) directly reaches the goal
      and looks locally best, so greedy takes it. One regrasp, total 26.
    Global: willing to pay a second regrasp (C-M->D-R->goal, cost 1+5=6) to
      avoid the expensive move. Two regrasps, total 12.
    """
    nodes = [
        RegraspNode("start"),       # 0
        RegraspNode("A_grasp1"),    # 1
        RegraspNode("B_grasp1"),    # 2
        RegraspNode("C_grasp2"),    # 3
        RegraspNode("D_grasp2"),    # 4
        RegraspNode("goal"),        # 5
    ]
    edges = [
        RegraspEdge(0, 1, cost=1.0,  kind="move"),
        RegraspEdge(1, 2, cost=1.0,  kind="move"),
        RegraspEdge(1, 3, cost=5.0,  kind="regrasp"),
        RegraspEdge(3, 4, cost=1.0,  kind="move"),
        RegraspEdge(3, 5, cost=20.0, kind="move"),
        RegraspEdge(4, 5, cost=5.0,  kind="regrasp"),
    ]
    return RegraspMap(nodes=nodes, edges=edges, start=0, goals=frozenset({5}))
