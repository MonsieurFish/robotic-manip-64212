"""
The three planners we are comparing (proposal, §Algorithms):

    baseline_plan   — never regrasp.                   (expected to fail often)
    greedy_plan     — regrasp only when stuck,
                      pick the regrasp that makes the
                      most local move-only progress.
    global_plan     — Dijkstra over the whole graph,
                      regrasp cost included.

All three return a `Plan` with the node sequence, the ordered list of edges
used, the total cost, and a success flag. Planners operate on the
`RegraspMap` interface — none of them touch Drake, geometry, or trajectory
code. When the real regrasp map is populated by the environment / trajectory
code, these planners drop in unchanged.
"""

import heapq
from dataclasses import dataclass
from typing import Optional

from regrasp_map import RegraspMap


@dataclass
class Plan:
    success: bool
    nodes: list           # node indices along the path, including start and goal
    edges: list           # RegraspEdge sequence, length len(nodes) - 1
    total_cost: float
    regrasp_count: int

    @classmethod
    def failed(cls):
        return cls(success=False, nodes=[], edges=[], total_cost=float("inf"),
                   regrasp_count=0)

    def describe(self, m: RegraspMap) -> str:
        if not self.success:
            return "FAILED: no plan found"
        parts = [m.node_label(self.nodes[0])]
        for e in self.edges:
            parts.append(f" --[{e.kind} c={e.cost:g}]--> {m.node_label(e.dst)}")
        parts.append(
            f"\n  total cost = {self.total_cost:g}, regrasps = {self.regrasp_count}"
        )
        return "".join(parts)


# ----------------------------------------------------------------------
# 1. Baseline: no regrasping allowed
# ----------------------------------------------------------------------
def baseline_plan(m: RegraspMap) -> Plan:
    """BFS using MOVE edges only. The proposal expects this to fail on most mazes."""
    return _dijkstra(m, allow_regrasp=False)


# ----------------------------------------------------------------------
# 2. Greedy: hold the grasp as long as possible; when stuck, pick the regrasp
#    whose destination gets closest (in move-only distance) to a goal.
# ----------------------------------------------------------------------
def greedy_plan(m: RegraspMap) -> Plan:
    """
    Strategy:
      while not at a goal:
        - take move edges as far as possible (shortest move-only path toward goal)
        - if a goal is reachable by moves alone, take that path and finish.
        - otherwise consider every reachable node from which a regrasp exits;
          pick the regrasp whose destination has the smallest move-only
          distance to any goal; take it; repeat.
    """
    current = m.start
    nodes = [current]
    edges = []
    total_cost = 0.0
    regrasp_count = 0
    visited_regrasps = set()  # (from, to) pairs already used, to avoid cycles

    while not m.is_goal(current):
        # 1. Try to move all the way to a goal from here.
        move_dists, move_prev = _move_only_distances(m, current)
        best_goal = _closest(move_dists, m.goals)
        if best_goal is not None:
            path = _reconstruct(move_prev, current, best_goal)
            path_edges = _edges_along(m, path, kind="move")
            nodes.extend(path[1:])
            edges.extend(path_edges)
            total_cost += sum(e.cost for e in path_edges)
            current = best_goal
            break

        # 2. No goal reachable by moves. From any node reachable by moves,
        #    find the cheapest regrasp whose destination gets closest to a goal.
        best = None   # (heuristic, cost_through_regrasp, from_node, edge)
        for intermediate, d_here in move_dists.items():
            for e in m.out_edges(intermediate):
                if e.kind != "regrasp":
                    continue
                if (e.src, e.dst) in visited_regrasps:
                    continue
                d_after, _ = _move_only_distances(m, e.dst)
                closest = _closest(d_after, m.goals, return_distance=True)
                heuristic = closest if closest is not None else min(
                    (d_after.get(n, float("inf")) for n in range(len(m.nodes))),
                    default=float("inf"),
                )
                candidate = (heuristic, d_here + e.cost, intermediate, e)
                if best is None or candidate < best:
                    best = candidate

        if best is None:
            return Plan.failed()

        _, _, intermediate, regrasp_edge = best
        path = _reconstruct(move_prev, current, intermediate)
        path_edges = _edges_along(m, path, kind="move")
        nodes.extend(path[1:])
        edges.extend(path_edges)
        total_cost += sum(e.cost for e in path_edges)

        nodes.append(regrasp_edge.dst)
        edges.append(regrasp_edge)
        total_cost += regrasp_edge.cost
        regrasp_count += 1
        visited_regrasps.add((regrasp_edge.src, regrasp_edge.dst))
        current = regrasp_edge.dst

    return Plan(success=m.is_goal(current), nodes=nodes, edges=edges,
                total_cost=total_cost, regrasp_count=regrasp_count)


# ----------------------------------------------------------------------
# 3. Global: Dijkstra over the whole graph.
# ----------------------------------------------------------------------
def global_plan(m: RegraspMap) -> Plan:
    return _dijkstra(m, allow_regrasp=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _dijkstra(m: RegraspMap, allow_regrasp: bool) -> Plan:
    dist = {m.start: 0.0}
    prev_edge = {}
    pq = [(0.0, m.start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if m.is_goal(u):
            return _build_plan_from_prev(m, u, dist, prev_edge)
        for e in m.out_edges(u):
            if not allow_regrasp and e.kind == "regrasp":
                continue
            nd = d + e.cost
            if nd < dist.get(e.dst, float("inf")):
                dist[e.dst] = nd
                prev_edge[e.dst] = e
                heapq.heappush(pq, (nd, e.dst))
    return Plan.failed()


def _move_only_distances(m: RegraspMap, src: int):
    """Shortest paths from `src` using MOVE edges only. Returns (dist, prev)."""
    dist = {src: 0.0}
    prev_edge = {}
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for e in m.move_out_edges(u):
            nd = d + e.cost
            if nd < dist.get(e.dst, float("inf")):
                dist[e.dst] = nd
                prev_edge[e.dst] = e
                heapq.heappush(pq, (nd, e.dst))
    return dist, prev_edge


def _closest(dists: dict, goals: frozenset,
             return_distance: bool = False) -> Optional[object]:
    reachable = [(dists[g], g) for g in goals if g in dists]
    if not reachable:
        return None
    reachable.sort()
    return reachable[0][0] if return_distance else reachable[0][1]


def _reconstruct(prev_edge: dict, src: int, dst: int) -> list:
    if src == dst:
        return [src]
    path = [dst]
    while path[-1] != src:
        e = prev_edge[path[-1]]
        path.append(e.src)
    return list(reversed(path))


def _edges_along(m: RegraspMap, path: list, kind: Optional[str] = None) -> list:
    out = []
    for u, v in zip(path, path[1:]):
        for e in m.out_edges(u):
            if e.dst == v and (kind is None or e.kind == kind):
                out.append(e)
                break
    return out


def _build_plan_from_prev(m: RegraspMap, goal: int, dist, prev_edge) -> Plan:
    path_nodes = [goal]
    path_edges = []
    while path_nodes[-1] != m.start:
        e = prev_edge[path_nodes[-1]]
        path_edges.append(e)
        path_nodes.append(e.src)
    path_nodes.reverse()
    path_edges.reverse()
    return Plan(
        success=True,
        nodes=path_nodes,
        edges=path_edges,
        total_cost=dist[goal],
        regrasp_count=sum(1 for e in path_edges if e.kind == "regrasp"),
    )
