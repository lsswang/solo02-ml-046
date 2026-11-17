"""Route planning under an externally supplied joint map distribution.

The legacy :func:`shortest` solves ordinary Dijkstra with a known blocked cell
set.  :func:`plan` instead accepts any joint map distribution (it never
rebuilds or re-estimates a map) and globally minimises route cost subject to
the *joint* success event -- every traversed cell being simultaneously free
-- meeting a probability threshold.
"""

import heapq

from mapmodel import validate_distribution, _is_int, _is_finite_number


def _validate_graph(graph, n):
    if not isinstance(graph, dict):
        raise ValueError("graph must be an object")
    vertices = graph.get("vertices")
    if not _is_int(vertices) or not (1 <= vertices <= 10):
        raise ValueError("graph.vertices must be an integer in 1..10")
    edges = graph.get("edges")
    if not isinstance(edges, (list, tuple)) or len(edges) > 24:
        raise ValueError("graph.edges must be a list of at most 24 edges")
    seen_ids = set()
    normalised = []
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("each edge must be an object")
        edge_id = edge.get("id")
        if not isinstance(edge_id, str) or edge_id == "":
            raise ValueError("edge id must be a unique non-empty string")
        if edge_id in seen_ids:
            raise ValueError(f"duplicate edge id {edge_id!r}")
        seen_ids.add(edge_id)
        u, v = edge.get("u"), edge.get("v")
        if not _is_int(u) or not _is_int(v) \
                or not (0 <= u < vertices) or not (0 <= v < vertices):
            raise ValueError("edge endpoints must be valid vertex indices")
        cost = edge.get("cost")
        if not _is_int(cost) or not (1 <= cost <= 9):
            raise ValueError("edge cost must be an integer in 1..9")
        cells = edge.get("cells", [])
        if not isinstance(cells, (list, tuple)):
            raise ValueError("edge cells must be a list of cell indices")
        seen_cells = set()
        for cell in cells:
            if not _is_int(cell) or not (0 <= cell < n):
                raise ValueError("edge cells must be valid map cell indices")
            if cell in seen_cells:
                raise ValueError(f"edge {edge_id!r} repeats cell {cell}")
            seen_cells.add(cell)
        mask = 0
        for cell in cells:
            mask |= 1 << cell
        normalised.append({"id": edge_id, "u": u, "v": v, "cost": cost,
                           "cells": list(cells), "mask": mask})
    return vertices, normalised


def _free_mass_table(probs, n):
    """free_mass[m] = P(all cells in m are free) under the joint distribution.

    A subset zeta transform gives F[t] = sum_{s subset t} P(mask=s); then
    free_mass[m] = F[full XOR m].  This is O(n*2^n) instead of O(4^n) and
    never factors the joint into per-cell marginals.
    """
    size = 1 << n
    subset_sum = list(probs)
    for bit in range(n):
        step = 1 << bit
        for mask in range(size):
            if mask & step:
                subset_sum[mask] += subset_sum[mask ^ step]
    full = size - 1
    return [subset_sum[full ^ mask] for mask in range(size)]


def plan(graph, start, goal, threshold, distribution):
    """Cost-optimal route whose joint success probability meets *threshold*.

    Edge costs are strictly positive and traversed cells only accumulate, so
    an optimal route contains no cycle: simple paths suffice.  All simple
    paths are enumerated (with explicit parallel-edge choices), which is
    exhaustive for V <= 10 and E <= 24 and guarantees the global optimum.
    """
    n, probs = validate_distribution(distribution)
    vertices, edges = _validate_graph(graph, n)
    if not _is_int(start) or not (0 <= start < vertices) \
            or not _is_int(goal) or not (0 <= goal < vertices):
        raise ValueError("start and goal must be valid vertex indices")
    if not _is_finite_number(threshold) or not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be a finite number in [0, 1]")

    free_mass = _free_mass_table(probs, n)
    adjacency = [[] for _ in range(vertices)]
    for edge in edges:
        adjacency[edge["u"]].append(edge)

    # Zero-edge route: the empty success event has probability 1.
    if start == goal:
        return {"status": "ok", "edges": [], "nodes": [start],
                "cells": [], "cost": 0, "success_probability": 1.0}

    best = None  # (cost, edge_ids, nodes, union_mask, success_probability)

    def dfs(node, visited, cell_mask, cost, edge_ids, nodes):
        nonlocal best
        if node == goal:
            success = free_mass[cell_mask]
            if success + 1e-12 >= threshold and (best is None
                                                 or cost < best[0]):
                best = (cost, list(edge_ids), list(nodes), cell_mask, success)
            return
        # Positive costs: a route already as expensive as the incumbent can
        # never improve on it, so prune the branch.
        if best is not None and cost >= best[0]:
            return
        for edge in adjacency[node]:
            nxt = edge["v"]
            if visited & (1 << nxt):
                continue
            dfs(nxt, visited | (1 << nxt), cell_mask | edge["mask"],
                cost + edge["cost"], edge_ids + [edge["id"]],
                nodes + [nxt])

    dfs(start, 1 << start, 0, 0, [], [start])

    if best is None:
        return {"status": "infeasible", "edges": [], "nodes": [],
                "cells": [], "cost": None, "success_probability": None}

    cost, edge_ids, nodes, cell_mask, success = best
    cells = [i for i in range(n) if (cell_mask >> i) & 1]
    return {"status": "ok", "edges": edge_ids, "nodes": nodes,
            "cells": cells, "cost": cost, "success_probability": success}


def shortest(graph, start, goal, blocked_cells=()):
    """Ordinary Dijkstra with a supplied set of known blocked cells."""
    blocked = set(blocked_cells)
    adjacency = [[] for _ in range(graph['vertices'])]
    for edge in graph['edges']:
        if not blocked.intersection(edge['cells']):
            adjacency[edge['u']].append(edge)
    heap = [(0, start, (), (start,))]
    best = {start: 0}
    while heap:
        cost, node, edges, nodes = heapq.heappop(heap)
        if cost != best.get(node):
            continue
        if node == goal:
            return {'status': 'ok', 'cost': cost, 'edges': list(edges),
                    'nodes': list(nodes)}
        for edge in adjacency[node]:
            after = cost + edge['cost']
            if after < best.get(edge['v'], float('inf')):
                best[edge['v']] = after
                heapq.heappush(heap, (after, edge['v'],
                                      edges + (edge['id'],),
                                      nodes + (edge['v'],)))
    return {'status': 'infeasible', 'cost': None, 'edges': [], 'nodes': []}
