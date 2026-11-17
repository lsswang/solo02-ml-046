"""Ordinary Dijkstra with a supplied set of known blocked cells."""
import heapq


def shortest(graph, start, goal, blocked_cells=()):
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
            return {'status':'ok','cost':cost,'edges':list(edges),'nodes':list(nodes)}
        for edge in adjacency[node]:
            after = cost + edge['cost']
            if after < best.get(edge['v'], float('inf')):
                best[edge['v']] = after
                heapq.heappush(heap,(after,edge['v'],edges+(edge['id'],),nodes+(edge['v'],)))
    return {'status':'infeasible','cost':None,'edges':[],'nodes':[]}
