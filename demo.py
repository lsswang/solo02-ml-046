"""End-to-end demonstrations.

Run with:  python demo.py

1. The original independent-cell inverse sensor update and ordinary Dijkstra
   with known blocked cells remain available unchanged.
2. A coupled scenario shows the full pipeline: exact joint inference over a
   class mixture with pair factors, then global chance-constrained routing on
   the returned distribution.  One legal report is changed while the graph,
   endpoints and threshold stay fixed; the frozen old distribution is
   replayed against the new joint model and shown to be no longer valid.
3. An external (non-model) correlated distribution drives the same planner
   directly, proving the route never rebuilds a map.
"""

import copy
import json
from pathlib import Path

import mapmodel
import routing


def show(title, payload):
    print(f"\n--- {title} ---")
    print(json.dumps(payload, indent=2, sort_keys=True))


def ordinary_demo(cases):
    print("=" * 70)
    print("LEGACY: independent inverse sensor + known-blocked Dijkstra")
    print("=" * 70)
    show("independent cell marginals",
         mapmodel.update_independent(**cases["ordinary_map"]))
    show("ordinary shortest path",
         routing.shortest(cases["graph"], 0, 2))
    show("same graph with cell 0 known blocked",
         routing.shortest(cases["graph"], 0, 2, blocked_cells=[0]))


def coupled_demo(cases):
    print("\n" + "=" * 70)
    print("COUPLED: exact joint inference -> global risk route")
    print("=" * 70)
    cfg = cases["coupling"]
    prior = cfg["prior"]
    base_ray = cfg["ray"]
    graph = cases["nonlocal_graph"]
    start, goal, threshold = cfg["start"], cfg["goal"], cfg["threshold"]

    def run(reading, tag):
        rays = [dict(base_ray, reading=reading)]
        result = mapmodel.infer(prior, rays)
        show(f"{tag}: inference (reading={reading})", {
            "status": result["status"],
            "log_evidence": result["log_evidence"],
            "class_probabilities": result["class_probabilities"],
            "cell_occupied": result["cell_occupied"],
            "distribution": result["distribution"],
        })
        if result["status"] != "ok":
            return result, None
        plan = routing.plan(graph, start, goal, threshold,
                            result["distribution"])
        show(f"{tag}: plan on the returned distribution", plan)
        return result, plan

    before, plan_before = run(cfg["before_reading"], "before")
    after, plan_after = run(cfg["after_reading"], "after")

    # Frozen-consumer check: evaluate the OLD plan under the NEW distribution.
    # Recomputing from the current joint model is the only valid move; merely
    # reprinting new marginals while still routing on the old distribution is
    # explicitly rejected.
    old_dist = before["distribution"]
    new_dist = after["distribution"]
    frozen = mapmodel.event(new_dist, plan_before["cells"])
    fresh = mapmodel.event(new_dist, plan_after["cells"])
    show("frozen old distribution replayed under the current posterior", {
        "old_route_edges": plan_before["edges"],
        "old_route_cells": plan_before["cells"],
        "old_route_success_under_new_distribution": frozen["probability"],
        "threshold": threshold,
        "old_route_still_acceptable":
            frozen["probability"] + 1e-12 >= threshold,
        "new_route_success_under_new_distribution": fresh["probability"],
        "note": "the planner consumed infer()'s distribution object directly; "
                "no second map was ever estimated",
    })


def external_demo(cases):
    print("\n" + "=" * 70)
    print("EXTERNAL DISTRIBUTION: planner accepts a map it did not produce")
    print("=" * 70)
    graph = cases["nonlocal_graph"]
    for name in ("external_anticorrelated", "external_correlated"):
        dist = cases[name]
        plan = routing.plan(graph, 0, 2, 0.8, dist)
        show(f"{name} -> plan(threshold=0.8)", plan)
        # Show why per-edge multiplication would be wrong on this joint law.
        marginals = mapmodel.event(dist, [0])["probability"], \
            mapmodel.event(dist, [1])["probability"]
        joint = mapmodel.event(dist, [0, 1])["probability"]
        show(f"{name} -> joint vs product-of-marginals", {
            "P(cell0 free)": marginals[0],
            "P(cell1 free)": marginals[1],
            "product (invalid here)": marginals[0] * marginals[1],
            "P(cell0 AND cell1 free) used by planner": joint,
        })


def missing_and_impossible_demo():
    print("\n" + "=" * 70)
    print("MISSING vs IMPOSSIBLE REPORTS")
    print("=" * 70)
    prior = {"n": 2, "class_weights": [1.0],
             "occupied": [[0.5, 0.5]], "pair_factors": []}
    missing_ray = {"cells": [0, 1],
                   "confusion": [[0.4, 0.5, 0.1], [0.0, 0.8, 0.2],
                                 [0.0, 0.0, 1.0]],
                   "reading": None}
    missing = mapmodel.infer(prior, [missing_ray])
    show("all readings missing -> evidence 1, posterior is the prior", {
        "status": missing["status"],
        "log_evidence": missing["log_evidence"],
        "posterior": missing["distribution"]["probabilities"],
    })
    impossible_ray = {"cells": [0, 1],
                      "confusion": [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                                    [0.0, 1.0, 0.0]],
                      "reading": 0}
    impossible = mapmodel.infer(prior, [impossible_ray])
    show("legal reading with zero model mass -> impossible_evidence", {
        "status": impossible["status"],
        "log_evidence": impossible["log_evidence"],
        "distribution": impossible["distribution"],
    })


if __name__ == "__main__":
    cases = json.loads(
        (Path(__file__).parent / "fixtures" / "cases.json").read_text())
    cases = copy.deepcopy(cases)
    ordinary_demo(cases)
    coupled_demo(cases)
    external_demo(cases)
    missing_and_impossible_demo()
