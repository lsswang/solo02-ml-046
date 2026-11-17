"""Independent brute-force verification for mapmodel and routing.

The checks here deliberately re-derive every quantity from first principles
instead of reusing the library's transforms:

* priors are rebuilt by literally multiplying class weight, Bernoulli masses
  and pair-table entries for every (z, mask) pair;
* likelihoods are evaluated from the first-hit rule map by map;
* event masses are summed over masks satisfying the literal cell constraints;
* every *simple path* of the road graph is enumerated independently (with
  parallel edges kept distinct), its traversed-cell union and joint success
  mass computed directly, and the global cost minimum re-derived.

Run:  python verify.py
"""

import itertools
import math

import mapmodel
import routing


PASS = 0
FAIL = 0


def check(name, got, expected, tol=1e-10):
    global PASS, FAIL
    if isinstance(expected, float) or isinstance(got, float):
        if math.isinf(expected) or math.isinf(got):
            ok = got == expected
        else:
            ok = got is not None and abs(got - expected) <= tol
    else:
        ok = got == expected
    if ok:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}: got {got!r}, expected {expected!r}")


# --------------------------------------------------------------------------
# Independent reference implementations
# --------------------------------------------------------------------------

def ref_prior_unnormalised(prior):
    n = prior["n"]
    table = {}
    for z, weight in enumerate(prior["class_weights"]):
        for mask in range(2 ** n):
            mass = weight
            for i in range(n):
                p = prior["occupied"][z][i]
                mass *= p if (mask >> i) & 1 else (1.0 - p)
            for factor in prior["pair_factors"]:
                mass *= factor["values"][(mask >> factor["i"]) & 1][
                    (mask >> factor["j"]) & 1]
            table[(z, mask)] = mass
    return table


def ref_first_hit(mask, cells):
    for offset, cell in enumerate(cells):
        if (mask >> cell) & 1:
            return offset
    return len(cells)


def ref_infer(prior, rays):
    n = prior["n"]
    raw = ref_prior_unnormalised(prior)
    z_total = sum(raw.values())
    map_prior = {m: sum(raw[(z, m)] for z in range(len(prior["class_weights"])))
                 / z_total for m in range(2 ** n)}

    # Joint log mass over (z, mask) then likelihood per observed ray.
    log_terms = []
    for (z, mask), mass in raw.items():
        if mass <= 0.0:
            continue
        log_lik = 0.0
        for ray in rays:
            if ray["reading"] is None:
                continue
            h = ref_first_hit(mask, ray["cells"])
            p = ray["confusion"][h][ray["reading"]]
            if p == 0.0:
                log_lik = None
                break
            log_lik += math.log(p)
        if log_lik is not None:
            log_terms.append(math.log(mass) + log_lik)
    if not log_terms:
        return None  # impossible evidence
    log_evidence = mapmodel._logsumexp(log_terms) - math.log(z_total)

    weighted = {}
    for mask in range(2 ** n):
        acc = 0.0
        for z in range(len(prior["class_weights"])):
            mass = raw[(z, mask)]
            if mass <= 0.0:
                continue
            keep = 1.0
            for ray in rays:
                if ray["reading"] is None:
                    continue
                keep *= ray["confusion"][
                    ref_first_hit(mask, ray["cells"])][ray["reading"]]
            acc += mass * keep
        weighted[mask] = acc
    evidence_mass = sum(weighted.values())
    posterior = [weighted[m] / evidence_mass for m in range(2 ** n)]
    class_post = []
    for z in range(len(prior["class_weights"])):
        total = 0.0
        for mask in range(2 ** n):
            mass = raw[(z, mask)]
            for ray in rays:
                if ray["reading"] is None:
                    continue
                mass *= ray["confusion"][
                    ref_first_hit(mask, ray["cells"])][ray["reading"]]
            total += mass
        class_post.append(total / evidence_mass)
    return {"log_evidence": log_evidence, "posterior": posterior,
            "class": class_post, "map_prior": map_prior}


def ref_event(distribution, clear, occupied, cond_clear, cond_occ):
    n, probs = mapmodel.validate_distribution(distribution)
    if set(cond_clear) & set(cond_occ):
        return "zero_condition", None
    cond = cevent = 0.0
    for mask, p in enumerate(probs):
        if p == 0.0:
            continue
        ok_cond = all((mask >> i) & 1 for i in cond_occ) and \
            all(not ((mask >> i) & 1) for i in cond_clear)
        if not ok_cond:
            continue
        cond += p
        if all((mask >> i) & 1 for i in occupied) and \
                all(not ((mask >> i) & 1) for i in clear):
            cevent += p
    if cond == 0.0:
        return "zero_condition", None
    if set(clear) & set(occupied):
        return "ok", 0.0
    return "ok", cevent / cond


def ref_plan(graph, start, goal, threshold, distribution):
    n, probs = mapmodel.validate_distribution(distribution)
    adjacency = {}
    for edge in graph["edges"]:
        adjacency.setdefault(edge["u"], []).append(edge)
    if start == goal:
        return ("ok", 0, [], [start], 0, 1.0)
    candidates = []

    def collect(node, visited, edge_seq, node_seq, mask, cost):
        if node == goal:
            success = sum(p for m, p in enumerate(probs)
                          if p > 0 and (m & mask) == 0)
            candidates.append((cost, list(edge_seq), list(node_seq),
                               mask, success))
        for edge in adjacency.get(node, []):
            nxt = edge["v"]
            if visited & (1 << nxt):
                continue
            em = 0
            for c in edge["cells"]:
                em |= 1 << c
            collect(nxt, visited | (1 << nxt), edge_seq + [edge["id"]],
                    node_seq + [nxt], mask | em, cost + edge["cost"])

    if start == goal:
        return ("ok", [], [start], 0, 0, 1.0)
    collect(start, 1 << start, [], [start], 0, 0)
    feasible = [c for c in candidates if c[4] + 1e-12 >= threshold]
    if not feasible:
        return "infeasible"
    feasible.sort(key=lambda c: c[0])
    return ("ok",) + feasible[0]


# --------------------------------------------------------------------------
# Scenario 1: tiny independent map, normalisation / likelihood / posterior
# --------------------------------------------------------------------------

def section(name):
    print(f"\n=== {name} ===")


def test_map_inference():
    section("map inference: brute-force normalisation, likelihood, posterior")
    prior = {
        "n": 3,
        "class_weights": [0.4, 0.6],
        "occupied": [[0.7, 0.2, 0.5], [0.1, 0.8, 0.4]],
        "pair_factors": [
            {"i": 0, "j": 2, "values": [[1.0, 0.3], [0.3, 1.0]]},
            {"i": 1, "j": 2, "values": [[1.0, 0.5], [0.5, 1.0]]},
        ],
    }
    rays = [
        {"cells": [0, 1, 2],
         "confusion": [[0.6, 0.3, 0.1, 0.0],
                       [0.1, 0.7, 0.1, 0.1],
                       [0.1, 0.1, 0.7, 0.1],
                       [0.0, 0.1, 0.3, 0.6]],
         "reading": 2},
        {"cells": [2, 0],
         "confusion": [[0.5, 0.4, 0.1],
                       [0.2, 0.6, 0.2],
                       [0.1, 0.2, 0.7]],
         "reading": 0},
    ]
    result = mapmodel.infer(prior, rays)
    ref = ref_infer(prior, rays)
    check("status ok", result["status"], "ok")
    check("log_evidence", result["log_evidence"], ref["log_evidence"], 1e-9)
    check("posterior length", len(result["distribution"]["probabilities"]),
          2 ** prior["n"])
    check("posterior sums to 1", sum(result["distribution"]["probabilities"]),
          1.0)
    for m in range(2 ** prior["n"]):
        check(f"posterior mask {m}",
              result["distribution"]["probabilities"][m],
              ref["posterior"][m])
    for z in range(2):
        check(f"class posterior {z}", result["class_probabilities"][z],
              ref["class"][z])
    # marginals and pair marginals re-derived from the brute posterior
    for i in range(3):
        p_i = sum(ref["posterior"][m] for m in range(8) if (m >> i) & 1)
        check(f"cell marginal {i}", result["cell_occupied"][i], p_i)
        check(f"pair diagonal {i}{i}", result["pair_occupied"][i][i], p_i)
    for i, j in itertools.combinations(range(3), 2):
        p_ij = sum(ref["posterior"][m] for m in range(8)
                   if (m >> i) & 1 and (m >> j) & 1)
        check(f"pair marginal {i},{j}", result["pair_occupied"][i][j], p_ij)
        check(f"pair marginal {j},{i}", result["pair_occupied"][j][i], p_ij)

    # All readings missing: evidence 1, posterior equals the brute prior.
    missing = [dict(ray, reading=None) for ray in rays]
    miss = mapmodel.infer(prior, missing)
    check("missing status", miss["status"], "ok")
    check("missing log_evidence 0", miss["log_evidence"], 0.0, 1e-12)
    for m in range(8):
        check(f"missing posterior == prior mask {m}",
              miss["distribution"]["probabilities"][m], ref["map_prior"][m])

    # Impossible report: column never reachable under any first-hit value.
    bad_ray = {"cells": [0, 1],
               "confusion": [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                             [0.0, 1.0, 0.0]],
               "reading": 0}
    impossible = mapmodel.infer(
        {"n": 2, "class_weights": [1.0],
         "occupied": [[0.5, 0.5]], "pair_factors": []}, [bad_ray])
    check("impossible status", impossible["status"], "impossible_evidence")
    check("impossible log_evidence", impossible["log_evidence"], float("-inf"))
    check("impossible posterior None", impossible["distribution"], None)


def test_event():
    section("event: joint masses from an external non-model distribution")
    dist = {"n": 3, "probabilities": [0.10, 0.05, 0.20, 0.00,
                                      0.15, 0.10, 0.25, 0.15]}
    cases = [
        ([0], [], [], []),
        ([0, 1], [], [], []),
        ([2], [0], [], []),
        ([1], [], [0], [2]),
        ([], [0, 2], [1], []),
        ([0], [0], [], []),              # contradictory event
        ([0], [], [1], [1]),             # contradictory condition
        ([0], [], [2], []),              # zero-mass condition (mask lacks bit2)
    ]
    for args in cases:
        got = mapmodel.event(dist, *args)
        status, prob = ref_event(dist, *args)
        check(f"event {args} status", got["status"], status)
        if prob is None:
            check(f"event {args} probability None", got["probability"], None)
        else:
            check(f"event {args} probability", got["probability"], prob)


def test_routing():
    section("routing: exhaustive simple paths, joint risk and global cost")
    graph = {"vertices": 5, "edges": [
        {"id": "a", "u": 0, "v": 1, "cost": 2, "cells": [0]},
        {"id": "b", "u": 1, "v": 2, "cost": 1, "cells": [0, 2]},
        {"id": "c", "u": 2, "v": 4, "cost": 2, "cells": [1]},
        {"id": "d", "u": 0, "v": 3, "cost": 4, "cells": [3]},
        {"id": "e", "u": 3, "v": 4, "cost": 1, "cells": [1, 3]},
        {"id": "p1", "u": 0, "v": 4, "cost": 6, "cells": [2]},
        {"id": "p2", "u": 0, "v": 4, "cost": 6, "cells": [3]},
        {"id": "loop", "u": 1, "v": 1, "cost": 1, "cells": []},
        {"id": "back", "u": 2, "v": 1, "cost": 1, "cells": [2]},
    ]}
    dist = {"n": 4, "probabilities": [
        0.08, 0.06, 0.05, 0.04, 0.07, 0.03, 0.06, 0.02,
        0.09, 0.05, 0.08, 0.04, 0.10, 0.06, 0.09, 0.08]}
    check("distribution mass", sum(dist["probabilities"]), 1.0)

    # Replay every simple path independently and confirm the planner's
    # success probability for the chosen route matches the brute sum.
    adjacency = {}
    for edge in graph["edges"]:
        adjacency.setdefault(edge["u"], []).append(edge)
    enumerated = []

    def collect(node, visited, edge_seq, node_seq, mask, cost):
        if node == 4:
            success = sum(p for m, p in enumerate(dist["probabilities"])
                          if p > 0 and (m & mask) == 0)
            enumerated.append((cost, tuple(edge_seq), mask, success))
        for edge in adjacency.get(node, []):
            nxt = edge["v"]
            if visited & (1 << nxt):
                continue
            em = 0
            for c in edge["cells"]:
                em |= 1 << c
            collect(nxt, visited | (1 << nxt), edge_seq + [edge["id"]],
                    node_seq + [nxt], mask | em, cost + edge["cost"])

    collect(0, 1 << 0, [], [0], 0, 0)
    print(f"  (enumerated {len(enumerated)} simple 0->4 paths; "
          f"self-loop/back edges never extend a simple path)")

    for threshold in (0.0, 0.35, 0.55, 0.8, 0.95):
        plan = routing.plan(graph, 0, 4, threshold, dist)
        ref_result = ref_plan(graph, 0, 4, threshold, dist)
        if ref_result == "infeasible":
            check(f"threshold {threshold}: infeasible", plan["status"],
                  "infeasible")
            check(f"threshold {threshold}: null cost", plan["cost"], None)
            continue
        _, ref_cost, ref_edges, ref_nodes, ref_mask, ref_success = ref_result
        check(f"threshold {threshold}: status", plan["status"], "ok")
        check(f"threshold {threshold}: global min cost", plan["cost"],
              min(c for c, _, _, s in enumerated if s + 1e-12 >= threshold))
        check(f"threshold {threshold}: reference cost", plan["cost"], ref_cost)
        check(f"threshold {threshold}: success joint",
              plan["success_probability"], ref_success)
        chosen_mask = 0
        for c in plan["cells"]:
            chosen_mask |= 1 << c
        check(f"threshold {threshold}: cell union", chosen_mask,
              # recompute union from the chosen edges
              _union_from_edges(graph, plan["edges"]))
        check(f"threshold {threshold}: edge sequence feasible",
              tuple(plan["edges"]) in [e for _, e, _, _ in enumerated], True)

    # Zero-edge route.
    zero = routing.plan(graph, 2, 2, 0.7, dist)
    check("zero-edge status", zero["status"], "ok")
    check("zero-edge cost", zero["cost"], 0)
    check("zero-edge success", zero["success_probability"], 1.0)
    check("zero-edge cells", zero["cells"], [])

    # Genuinely no feasible route: every edge crosses a certainly occupied cell.
    hard_graph = {"vertices": 2, "edges": [
        {"id": "x", "u": 0, "v": 1, "cost": 1, "cells": [0]},
        {"id": "y", "u": 0, "v": 1, "cost": 2, "cells": [0, 1]}]}
    hard_dist = {"n": 2, "probabilities": [0.0, 0.0, 0.4, 0.6]}
    no_route = routing.plan(hard_graph, 0, 1, 0.5, hard_dist)
    check("no feasible route status", no_route["status"], "infeasible")
    check("no feasible route cost", no_route["cost"], None)
    check("no feasible route edges", no_route["edges"], [])

    # Repeated traversal of one cell counts once: two edges both on cell 0.
    repeat_graph = {"vertices": 3, "edges": [
        {"id": "r1", "u": 0, "v": 1, "cost": 1, "cells": [0]},
        {"id": "r2", "u": 1, "v": 2, "cost": 1, "cells": [0]}]}
    repeat_dist = {"n": 1, "probabilities": [0.7, 0.3]}
    rep = routing.plan(repeat_graph, 0, 2, 0.0, repeat_dist)
    check("repeated cell union", rep["cells"], [0])
    check("repeated cell success", rep["success_probability"], 0.7)
    check("repeated cell cost", rep["cost"], 2)


def _union_from_edges(graph, edge_ids):
    mask = 0
    by_id = {e["id"]: e for e in graph["edges"]}
    for edge_id in edge_ids:
        for c in by_id[edge_id]["cells"]:
            mask |= 1 << c
    return mask


def test_boundaries_and_validation():
    section("boundaries: n=1, deterministic cells, missing == prior, errors")
    # n=1 posterior against a one-cell Bayes computation.
    prior1 = {"n": 1, "class_weights": [1.0], "occupied": [[0.3]],
              "pair_factors": []}
    # confusion[h][y]: h=0 means the cell is occupied, h=1 means it is free.
    ray1 = {"cells": [0], "confusion": [[0.2, 0.8], [0.6, 0.4]],
            "reading": 1}
    out = mapmodel.infer(prior1, [ray1])
    # P(y=1|free)=0.4, P(y=1|occupied)=0.8 under confusion rows [h][y].
    p_y_free, p_y_occ = 0.4, 0.8
    evidence1 = 0.7 * p_y_free + 0.3 * p_y_occ
    p_occ = 0.3 * p_y_occ / evidence1
    check("n=1 free mass", out["distribution"]["probabilities"][0],
          1.0 - p_occ)
    check("n=1 occupied mass", out["distribution"]["probabilities"][1], p_occ)
    check("n=1 pair diagonal equals marginal",
          out["pair_occupied"][0][0], out["cell_occupied"][0])

    # Deterministic cells pin the support to a single mask; zero entries are
    # genuine zero probabilities, not underflow.
    forced = {"n": 2, "class_weights": [1.0], "occupied": [[0.0, 1.0]],
              "pair_factors": []}
    fp = mapmodel.infer(forced, [])
    check("deterministic support",
          fp["distribution"]["probabilities"], [0.0, 0.0, 1.0, 0.0])
    check("deterministic log evidence", fp["log_evidence"], 0.0)

    # Missing reading contributes likelihood 1 and reproduces the prior.
    miss_ray = dict(ray1, reading=None)
    mm = mapmodel.infer(prior1, [miss_ray])
    check("missing reproduces prior free",
          mm["distribution"]["probabilities"][0], 0.7)
    check("missing evidence is 1", mm["log_evidence"], 0.0, 1e-12)

    # Illegal inputs raise ValueError rather than looking like zero evidence.
    def raises(fn):
        try:
            fn()
            return False
        except ValueError:
            return True

    bad_priors = [
        {"n": 0, "class_weights": [1.0], "occupied": [[]],
         "pair_factors": []},
        {"n": 2, "class_weights": [0.5, 0.5],
         "occupied": [[0.5, 0.5], [0.5, 0.5]],
         "pair_factors": [{"i": 0, "j": 1, "values": [[1, 1], [1, 1]]},
                           {"i": 1, "j": 0, "values": [[1, 1], [1, 1]]}]},
        {"n": 2, "class_weights": [0.5, 0.4],
         "occupied": [[0.5, 0.5], [0.5, 0.5]], "pair_factors": []},
        {"n": 2, "class_weights": [1.0], "occupied": [[0.5, 1.2]],
         "pair_factors": []},
    ]
    for k, bad in enumerate(bad_priors):
        check(f"bad prior {k} raises", raises(lambda b=bad:
                                              mapmodel.infer(b, [])), True)
    bad_rays = [
        [{"cells": [0, 0], "confusion": [[1], [1]], "reading": 0}],
        [{"cells": [0], "confusion": [[0.5, 0.5], [0.5, 0.5]],
          "reading": 7}],
        [{"cells": [0], "confusion": [[0.4, 0.5], [0.5, 0.5]],
          "reading": 0}],  # row does not sum to 1
    ]
    for k, bad in enumerate(bad_rays):
        check(f"bad ray {k} raises",
              raises(lambda b=bad: mapmodel.infer(prior1, b)), True)
    bad_graphs = [
        lambda: routing.plan(
            {"vertices": 1, "edges": [{"id": "x", "u": 0, "v": 0,
                                       "cost": 1, "cells": [5]}]},
            0, 0, 0.5, {"n": 1, "probabilities": [1.0]}),
        lambda: routing.plan(
            {"vertices": 2, "edges": [{"id": "x", "u": 0, "v": 1,
                                       "cost": 9, "cells": []}]},
            0, 5, 0.5, {"n": 1, "probabilities": [1.0]}),
    ]
    for k, bad in enumerate(bad_graphs):
        check(f"bad graph {k} raises", raises(bad), True)


if __name__ == "__main__":
    test_map_inference()
    test_event()
    test_routing()
    test_boundaries_and_validation()
    print(f"\n{FAIL} failures, {PASS} checks passed")
    raise SystemExit(1 if FAIL else 0)
