# Occupancy evidence and risk-bounded routing

Python 3.11, standard library only, CPU-only. Synthetic small maps
(`n <= 12`, at most 12 rays, at most 10 road vertices / 24 edges). All
guarantees are statements about the supplied probability model, not claims
about collision avoidance in a real environment; no model is downloaded and
no observation or obstacle changes while travelling.

## Reproduce

```bash
python demo.py      # structured end-to-end results, including the legacy demo
python verify.py    # 100 independent brute-force cross-checks, exit 0 on pass
```

## Modules

* `mapmodel.py`
  * `infer(prior, rays)` — exact posterior over all `2**n` map masks and the
    scene-class mixture, in the log domain.
  * `event(distribution, clear, occupied, condition_clear,
    condition_occupied)` — joint event/conditional probabilities on any
    externally supplied distribution (it need not be one this model can
    produce); the distribution is renormalised by its actual total mass.
  * `update_independent` / `independent_clear_probability` — the original
    independent-cell inverse sensor helpers, kept for the ordinary demo.
* `routing.py`
  * `plan(graph, start, goal, threshold, distribution)` — globally
    cost-optimal route whose **joint** success event (every traversed cell
    simultaneously free) meets the threshold. It consumes the distribution
    object directly and never rebuilds or re-estimates a map.
  * `shortest(graph, start, goal, blocked_cells)` — the original Dijkstra
    entry with known blocked cells.
* `demo.py` — runnable walk-through; `verify.py` — independent verifier;
  `fixtures/cases.json` — the scenarios printed by the demo.

## Probabilistic model and inference method

The prior is a finite mixture of `K <= 4` scene classes. For class `z` and
map mask `m` the unnormalised mass is

```
w(z) * prod_i Bernoulli_p(z,i,m_i) * prod_(i,j in pair_factors) f_ij(m_i, m_j)
```

with pair tables allowed to carry zeros (cells therefore need not be
independent). The whole table over `(z, m)` is normalised exactly, so an
all-zero prior is rejected with `ValueError` rather than treated as
probability.

A ray observes the **first occupied cell** of its ordered cell list; cells
behind the hit are occluded and do not change it. The noisy report follows
the confusion row of that hit, rows are renormalised by their actual sums,
and `reading=None` contributes likelihood 1. Given a full map the non-missing
reports are independent, but a single report is one joint observation of the
whole ray — it is never split into per-cell evidence and the report position
is never treated as a noiseless truth.

Because `n <= 12`, inference enumerates every `(z, mask)` pair:

1. build log prior masses (zero factor/Bernoulli entries become `-inf`, so
   true zero mass is distinct from floating-point underflow);
2. for each observed ray precompute, per mask, the confusion probability of
   its first-hit position;
3. accumulate unnormalised log posterior masses, obtain
   `log_evidence = logsumexp(terms) - log Z`, and normalise;
4. sum the joint posterior into class, cell and pair marginals
   (`pair_occupied[i, i] = P(cell i occupied)`).

If no `(z, mask)` retains positive likelihood the result is
`impossible_evidence` with `log_evidence = -inf` and all posterior fields
`None`; there is no fallback to the prior. With every reading missing the
evidence is 1 and the posterior is exactly the normalised prior. The full
`2**n` posterior is always returned.

## Routing under a correlated distribution

A route succeeds on one joint event: all cells in the **union** of its edge
cell lists are free at once. Repeating a cell across edges counts once. The
planner never multiplies per-edge success rates, accumulates per-cell risk,
or substitutes an expected obstacle count.

`F(t) = sum_{s subset t} P(mask = s)` is built once with a subset zeta
transform in `O(n 2^n)`; then `P(all cells in c free) = F(full XOR c)` is an
`O(1)` lookup for any cell set `c`. Since edge costs are positive and
traversed cells only accumulate, an optimal route is simple (cycles can be
removed), so all simple paths are enumerated with parallel edges kept as
distinct choices. Branch-and-bound prunes any partial route whose cost has
already reached the incumbent. For every threshold this is exhaustive over
the given bounds and therefore globally cost-optimal; `start == goal`
returns the zero-edge route (cost 0, success 1), and when no simple path
meets the threshold the result is `infeasible` with `cost = null`.

## Coupling demonstration

`demo.py` keeps the graph, endpoints and threshold fixed and changes one
legal report. With an all-clear report the cheap two-edge route is selected
on the returned joint posterior; after a report indicating an early hit the
same route's recomputed joint success falls below the threshold, so the
planner switches to the safer edge. The previously returned distribution is
frozen and re-evaluated under the **current** posterior to show it is no
longer compliant — printing new marginals while still routing on the old
distribution is not accepted, and the external-distribution path never
constructs a second map. `impossible_evidence` produces no distribution, so
no route is fabricated from it.

## Organisation trade-offs

* **Full map enumeration vs factor elimination.** Enumerating `K * 2**n`
  masks is simple, exact, keeps zero masses explicit in the log domain, and
  yields the required full posterior and arbitrary marginals directly. Its
  cost is `O(K n 2^n)` time and `O(K 2^n)` intermediate storage. Variable
  elimination on the pair factors would exploit sparse factor structure and
  avoid materialising every mask for *evidence* computation, but at
  `n <= 12` the tables are tiny (at most 4096 masks), and the contract
  requires the complete `2**n` posterior regardless — elimination would
  still need that output storage and a second mechanism for pair marginals,
  so only the enumeration path is implemented.
* **Simple paths vs state-augmented search.** Enumerating simple paths is
  exhaustive and easy to audit against an independent brute force; its worst
  case is exponential in the number of vertices. A search over augmented
  states `(vertex, traversed-cell mask)` (with dominance on cost and the
  monotone cell-union structure) bounds intermediate storage by
  `O(V 2^n)` and never revisits a cell set, which is the better shape for
  larger graphs or denset cell coverage, but bookkeeping for parallel edges
  and threshold feasibility is more involved. At `V <= 10, E <= 24` plain
  simple-path enumeration with cost pruning is comfortably exact and is the
  single implementation used; the zeta transform already removes the
  per-path cost of evaluating the joint event.

## Numerical and validation contract

Probabilities and marginals are held to 1e-10 absolute error, finite
`log_evidence` to 1e-9, and route cost / edge and cell identities are exact;
`success_probability` is held to 1e-10. Illegal dimensions, indices,
duplicate pairs or cells, probabilities, readings, endpoints, costs and
non-finite inputs raise `ValueError`, kept distinct from legal zero
evidence, zero-probability conditions and the (legal) absence of any
feasible route. Inputs are never mutated.
