"""Exact probabilistic occupancy inference on a finite static grid.

The prior over a binary map of ``n`` cells is a finite mixture of scene
classes ``z`` with per-class independent Bernoulli cell masses multiplied by
arbitrary unnormalised pairwise table factors.  Sensor reports are noisy
observations of the *first occupied cell* (hit position) of a ray, so a
report is one joint observation of the whole ray rather than independent
per-cell evidence.

Every inference routine enumerates the full ``2**n`` map masks (n <= 12) and
works in the log domain so that genuine zero probability is distinguishable
from floating point underflow.
"""

import math

# Inputs are declared accurate at 1e-12; numeric outputs are checked at 1e-10.
_TOL = 1e-12
_NEG_INF = float("-inf")


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------

def _is_int(value):
    """Strict integer test: booleans are not accepted as structural ints."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value)


def _check_index_list(values, n, name, allow_empty=True):
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list of cell indices")
    if not allow_empty and len(values) == 0:
        raise ValueError(f"{name} must not be empty")
    seen = set()
    for value in values:
        if not _is_int(value) or value < 0 or value >= n:
            raise ValueError(f"{name} contains out-of-range cell index {value!r}")
        if value in seen:
            raise ValueError(f"{name} contains duplicated cell index {value}")
        seen.add(value)
    return list(values)


def _validate_prior(prior):
    if not isinstance(prior, dict):
        raise ValueError("prior must be an object")
    n = prior.get("n")
    if not _is_int(n) or not (1 <= n <= 12):
        raise ValueError("prior.n must be an integer in 1..12")

    weights = prior.get("class_weights")
    if not isinstance(weights, (list, tuple)) or not (1 <= len(weights) <= 4):
        raise ValueError("prior.class_weights must be 1..4 class probabilities")
    for w in weights:
        if not _is_finite_number(w) or w < 0.0:
            raise ValueError("class weights must be finite non-negative numbers")
    if abs(sum(weights) - 1.0) > _TOL:
        raise ValueError("class_weights must sum to 1 (tolerance 1e-12)")

    occupied = prior.get("occupied")
    k = len(weights)
    if not isinstance(occupied, (list, tuple)) or len(occupied) != k:
        raise ValueError("prior.occupied must give one row per class")
    for row in occupied:
        if not isinstance(row, (list, tuple)) or len(row) != n:
            raise ValueError("each occupied row must have n entries")
        for p in row:
            if not _is_finite_number(p) or not (0.0 <= p <= 1.0):
                raise ValueError("occupied probabilities must lie in [0, 1]")

    factors = prior.get("pair_factors", [])
    if not isinstance(factors, (list, tuple)) or len(factors) > 12:
        raise ValueError("pair_factors must be a list of at most 12 entries")
    seen_pairs = set()
    for factor in factors:
        if not isinstance(factor, dict):
            raise ValueError("each pair factor must be an object")
        i, j = factor.get("i"), factor.get("j")
        if not _is_int(i) or not _is_int(j) or not (0 <= i < n) \
                or not (0 <= j < n) or i == j:
            raise ValueError("pair factor cells must be distinct valid indices")
        key = (i, j) if i < j else (j, i)
        if key in seen_pairs:
            raise ValueError(f"duplicate unordered pair factor {key}")
        seen_pairs.add(key)
        table = factor.get("values")
        if not isinstance(table, (list, tuple)) or len(table) != 2:
            raise ValueError("pair factor values must be a 2x2 table")
        for row in table:
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise ValueError("pair factor values must be a 2x2 table")
            for value in row:
                if not _is_finite_number(value) or value < 0.0 or value > 10.0:
                    raise ValueError(
                        "pair factor entries must be finite, in [0, 10]")
    return n, list(weights), [list(row) for row in occupied], list(factors)


def _validate_rays(rays, n):
    if not isinstance(rays, (list, tuple)) or len(rays) > 12:
        raise ValueError("rays must be a list of at most 12 rays")
    normalised = []
    for ray in rays:
        if not isinstance(ray, dict):
            raise ValueError("each ray must be an object")
        cells = _check_index_list(ray.get("cells"), n, "ray.cells",
                                  allow_empty=False)
        length = len(cells)
        confusion = ray.get("confusion")
        if not isinstance(confusion, (list, tuple)) or len(confusion) != length + 1:
            raise ValueError("confusion must be (L+1)x(L+1)")
        rows = []
        for row in confusion:
            if not isinstance(row, (list, tuple)) or len(row) != length + 1:
                raise ValueError("confusion must be (L+1)x(L+1)")
            for value in row:
                if not _is_finite_number(value) or value < 0.0:
                    raise ValueError("confusion entries must be finite and >= 0")
            row_sum = sum(row)
            if abs(row_sum - 1.0) > _TOL:
                raise ValueError("each confusion row must sum to 1 (1e-12)")
            # Re-normalise with the actual row sum as specified.
            rows.append([value / row_sum for value in row])
        reading = ray.get("reading")
        if reading is not None and (not _is_int(reading)
                                    or not (0 <= reading <= length)):
            raise ValueError("reading must be None or an integer in 0..L")
        normalised.append({"cells": cells, "confusion": rows,
                           "reading": reading})
    return normalised


def validate_distribution(distribution):
    """Validate an external joint distribution and return (n, renormalised p)."""
    if not isinstance(distribution, dict):
        raise ValueError("distribution must be an object")
    n = distribution.get("n")
    if not _is_int(n) or not (1 <= n <= 12):
        raise ValueError("distribution.n must be an integer in 1..12")
    probs = distribution.get("probabilities")
    if not isinstance(probs, (list, tuple)) or len(probs) != 2 ** n:
        raise ValueError("distribution.probabilities must have length 2**n")
    total = 0.0
    for p in probs:
        if not _is_finite_number(p) or p < 0.0:
            raise ValueError("distribution probabilities must be finite and >= 0")
        total += p
    if total <= 0.0:
        raise ValueError("distribution total mass must be positive")
    return n, [p / total for p in probs]


# --------------------------------------------------------------------------
# Numeric helpers
# --------------------------------------------------------------------------

def _safe_log(x):
    return math.log(x) if x > 0.0 else _NEG_INF


def _logsumexp(log_values):
    finite = [v for v in log_values if v != _NEG_INF]
    if not finite:
        return _NEG_INF
    peak = max(finite)
    return peak + math.log(sum(math.exp(v - peak) for v in finite))


# --------------------------------------------------------------------------
# Prior joint mass (enumerated over all classes and 2**n masks)
# --------------------------------------------------------------------------

def prior_log_weights(prior):
    """Return (n, log_mass[k][mask]) for every class and map mask.

    The returned masses are unnormalised; callers normalise with
    ``_logsumexp`` over the whole table.
    """
    n, weights, occupied, factors = _validate_prior(prior)
    masks = 1 << n
    log_p = [[_safe_log(p) for p in row] for row in occupied]
    log_q = [[_safe_log(1.0 - p) for p in row] for row in occupied]
    log_w = [_safe_log(w) for w in weights]
    log_factor = []
    for factor in factors:
        i, j, table = factor["i"], factor["j"], factor["values"]
        log_factor.append((i, j,
                           [[_safe_log(table[a][b]) for b in (0, 1)]
                            for a in (0, 1)]))

    log_mass = [[_NEG_INF] * masks for _ in weights]
    for k in range(len(weights)):
        for mask in range(masks):
            value = log_w[k]
            if value == _NEG_INF:
                continue
            for i in range(n):
                value += log_p[k][i] if (mask >> i) & 1 else log_q[k][i]
                if value == _NEG_INF:
                    break
            if value == _NEG_INF:
                continue
            for i, j, table in log_factor:
                value += table[(mask >> i) & 1][(mask >> j) & 1]
                if value == _NEG_INF:
                    break
            log_mass[k][mask] = value
    return n, log_mass


def _ray_hit(mask, cells):
    """Position of the first occupied cell, or L when every cell is free."""
    for offset, cell in enumerate(cells):
        if (mask >> cell) & 1:
            return offset
    return len(cells)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def infer(prior, rays):
    """Exact Bayesian update over all map masks and scene classes.

    Returns status, log_evidence, class_probabilities, cell_occupied,
    pair_occupied and the full 2**n posterior distribution (summed over z).
    """
    n, log_mass = prior_log_weights(prior)
    rays = _validate_rays(rays, n)
    masks = 1 << n
    k_classes = len(log_mass)

    log_norm = _logsumexp(w for row in log_mass for w in row)
    if log_norm == _NEG_INF:
        raise ValueError("prior assigns zero mass to every (z, map) pair")

    # Per-ray likelihood table indexed by mask, skipping missing readings.
    observed = []
    for ray in rays:
        if ray["reading"] is None:
            continue
        cells, confusion, reading = (ray["cells"], ray["confusion"],
                                     ray["reading"])
        likelihood = [0.0] * masks
        for mask in range(masks):
            likelihood[mask] = confusion[_ray_hit(mask, cells)][reading]
        observed.append(likelihood)

    # Unnormalised joint log posterior log P(r|m) P(m,z) over (z, mask).
    log_joint = [[_NEG_INF] * masks for _ in range(k_classes)]
    log_evidence_terms = []
    for k in range(k_classes):
        for mask in range(masks):
            log_lik = 0.0
            for likelihood in observed:
                p = likelihood[mask]
                if p == 0.0:
                    log_lik = _NEG_INF
                    break
                log_lik += math.log(p)
            if log_lik != _NEG_INF and log_mass[k][mask] != _NEG_INF:
                log_joint[k][mask] = log_mass[k][mask] + log_lik
                log_evidence_terms.append(log_joint[k][mask])

    # log P(r) under the normalised prior; absent observations give P=1.
    log_evidence = (_logsumexp(log_evidence_terms) - log_norm
                    if log_evidence_terms else 0.0)
    if not log_evidence_terms or log_evidence == _NEG_INF:
        return {
            "status": "impossible_evidence",
            "log_evidence": _NEG_INF,
            "class_probabilities": None,
            "cell_occupied": None,
            "pair_occupied": None,
            "distribution": None,
        }

    # Posterior weights exp(log_joint - log_norm - log_evidence).  Computing
    # the single subtraction inside exp keeps the work in the log domain even
    # when the evidence itself is extremely small.
    log_scale = log_norm + log_evidence
    class_mass = [0.0] * k_classes
    pair_mass = [[0.0] * n for _ in range(n)]
    posterior = [0.0] * masks

    for k in range(k_classes):
        for mask in range(masks):
            if log_joint[k][mask] == _NEG_INF:
                continue
            p = math.exp(log_joint[k][mask] - log_scale)
            class_mass[k] += p
            posterior[mask] += p
            bits = mask
            while bits:
                bit = bits & -bits
                i = bit.bit_length() - 1
                pair_mass[i][i] += p
                other = bits ^ bit
                while other:
                    obit = other & -other
                    j = obit.bit_length() - 1
                    pair_mass[i][j] += p
                    pair_mass[j][i] += p
                    other ^= obit
                bits ^= bit
    cell_mass = [pair_mass[i][i] for i in range(n)]

    # Remove cumulative rounding drift while every mass stays joint-derived.
    total = sum(posterior)
    posterior = [p / total for p in posterior]
    cell_mass = [p / total for p in cell_mass]
    pair_mass = [[p / total for p in row] for row in pair_mass]
    class_total = sum(class_mass)
    class_mass = [p / class_total for p in class_mass]

    return {
        "status": "ok",
        "log_evidence": log_evidence,
        "class_probabilities": class_mass,
        "cell_occupied": cell_mass,
        "pair_occupied": pair_mass,
        "distribution": {"n": n, "probabilities": posterior},
    }


def event(distribution, clear, occupied=(), condition_clear=(),
          condition_occupied=()):
    """Joint event probability under an externally supplied map distribution.

    All queried cells must be simultaneously free/occupied; conditions are
    handled on the same joint distribution rather than via marginal products.
    """
    n, probs = validate_distribution(distribution)
    clear = _check_index_list(clear, n, "clear")
    occupied = _check_index_list(occupied, n, "occupied")
    condition_clear = _check_index_list(condition_clear, n, "condition_clear")
    condition_occupied = _check_index_list(condition_occupied, n,
                                           "condition_occupied")

    cond_free = _bitmask(condition_clear, n)
    cond_occ = _bitmask(condition_occupied, n)
    # A condition demanding the same cell both ways is the empty condition.
    if cond_free & cond_occ:
        return {"status": "zero_condition", "probability": None}
    cond_mass = _joint_mass(probs, cond_free, cond_occ)
    if cond_mass == 0.0:
        return {"status": "zero_condition", "probability": None}

    # An event demanding the same cell both ways is the empty event.
    if set(clear).intersection(occupied):
        return {"status": "ok", "probability": 0.0}

    # Cross event/condition contradictions simply yield zero joint mass here,
    # as do configurations absent from the supplied distribution.
    event_mass = _joint_mass(
        probs,
        _bitmask(clear + condition_clear, n),
        _bitmask(occupied + condition_occupied, n))
    return {"status": "ok", "probability": event_mass / cond_mass}


def _bitmask(indices, n):
    mask = 0
    for i in indices:
        mask |= 1 << i
    return mask


def _joint_mass(probs, free_mask, occupied_mask):
    total = 0.0
    for mask, p in enumerate(probs):
        if p > 0.0 and (mask & occupied_mask) == occupied_mask \
                and (mask & free_mask) == 0:
            total += p
    return total


# --------------------------------------------------------------------------
# Legacy independent-cell inverse sensor (kept for the ordinary demo)
# --------------------------------------------------------------------------

def update_independent(prior_occupied, cells, reading):
    result = list(prior_occupied)
    if reading is None:
        return result
    # Standard fixed inverse sensor probabilities: free before the report,
    # occupied at the report, and unseen cells unchanged.
    for offset, cell in enumerate(cells):
        if offset > reading:
            break
        sensor = .8 if offset == reading else .2
        p = result[cell]
        if p == 0 or p == 1:
            continue
        odds = p / (1-p) * sensor / (1-sensor)
        result[cell] = odds / (1+odds)
    return result


def independent_clear_probability(occupied, cells):
    return math.prod(1-occupied[c] for c in set(cells))
