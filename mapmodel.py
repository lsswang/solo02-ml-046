"""Ordinary independent-cell inverse sensor update; no correlated inference."""
import math


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
