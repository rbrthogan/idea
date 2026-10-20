"""Whole-generation birth allocation, independent of island rounding."""

import math

BIRTH_TYPES = ("independent", "single_parent", "crossover")


def allocate_birth_counts(total, independent_rate, single_parent_rate, generation=0):
    rates = (independent_rate, single_parent_rate,
             max(0.0, 1.0-independent_rate-single_parent_rate))
    exact = [total*rate for rate in rates]
    counts = [math.floor(value) for value in exact]
    # Rotate equal remainder priorities so a tiny run does not permanently
    # exclude one operator (e.g. single-parent vs crossover at 25% each).
    order = sorted(range(3), key=lambda idx: (-(exact[idx]-counts[idx]),
                                             (idx-generation) % 3))
    for idx in order[:total-sum(counts)]:
        counts[idx] += 1
    return dict(zip(BIRTH_TYPES, counts))


def distribute_births(capacities, independent_rate, single_parent_rate, generation=0):
    counts = allocate_birth_counts(sum(capacities), independent_rate,
                                   single_parent_rate, generation)
    plans = [[] for _ in capacities]
    if not capacities or not sum(capacities):
        return plans
    cursor = generation % len(capacities)
    for kind in BIRTH_TYPES:
        for _ in range(counts[kind]):
            while len(plans[cursor]) >= capacities[cursor]:
                cursor = (cursor+1) % len(capacities)
            plans[cursor].append(kind)
            cursor = (cursor+1) % len(capacities)
    return plans
