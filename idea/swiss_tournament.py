from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple


def elo_update(
    elo_a: float,
    elo_b: float,
    winner: Optional[str],
    k_factor: int = 32,
) -> Tuple[float, float]:
    """Apply a standard Elo update for a head-to-head result."""
    expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    expected_b = 1 / (1 + 10 ** ((elo_a - elo_b) / 400))

    if winner == "A":
        elo_a = elo_a + k_factor * (1 - expected_a)
        elo_b = elo_b + k_factor * (0 - expected_b)
    elif winner == "B":
        elo_a = elo_a + k_factor * (0 - expected_a)
        elo_b = elo_b + k_factor * (1 - expected_b)
    else:
        elo_a = elo_a + k_factor * (0.5 - expected_a)
        elo_b = elo_b + k_factor * (0.5 - expected_b)

    return elo_a, elo_b


def select_swiss_bye(ordered_indices: List[int], bye_counts: Dict[int, int]) -> Optional[int]:
    """
    Select a bye candidate for an odd-sized Swiss round.

    Preference order:
    1) Fewest byes so far
    2) Lowest-ranked (last in ordered list)
    """
    if not ordered_indices:
        return None

    min_byes = min(bye_counts.get(idx, 0) for idx in ordered_indices)
    for idx in reversed(ordered_indices):
        if bye_counts.get(idx, 0) == min_byes:
            return idx
    return ordered_indices[-1]


def pair_players_swiss(
    ordered_indices: List[int],
    match_history: Set[Tuple[int, int]],
    backtrack_limit: int = 20000,
) -> Optional[List[Tuple[int, int]]]:
    """
    Pair players in Swiss style while minimizing repeat matchups.

    Returns a list of pairs (idx_a, idx_b), or None if no pairing is found
    within the backtracking limit.
    """
    steps = 0

    def backtrack(remaining: List[int]) -> Optional[Tuple[List[Tuple[int, int]], int]]:
        nonlocal steps
        steps += 1
        if steps > backtrack_limit:
            return None
        if not remaining:
            return [], 0

        first = remaining[0]
        candidates = []
        for i in range(1, len(remaining)):
            second = remaining[i]
            pair_key = (min(first, second), max(first, second))
            repeat = pair_key in match_history
            candidates.append((repeat, second, i))

        candidates.sort(key=lambda x: (x[0], x[1]))

        best_pairs = None
        best_repeats = None

        for repeat, second, idx in candidates:
            next_remaining = remaining[1:idx] + remaining[idx + 1 :]
            result = backtrack(next_remaining)
            if result is None:
                continue
            sub_pairs, sub_repeats = result
            total_repeats = (1 if repeat else 0) + sub_repeats
            if best_repeats is None or total_repeats < best_repeats:
                best_pairs = [(first, second)] + sub_pairs
                best_repeats = total_repeats
                if best_repeats == 0:
                    break

        if best_pairs is None:
            return None
        return best_pairs, best_repeats

    result = backtrack(ordered_indices)
    if result is None:
        return None
    return result[0]


def generate_swiss_round_pairs(
    ranks: Dict[int, float],
    match_history: Set[Tuple[int, int]],
    bye_counts: Dict[int, int],
) -> Tuple[List[Tuple[int, int]], Optional[int]]:
    """
    Generate Swiss pairings for a single round with minimal repeat matchups.

    Notes:
    - `match_history` is mutated in place with generated pairs.
    - `bye_counts` is mutated in place when a bye is assigned.
    """
    ordered_indices = sorted(ranks.keys(), key=lambda i: (-ranks[i], i))

    bye_idx = None
    if len(ordered_indices) % 2 == 1:
        bye_idx = select_swiss_bye(ordered_indices, bye_counts)
        if bye_idx is not None:
            ordered_indices.remove(bye_idx)

    pairs = pair_players_swiss(ordered_indices, match_history)
    if pairs is None:
        # Fallback greedy pairing if backtracking exceeded limit
        pairs = []
        remaining = ordered_indices[:]
        while len(remaining) >= 2:
            a = remaining.pop(0)
            partner_idx = None
            for i, b in enumerate(remaining):
                pair_key = (min(a, b), max(a, b))
                if pair_key not in match_history:
                    partner_idx = i
                    break
            if partner_idx is None:
                partner_idx = 0
            b = remaining.pop(partner_idx)
            pairs.append((a, b))

    for a, b in pairs:
        match_history.add((min(a, b), max(a, b)))

    if bye_idx is not None:
        bye_counts[bye_idx] = bye_counts.get(bye_idx, 0) + 1

    return pairs, bye_idx
