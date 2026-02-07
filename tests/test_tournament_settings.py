import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.viewer import _resolve_tournament_settings


def test_tournament_count_one_full_tournament():
    count, full_rounds, target_rounds = _resolve_tournament_settings(
        pop_size=5,
        tournament_count_input=1.0,
        legacy_rounds_input=None,
    )
    assert full_rounds == 4
    assert count == 1.0
    assert target_rounds == 4


def test_tournament_count_fractional_rounding():
    count, full_rounds, target_rounds = _resolve_tournament_settings(
        pop_size=5,
        tournament_count_input=0.75,
        legacy_rounds_input=None,
    )
    assert full_rounds == 4
    assert count == 0.75
    assert target_rounds == 3


def test_tournament_count_multiple_tournaments():
    count, full_rounds, target_rounds = _resolve_tournament_settings(
        pop_size=5,
        tournament_count_input=1.5,
        legacy_rounds_input=None,
    )
    assert full_rounds == 4
    assert count == 1.5
    assert target_rounds == 6


def test_legacy_tournament_rounds_compatibility():
    count, full_rounds, target_rounds = _resolve_tournament_settings(
        pop_size=5,
        tournament_count_input=None,
        legacy_rounds_input=2,
    )
    assert full_rounds == 4
    assert target_rounds == 2
    assert count == 0.5
