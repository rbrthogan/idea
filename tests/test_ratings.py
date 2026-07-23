import time
from unittest.mock import patch

from idea.ratings import parallel_evaluate_pairs


def _slow_compare(_a, _b, _idea_type):
    time.sleep(1.5)
    return "A"


def test_parallel_evaluate_pairs_timeout_marks_pending_as_failures():
    pairs = [(0, 1)]
    items = [{"title": "A"}, {"title": "B"}]

    with patch.dict("os.environ", {"PAIR_EVAL_TIMEOUT_SECONDS": "1"}):
        results = parallel_evaluate_pairs(
            pairs=pairs,
            items=items,
            compare_fn=_slow_compare,
            idea_type="airesearch",
            concurrency=1,
            randomize_presentation=False,
        )

    assert results == [(0, 1, None)]


def test_parallel_evaluate_pairs_stop_request_marks_pending_as_failures():
    pairs = [(0, 1), (2, 3)]
    items = [
        {"title": "A"},
        {"title": "B"},
        {"title": "C"},
        {"title": "D"},
    ]

    with patch.dict("os.environ", {"PAIR_EVAL_TIMEOUT_SECONDS": "5"}):
        results = parallel_evaluate_pairs(
            pairs=pairs,
            items=items,
            compare_fn=_slow_compare,
            idea_type="airesearch",
            concurrency=2,
            randomize_presentation=False,
            should_stop=lambda: True,
        )

    assert len(results) == len(pairs)
    assert all(winner is None for _, _, winner in results)


def test_parallel_evaluate_pairs_preserves_exception_as_failed_evidence():
    failure_details = {}

    def explode(_a, _b, _idea_type):
        raise RuntimeError("judge unavailable")

    results = parallel_evaluate_pairs(
        pairs=[(0, 1)],
        items=[{"title": "A"}, {"title": "B"}],
        compare_fn=explode,
        idea_type="airesearch",
        concurrency=1,
        randomize_presentation=False,
        failure_details=failure_details,
    )

    assert results == [(0, 1, None)]
    assert failure_details == {(0, 1): "RuntimeError: judge unavailable"}


def test_parallel_evaluate_pairs_rejects_invalid_winner_instead_of_tie():
    failure_details = {}
    results = parallel_evaluate_pairs(
        pairs=[(0, 1)],
        items=[{"title": "A"}, {"title": "B"}],
        compare_fn=lambda _a, _b, _idea_type: "candidate-a",
        idea_type="airesearch",
        concurrency=1,
        randomize_presentation=False,
        failure_details=failure_details,
    )

    assert results == [(0, 1, None)]
    assert "invalid comparison winner" in failure_details[(0, 1)]
