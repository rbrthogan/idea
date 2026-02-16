import time
from unittest.mock import patch

from idea.ratings import parallel_evaluate_pairs


def _slow_compare(_a, _b, _idea_type):
    time.sleep(1.5)
    return "A"


def test_parallel_evaluate_pairs_timeout_marks_pending_as_ties():
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

    assert results == [(0, 1, "tie")]


def test_parallel_evaluate_pairs_stop_request_marks_pending_as_ties():
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
    assert all(winner == "tie" for _, _, winner in results)
