from __future__ import annotations

import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from idea.models import CandidateRecord


class PairEvaluationError(RuntimeError):
    """One or more pair comparisons failed to produce judge evidence."""


class PairEvaluationInputError(PairEvaluationError, ValueError):
    """A comparison input, including required artifacts, is invalid."""


def extract_title_content(item: Any) -> Dict[str, Any]:
    """Best-effort extraction of a dict with title/content from various shapes.

    - If `item` is a CandidateRecord, unwrap its legacy candidate dictionary
    - If another `item` has .model_dump()/.dict(), use that
    - If it's a dict with title/content, return as-is
    - If it's a dict with nested 'idea', unwrap
    - Otherwise, coerce to a string content with Untitled title
    """
    if isinstance(item, CandidateRecord):
        item = item.to_candidate_dict()

    if hasattr(item, "model_dump") or hasattr(item, "dict"):
        try:
            d = item.model_dump() if hasattr(item, "model_dump") else item.dict()
            title = d.get("title", "Untitled")
            content = d.get("content", str(d))
            return {"title": title, "content": content}
        except Exception:
            pass

    if isinstance(item, dict):
        if "title" in item or "content" in item:
            extracted: Dict[str, Any] = {
                "title": item.get("title", "Untitled"),
                "content": item.get("content", str(item)),
            }
            # Prepared pairwise payloads carry already-loaded image bytes.
            # Keep those bytes paired with their owning candidate rather than
            # silently degrading the comparison to text-only input.
            if "images" in item:
                extracted["images"] = item["images"]
            return extracted
        if "idea" in item:
            inner = item["idea"]
            return extract_title_content(inner)

    return {"title": "Untitled", "content": str(item)}


def parallel_evaluate_pairs(
    pairs: Sequence[Tuple[int, int]],
    items: Sequence[Any],
    compare_fn: Callable[[Dict[str, Any], Dict[str, Any], str], Optional[str]],
    idea_type: str,
    concurrency: int = 8,
    randomize_presentation: bool = True,
    progress_callback: Callable[[int, int], None] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    failure_details: Optional[Dict[Tuple[int, int], str]] = None,
) -> List[Tuple[int, int, Optional[str]]]:
    """Run LLM comparisons for pairs in parallel and return winners.

    Returns a list of tuples (idx_a, idx_b, winner) in completion order.
    """
    results: List[Tuple[int, int, Optional[str]]] = []
    total_pairs = len(pairs)
    completed_count = 0
    try:
        pair_timeout = max(1.0, float(os.environ.get("PAIR_EVAL_TIMEOUT_SECONDS", "90")))
    except ValueError:
        pair_timeout = 90.0
    workers = max(1, min(32, int(concurrency)))
    executor = ThreadPoolExecutor(max_workers=workers)
    pending = {}
    remaining_pairs = iter(pairs)
    exhausted = False
    stop_reason = None

    def report(ia, ib, winner, error=None):
        nonlocal completed_count
        results.append((ia, ib, winner))
        if error and failure_details is not None:
            failure_details[(ia, ib)] = error
        completed_count += 1
        if progress_callback:
            try:
                progress_callback(completed_count, total_pairs)
            except Exception as exc:
                print(f"Error in progress callback: {exc}")

    def fill_window():
        nonlocal exhausted, stop_reason
        while not exhausted and len(pending) < workers:
            if callable(should_stop) and should_stop():
                stop_reason = "stop requested"
                return
            try:
                ia, ib = next(remaining_pairs)
            except StopIteration:
                exhausted = True
                return
            a = extract_title_content(items[ia])
            b = extract_title_content(items[ib])
            swap = random.random() < 0.5 if randomize_presentation else False
            def compare(a=a, b=b, swap=swap):
                if callable(should_stop) and should_stop():
                    return None
                winner = compare_fn(b, a, idea_type) if swap else compare_fn(a, b, idea_type)
                if swap and winner in {"A", "B"}:
                    winner = "B" if winner == "A" else "A"
                return winner
            future = executor.submit(compare)
            pending[future] = (ia, ib, time.monotonic())

    try:
        fill_window()
        while pending and not stop_reason:
            if callable(should_stop) and should_stop():
                stop_reason = "stop requested"
                break
            now = time.monotonic()
            if any(not future.done() and now - started >= pair_timeout
                   for future, (_, _, started) in pending.items()):
                stop_reason = f"timeout ({pair_timeout:.1f}s per comparison)"
                break
            done, _ = wait(tuple(pending), timeout=.1, return_when=FIRST_COMPLETED)
            for future in done:
                ia, ib, _ = pending.pop(future)
                error = None
                try:
                    winner = future.result()
                except Exception as exc:
                    winner = None
                    error = f"{exc.__class__.__name__}: {str(exc) or 'comparison failed'}"
                if winner not in {"A", "B", "tie"}:
                    error = error or f"invalid comparison winner: {winner!r}"
                    winner = None
                report(ia, ib, winner, error)
            fill_window()
        for future, (ia, ib, _) in pending.items():
            future.cancel()
            report(ia, ib, None, stop_reason or "comparison unfinished")
        for ia, ib in remaining_pairs:
            report(ia, ib, None, stop_reason or "comparison unfinished")
    finally:
        # Retain the run's claim until started provider calls finish. Python cannot
        # kill a running thread; provider transport deadlines bound this drain.
        executor.shutdown(wait=True, cancel_futures=True)
    return results


__all__ = [
    "PairEvaluationError",
    "PairEvaluationInputError",
    "extract_title_content",
    "parallel_evaluate_pairs",
]
