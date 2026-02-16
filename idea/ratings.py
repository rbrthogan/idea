from __future__ import annotations

import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def extract_title_content(item: Any) -> Dict[str, str]:
    """Best-effort extraction of a dict with title/content from various shapes.

    - If `item` has .dict(), use that
    - If it's a dict with title/content, return as-is
    - If it's a dict with nested 'idea', unwrap
    - Otherwise, coerce to a string content with Untitled title
    """
    if hasattr(item, "dict"):
        try:
            d = item.dict()
            title = d.get("title", "Untitled")
            content = d.get("content", str(d))
            return {"title": title, "content": content}
        except Exception:
            pass

    if isinstance(item, dict):
        if "title" in item or "content" in item:
            return {"title": item.get("title", "Untitled"), "content": item.get("content", str(item))}
        if "idea" in item:
            inner = item["idea"]
            return extract_title_content(inner)

    return {"title": "Untitled", "content": str(item)}


def parallel_evaluate_pairs(
    pairs: Sequence[Tuple[int, int]],
    items: Sequence[Any],
    compare_fn: Callable[[Dict[str, str], Dict[str, str], str], str],
    idea_type: str,
    concurrency: int = 8,
    randomize_presentation: bool = True,
    progress_callback: Callable[[int, int], None] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> List[Tuple[int, int, str]]:
    """Run LLM comparisons for pairs in parallel and return winners.

    Returns a list of tuples (idx_a, idx_b, winner) in completion order.
    """
    futures: Dict[Any, Tuple[int, int]] = {}
    results: List[Tuple[int, int, str]] = []
    total_pairs = len(pairs)
    completed_count = 0
    timeout_raw = os.environ.get("PAIR_EVAL_TIMEOUT_SECONDS", "90")
    try:
        pair_eval_timeout = max(1.0, float(timeout_raw))
    except ValueError:
        pair_eval_timeout = 90.0
    poll_seconds = 0.25

    executor = ThreadPoolExecutor(max_workers=max(1, int(concurrency)))
    try:
        for idx_a, idx_b in pairs:
            item_a = items[idx_a]
            item_b = items[idx_b]
            a_dict = extract_title_content(item_a)
            b_dict = extract_title_content(item_b)

            swap = random.random() < 0.5 if randomize_presentation else False

            def task(a=a_dict, b=b_dict, s=swap, ia=idx_a, ib=idx_b):
                if s:
                    w = compare_fn(b, a, idea_type)
                    if w == "A":
                        w = "B"
                    elif w == "B":
                        w = "A"
                else:
                    w = compare_fn(a, b, idea_type)
                return ia, ib, w

            future = executor.submit(task)
            futures[future] = (idx_a, idx_b)

        pending = set(futures.keys())
        deadline = time.monotonic() + pair_eval_timeout
        stop_reason = None

        while pending:
            if callable(should_stop) and should_stop():
                stop_reason = "stop requested"
                break

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stop_reason = f"timeout ({pair_eval_timeout:.1f}s)"
                break

            done, pending = wait(
                pending,
                timeout=min(poll_seconds, remaining),
                return_when=FIRST_COMPLETED,
            )

            for future in done:
                idx_a, idx_b = futures[future]
                try:
                    out_a, out_b, winner = future.result()
                except Exception:
                    out_a, out_b, winner = idx_a, idx_b, "tie"

                if winner not in {"A", "B", "tie"}:
                    winner = "tie"

                results.append((out_a, out_b, winner))
                completed_count += 1
                if progress_callback:
                    try:
                        progress_callback(completed_count, total_pairs)
                    except Exception as e:
                        print(f"Error in progress callback: {e}")

        if pending:
            if stop_reason:
                print(
                    f"parallel_evaluate_pairs ended early ({stop_reason}); "
                    f"marking {len(pending)} unfinished comparisons as ties."
                )
            for future in pending:
                idx_a, idx_b = futures[future]
                future.cancel()
                results.append((idx_a, idx_b, "tie"))
                completed_count += 1
                if progress_callback:
                    try:
                        progress_callback(completed_count, total_pairs)
                    except Exception as e:
                        print(f"Error in progress callback: {e}")
    finally:
        # Never block shutdown waiting on worker threads that might be stuck in network calls.
        executor.shutdown(wait=False, cancel_futures=True)

    return results
