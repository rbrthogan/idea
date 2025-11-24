from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Sequence, Tuple


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
) -> List[Tuple[int, int, str]]:
    """Run LLM comparisons for pairs in parallel and return winners.

    Returns a list of tuples (idx_a, idx_b, winner) in completion order.
    """
    futures = []
    results: List[Tuple[int, int, str]] = []
    total_pairs = len(pairs)
    completed_count = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
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
                    elif w == "B":
                        w = "A"
                else:
                    w = compare_fn(a, b, idea_type)
                return ia, ib, w

            futures.append(executor.submit(task))

        for f in as_completed(futures):
            results.append(f.result())
            completed_count += 1
            if progress_callback:
                try:
                    progress_callback(completed_count, total_pairs)
                except Exception as e:
                    print(f"Error in progress callback: {e}")

    return results
