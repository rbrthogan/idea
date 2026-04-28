from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def _idea_id(idea: Dict[str, Any], generation_index: int, idea_index: int) -> str:
    raw_id = idea.get("id")
    if raw_id is None or raw_id == "":
        return f"generated:{generation_index}:{idea_index}"
    return str(raw_id)


def _idea_title(idea: Dict[str, Any]) -> str:
    if idea.get("title"):
        return str(idea["title"])
    nested = idea.get("idea")
    if isinstance(nested, dict) and nested.get("title"):
        return str(nested["title"])
    return "Untitled"


def _coerce_optional_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite_optional_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _metric_value(idea: Dict[str, Any], metric: str) -> Optional[float]:
    selection_metrics = idea.get("selection_metrics")
    if isinstance(selection_metrics, dict):
        value = _finite_optional_float(selection_metrics.get(metric))
        if value is not None:
            return value

    if metric == "elo":
        ratings = idea.get("ratings")
        if isinstance(ratings, dict):
            value = _finite_optional_float(ratings.get("auto"))
            if value is not None:
                return value
        return _finite_optional_float(idea.get("elo"))

    if metric == "diversity":
        for key in ("diversity_score", "diversity"):
            value = _finite_optional_float(idea.get(key))
            if value is not None:
                return value
        return None

    if metric == "fitness":
        survival_metrics = idea.get("survival_metrics")
        if isinstance(survival_metrics, dict):
            fitness = _finite_optional_float(survival_metrics.get("survival_score"))
            if fitness is not None:
                return fitness
        return _finite_optional_float(idea.get("fitness"))

    return None


def _fitness_value(idea: Dict[str, Any]) -> Optional[float]:
    return _metric_value(idea, "fitness")


def _mean(values: List[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def _rank_desc(value: Optional[float], values: List[float]) -> Optional[int]:
    if value is None or not values:
        return None
    sorted_values = sorted(values, reverse=True)
    return next(
        (index + 1 for index, candidate in enumerate(sorted_values) if candidate <= value),
        len(sorted_values),
    )


def _edge_type(idea: Dict[str, Any], parent_id: str) -> str:
    migrated_from_id = idea.get("migrated_from_id")
    refined_from_id = idea.get("survivor_refined_from_id")
    if idea.get("migration_event") and migrated_from_id is not None and str(migrated_from_id) == parent_id:
        return "migration"
    if idea.get("survivor_refined") and refined_from_id is not None and str(refined_from_id) == parent_id:
        return "survivor_refinement"
    return "breeding"


def _parent_ids_for_idea(idea: Dict[str, Any]) -> List[str]:
    parent_ids = [str(parent_id) for parent_id in idea.get("parent_ids", []) if parent_id]
    migrated_from_id = idea.get("migrated_from_id")
    if idea.get("migration_event") and migrated_from_id:
        return [str(migrated_from_id)]
    refined_from_id = idea.get("survivor_refined_from_id")
    if idea.get("survivor_refined") and refined_from_id:
        return [str(refined_from_id)]
    return parent_ids


def _metric_generation(node: Dict[str, Any]) -> int:
    idea = node.get("idea", {})
    if isinstance(idea, dict):
        selection_metrics = idea.get("selection_metrics")
        if isinstance(selection_metrics, dict):
            generation = _coerce_optional_int(selection_metrics.get("evaluation_generation"))
            if generation is not None:
                return generation

        survival_metrics = idea.get("survival_metrics")
        if isinstance(survival_metrics, dict):
            generation = _coerce_optional_int(survival_metrics.get("survived_to_generation"))
            if generation is not None:
                return generation

    return int(node.get("last_generation", node.get("first_generation", 0)))


def _transitive_ids(start_id: str, adjacency: Dict[str, List[str]]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    stack = list(reversed(adjacency.get(start_id, [])))
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        ordered.append(node_id)
        stack.extend(reversed(adjacency.get(node_id, [])))
    return ordered


def _nearest_available_lane(preferred_lane: Optional[int], occupied_lanes: set[int]) -> int:
    if preferred_lane is None:
        lane = 0
        while lane in occupied_lanes:
            lane += 1
        return lane

    if preferred_lane >= 0 and preferred_lane not in occupied_lanes:
        return preferred_lane

    for distance in range(1, len(occupied_lanes) + preferred_lane + 3):
        left = preferred_lane - distance
        right = preferred_lane + distance
        if left >= 0 and left not in occupied_lanes:
            return left
        if right not in occupied_lanes:
            return right

    return len(occupied_lanes)


def build_evolution_tree(history: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Build a compressed lineage tree from serialized evolution history.

    A repeated idea ID across adjacent generations represents an unchanged
    survivor, so it is modelled as a continuation segment rather than another
    node.
    """
    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    occurrences_by_id: Dict[str, List[int]] = {}
    ordered_ids: List[str] = []

    for generation_index, generation in enumerate(history or []):
        for idea_index, idea in enumerate(generation or []):
            if not isinstance(idea, dict):
                continue

            node_id = _idea_id(idea, generation_index, idea_index)
            if node_id not in nodes_by_id:
                nodes_by_id[node_id] = {
                    "id": node_id,
                    "title": _idea_title(idea),
                    "first_generation": generation_index,
                    "last_generation": generation_index,
                    "first_index": idea_index,
                    "island_id": _coerce_optional_int(idea.get("island_id")),
                    "origin_island_id": _coerce_optional_int(idea.get("origin_island_id")),
                    "idea": idea,
                    "continuation_generations": [],
                    "lane_index": None,
                }
                ordered_ids.append(node_id)

            occurrences_by_id.setdefault(node_id, []).append(generation_index)
            nodes_by_id[node_id]["idea"] = idea
            nodes_by_id[node_id]["last_generation"] = max(
                nodes_by_id[node_id]["last_generation"],
                generation_index,
            )

    continuations: List[Dict[str, Any]] = []
    for node_id, generations in occurrences_by_id.items():
        unique_generations = sorted(set(generations))
        if not unique_generations:
            continue

        nodes_by_id[node_id]["continuation_generations"] = unique_generations[1:]
        for previous, current in zip(unique_generations, unique_generations[1:]):
            if current == previous + 1:
                continuations.append(
                    {
                        "id": f"{node_id}:{previous}->{current}",
                        "node_id": node_id,
                        "from_generation": previous,
                        "to_generation": current,
                    }
                )

    edges: List[Dict[str, Any]] = []
    seen_edges = set()
    for generation_index, generation in enumerate(history or []):
        for idea_index, idea in enumerate(generation or []):
            if not isinstance(idea, dict):
                continue
            child_id = _idea_id(idea, generation_index, idea_index)
            if nodes_by_id.get(child_id, {}).get("first_generation") != generation_index:
                continue

            for parent_id in _parent_ids_for_idea(idea):
                if parent_id not in nodes_by_id:
                    continue
                edge_key = (parent_id, child_id, generation_index)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                edges.append(
                    {
                        "id": f"{parent_id}->{child_id}:{generation_index}",
                        "from": parent_id,
                        "to": child_id,
                        "generation": generation_index,
                        "type": _edge_type(idea, parent_id),
                    }
                )

    previous_lanes_by_id: Dict[str, int] = {}
    max_lane_count = 0
    generations = []
    for generation_index, generation in enumerate(history or []):
        node_ids = []
        continuation_ids = []
        entries = []
        for idea_index, idea in enumerate(generation or []):
            if not isinstance(idea, dict):
                continue
            node_id = _idea_id(idea, generation_index, idea_index)
            is_new_node = nodes_by_id[node_id]["first_generation"] == generation_index
            kind = "node" if is_new_node else "continuation"
            if is_new_node:
                node_ids.append(node_id)
            else:
                continuation_ids.append(node_id)
            entries.append({"node_id": node_id, "kind": kind, "idea": idea})

        occupied_lanes: set[int] = set()
        current_lanes_by_id: Dict[str, int] = {}
        slots = []

        for entry in entries:
            if entry["kind"] != "continuation":
                continue
            node_id = entry["node_id"]
            preferred_lane = previous_lanes_by_id.get(node_id)
            if preferred_lane is None:
                preferred_lane = nodes_by_id[node_id].get("lane_index")
            lane = _nearest_available_lane(preferred_lane, occupied_lanes)
            occupied_lanes.add(lane)
            current_lanes_by_id[node_id] = lane
            if nodes_by_id[node_id].get("lane_index") is None:
                nodes_by_id[node_id]["lane_index"] = lane
            slots.append({"node_id": node_id, "kind": "continuation", "lane_index": lane})

        for entry in entries:
            if entry["kind"] != "node":
                continue
            node_id = entry["node_id"]
            parent_lanes = [
                previous_lanes_by_id[parent_id]
                for parent_id in _parent_ids_for_idea(entry["idea"])
                if parent_id in previous_lanes_by_id
            ]
            if parent_lanes:
                preferred_lane = round(sum(parent_lanes) / len(parent_lanes))
            else:
                preferred_lane = None
            lane = _nearest_available_lane(preferred_lane, occupied_lanes)
            occupied_lanes.add(lane)
            current_lanes_by_id[node_id] = lane
            nodes_by_id[node_id]["lane_index"] = lane
            slots.append({"node_id": node_id, "kind": "node", "lane_index": lane})

        slots.sort(key=lambda slot: (slot["lane_index"], slot["kind"], slot["node_id"]))
        if occupied_lanes:
            max_lane_count = max(max_lane_count, max(occupied_lanes) + 1)
        previous_lanes_by_id = current_lanes_by_id
        generations.append(
            {
                "generation": generation_index,
                "node_ids": node_ids,
                "continuation_ids": continuation_ids,
                "slots": slots,
            }
        )

    nodes = [nodes_by_id[node_id] for node_id in ordered_ids]
    metric_names = ("elo", "diversity", "fitness")
    metric_records: Dict[str, List[Dict[str, Any]]] = {name: [] for name in metric_names}
    for node in nodes:
        idea = node.get("idea", {})
        node["metric_generation"] = _metric_generation(node)
        node["metric_values"] = {
            name: _metric_value(idea, name) for name in metric_names
        }
        node["fitness_value"] = node["metric_values"]["fitness"]
        for name, value in node["metric_values"].items():
            if value is not None:
                metric_records[name].append(
                    {
                        "node_id": node["id"],
                        "generation": node["metric_generation"],
                        "value": value,
                    }
                )

    fitness_values = [record["value"] for record in metric_records["fitness"]]
    fitness_range = {
        "min": min(fitness_values) if fitness_values else None,
        "max": max(fitness_values) if fitness_values else None,
    }

    parents_by_id: Dict[str, List[str]] = {node_id: [] for node_id in ordered_ids}
    children_by_id: Dict[str, List[str]] = {node_id: [] for node_id in ordered_ids}
    for edge in edges:
        parent_id = edge["from"]
        child_id = edge["to"]
        parents_by_id.setdefault(child_id, []).append(parent_id)
        children_by_id.setdefault(parent_id, []).append(child_id)

    for node in nodes:
        node_id = node["id"]
        node["parent_ids"] = parents_by_id.get(node_id, [])
        node["child_ids"] = children_by_id.get(node_id, [])
        node["ancestor_ids"] = _transitive_ids(node_id, parents_by_id)
        node["descendant_ids"] = _transitive_ids(node_id, children_by_id)

    generation_fitness_stats: Dict[int, Dict[str, Any]] = {}
    for generation_index, generation in enumerate(history or []):
        values = [
            _fitness_value(idea)
            for idea in generation
            if isinstance(idea, dict)
        ]
        finite_values = [value for value in values if value is not None]
        generation_fitness_stats[generation_index] = {
            "mean": _mean(finite_values),
            "values": finite_values,
            "population_size": len(finite_values),
        }

    global_metric_stats: Dict[str, Dict[str, Any]] = {}
    generation_metric_stats: Dict[str, Dict[int, Dict[str, Any]]] = {
        name: {} for name in metric_names
    }
    for name, records in metric_records.items():
        global_values = [record["value"] for record in records]
        global_metric_stats[name] = {
            "mean": _mean(global_values),
            "values": global_values,
            "population_size": len(global_values),
        }
        generations_with_values = sorted({record["generation"] for record in records})
        for generation_index in generations_with_values:
            generation_values = [
                record["value"]
                for record in records
                if record["generation"] == generation_index
            ]
            history_values = [
                record["value"]
                for record in records
                if record["generation"] <= generation_index
            ]
            generation_metric_stats[name][generation_index] = {
                "generation_values": generation_values,
                "generation_mean": _mean(generation_values),
                "generation_population_size": len(generation_values),
                "history_values": history_values,
                "history_mean": _mean(history_values),
                "history_population_size": len(history_values),
            }

    for node in nodes:
        metric_generation = _metric_generation(node)
        metric_contexts: Dict[str, Dict[str, Any]] = {}
        for metric_name in metric_names:
            value = node["metric_values"].get(metric_name)
            global_stats = global_metric_stats.get(metric_name, {})
            scoped_stats = generation_metric_stats.get(metric_name, {}).get(
                metric_generation, {}
            )
            global_values = global_stats.get("values", [])
            history_values = scoped_stats.get("history_values", [])
            generation_values = scoped_stats.get("generation_values", [])
            global_mean = global_stats.get("mean")
            history_mean = scoped_stats.get("history_mean")
            generation_mean = scoped_stats.get("generation_mean")
            metric_contexts[metric_name] = {
                "value": value,
                "generation": metric_generation,
                "global_rank": _rank_desc(value, global_values),
                "global_population_size": global_stats.get("population_size", 0),
                "history_rank": _rank_desc(value, history_values),
                "history_population_size": scoped_stats.get("history_population_size", 0),
                "generation_rank": _rank_desc(value, generation_values),
                "generation_population_size": scoped_stats.get("generation_population_size", 0),
                "global_mean": global_mean,
                "history_mean": history_mean,
                "generation_mean": generation_mean,
                "delta_vs_global_mean": (
                    value - global_mean
                    if value is not None and global_mean is not None
                    else None
                ),
                "delta_vs_history_mean": (
                    value - history_mean
                    if value is not None and history_mean is not None
                    else None
                ),
                "delta_vs_generation_mean": (
                    value - generation_mean
                    if value is not None and generation_mean is not None
                    else None
                ),
            }

        value = node.get("fitness_value")
        parent_values = [
            nodes_by_id[parent_id].get("fitness_value")
            for parent_id in node.get("parent_ids", [])
            if parent_id in nodes_by_id and nodes_by_id[parent_id].get("fitness_value") is not None
        ]
        parent_average = (sum(parent_values) / len(parent_values)) if parent_values else None

        descendant_candidates = [
            (descendant_id, nodes_by_id[descendant_id].get("fitness_value"))
            for descendant_id in node.get("descendant_ids", [])
            if descendant_id in nodes_by_id and nodes_by_id[descendant_id].get("fitness_value") is not None
        ]
        best_descendant_id = None
        best_descendant_fitness = None
        if descendant_candidates:
            best_descendant_id, best_descendant_fitness = max(
                descendant_candidates,
                key=lambda item: item[1],
            )

        node["metric_contexts"] = metric_contexts
        node["fitness_context"] = {
            **metric_contexts["fitness"],
            "parent_average": parent_average,
            "delta_vs_parent_average": (
                value - parent_average
                if value is not None and parent_average is not None
                else None
            ),
            "best_descendant_id": best_descendant_id,
            "best_descendant_fitness": best_descendant_fitness,
            "delta_vs_best_descendant": (
                best_descendant_fitness - value
                if value is not None and best_descendant_fitness is not None
                else None
            ),
        }

    return {
        "nodes": nodes,
        "nodes_by_id": nodes_by_id,
        "edges": edges,
        "continuations": continuations,
        "generations": generations,
        "lane_count": max_lane_count,
        "fitness_range": fitness_range,
        "generation_fitness_stats": generation_fitness_stats,
        "global_metric_stats": global_metric_stats,
        "generation_metric_stats": generation_metric_stats,
    }
