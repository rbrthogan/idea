"""Owner-scoped, recorded usage estimates. These records are not provider invoices."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Mapping

TOKEN_FIELDS = {
    "input_tokens": "input", "output_tokens": "output", "thought_tokens": "thoughts",
    "cached_input_tokens": "cached_input", "cache_write_input_tokens": "cache_write_input",
}
PRICE_FIELDS = ("pricing_known", "price_source", "price_version", "price_effective_date",
                "pricing_regime", "input_rate", "cached_input_rate", "cache_write_input_rate",
                "output_rate", "thought_rate", "large_prompt_tier")


def number(value: Any) -> float:
    try:
        result = float(value or 0)
        return max(result, 0) if math.isfinite(result) else 0.0
    except (TypeError, ValueError, OverflowError):
        return 0.0


def recorded_cost(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 and not isinstance(value, bool) else None
    except (TypeError, ValueError, OverflowError):
        return None


def provider_for_model(model: str) -> str:
    if model.startswith(("gemini", "models/gemini")):
        return "google_generative_ai"
    if model.startswith(("gpt", "o1", "o3", "o4", "text-embedding", "codex")):
        return "openai"
    if model.startswith(("ollama", "local")):
        return "local"
    return "unknown"


def record_price_usage(bins: dict, cost: Mapping, usage: Mapping) -> None:
    """Merge identical captured price schedules; never reprice historical counts."""
    price = {key: cost[key] for key in PRICE_FIELDS if key in cost}
    price.setdefault("pricing_known", False)
    key = json.dumps(price, sort_keys=True, default=str)
    row = bins.setdefault(key, {**price, "requests": 0, "cost": 0.0, "cache_savings": 0.0,
                                **{field: 0 for field in TOKEN_FIELDS.values()}, "estimated_input": 0})
    row["requests"] += 1
    row["cost"] += number(cost.get("total_cost"))
    # Savings exclude cache writes, whose premium is already in the recorded input cost.
    row["cache_savings"] += number(usage.get("cached_input")) * max(
        number(cost.get("input_rate")) - number(cost.get("cached_input_rate")), 0) / 1_000_000
    for field in TOKEN_FIELDS.values():
        row[field] += int(number(usage.get(field)))
    row["input"] += int(number(usage.get("tool_input")))
    row["estimated_input"] += int(number(usage.get("estimated_input")))


def usage_components(token_counts: Any) -> list[dict]:
    """Flatten restored checkpoints without counting their aggregate a second time."""
    if not isinstance(token_counts, dict):
        return []
    result = []
    for key, component in token_counts.items():
        if not isinstance(component, dict) or not ("model" in component or key == "restored_usage"):
            continue
        if isinstance(component.get("components"), list):
            result.extend(item for item in component["components"] if isinstance(item, dict))
        elif isinstance(component.get("per_model"), dict) and component["per_model"]:
            for model, counts in component["per_model"].items():
                if not isinstance(counts, dict):
                    continue
                result.append({**counts, "model": model, "provider": component.get("provider")})
        else:
            result.append(dict(component))
    return result


def compact_usage(token_counts: Any) -> dict:
    """Small manifest summary, retaining stored cost even when older attribution is sparse."""
    token_counts = token_counts if isinstance(token_counts, dict) else {}
    cost = token_counts.get("cost")
    saved_cost = recorded_cost(cost.get("total_cost")) if isinstance(cost, dict) else None
    models = []
    grouped: dict[tuple, dict] = {}
    components = usage_components(token_counts)
    for component in components:
        if not any(number(component.get(k)) for k in ("total", "input", "output", "thoughts", "cost", "requests", "missing_usage_requests")):
            continue
        model = str(component.get("model") or "unknown")
        provider = component.get("provider") or provider_for_model(model)
        identity = (provider, model)
        row = grouped.setdefault(identity, {"model": model, "provider": provider,
            "estimated_cost": 0.0, "cache_savings": 0.0, "estimated_input_tokens": 0,
            "missing_usage_requests": 0, "legacy_price": False, "pricing_known": True,
            "usage_basis": "provider_reported", "price_snapshots": [],
            **{field: 0 for field in TOKEN_FIELDS}})
        row["estimated_cost"] += number(component.get("cost"))
        row["cache_savings"] += number(component.get("cache_savings"))
        row["estimated_input_tokens"] += int(number(component.get("estimated_input")))
        row["missing_usage_requests"] += int(number(component.get("missing_usage_requests")))
        for field, old_field in TOKEN_FIELDS.items():
            row[field] += int(number(component.get(old_field)))
        row["input_tokens"] += int(number(component.get("tool_input")))
        snapshots = component.get("usage_by_price") or []
        snapshots = [item for item in snapshots if isinstance(item, dict)] if isinstance(snapshots, list) else []
        row["price_snapshots"].extend(snapshots)
        row["legacy_price"] |= not bool(snapshots)
        row["pricing_known"] &= all(s.get("pricing_known") is True for s in snapshots) if snapshots else component.get("pricing_known") is True
        if row["estimated_input_tokens"]:
            row["usage_basis"] = "includes_local_estimates"
    models = list(grouped.values())
    # Compact repeated resume snapshots by their schedule and sum usage.
    for row in models:
        bins = {}
        for snapshot in row["price_snapshots"]:
            identity = json.dumps({k: snapshot[k] for k in PRICE_FIELDS if k in snapshot}, sort_keys=True, default=str)
            if identity not in bins:
                bins[identity] = dict(snapshot)
            else:
                for key in ("cost", "cache_savings", "requests", "estimated_input", *TOKEN_FIELDS.values()):
                    bins[identity][key] = number(bins[identity].get(key)) + number(snapshot.get(key))
        row["price_snapshots"] = list(bins.values())
    summary = {"schema_version": 1, "estimated_cost": saved_cost,
               "models": models, "has_usage": bool(models), "legacy_price": any(m["legacy_price"] for m in models) or (saved_cost is not None and saved_cost > 0 and not models),
               "cache_savings": sum(m["cache_savings"] for m in models),
               "estimated_input_tokens": sum(m["estimated_input_tokens"] for m in models),
               "missing_usage_requests": sum(m["missing_usage_requests"] for m in models)}
    for field, old in TOKEN_FIELDS.items():
        summary[field] = int(number(token_counts.get("total_" + old)))
        if "total_" + old not in token_counts:
            summary[field] = sum(m[field] for m in models)
    total_tool_input = int(number(token_counts.get("total_tool_input")))
    if "total_input" in token_counts:
        summary["input_tokens"] += total_tool_input
    else:
        # Model rows already include their tool input. Only add any aggregate
        # tool usage whose component details were absent in a legacy record.
        component_tool_input = sum(int(number(c.get("tool_input"))) for c in components)
        summary["input_tokens"] += max(total_tool_input - component_tool_input, 0)
    return summary


def build_billing_response(runs: list[dict], events: list[dict], *, truncated: bool = False) -> dict:
    # A checkpoint and a completed export of the same evolution are cumulative alternatives.
    unique = {}
    for run in runs:
        identity = str(run.get("evolution_id") or run.get("id"))
        previous = unique.get(identity)
        if previous is None or str(run.get("updated_at", "")) > str(previous.get("updated_at", "")):
            unique[identity] = run
    event_groups = defaultdict(list)
    for event in {str(e.get("id")): e for e in events}.values():
        event_groups[str(event.get("evolution_id") or "unassigned")].append(event)
    rows = []
    for identity in dict.fromkeys([*unique, *event_groups]):
        run = unique.get(identity, {})
        base = run.get("_billing_summary") or compact_usage(run.get("token_counts"))
        rating_summaries = [e.get("summary") or {} for e in event_groups[identity]]
        evolution_cost = recorded_cost(base.get("estimated_cost"))
        rerating = sum(number(s.get("estimated_cost")) for s in rating_summaries)
        row = {"id": identity, "name": run.get("evolution_name") or run.get("name") or identity,
               "status": run.get("status", "saved"), "updated_at": run.get("updated_at"),
               "evolution_cost": evolution_cost, "rerating_cost": rerating,
               "estimated_cost": (number(evolution_cost) + rerating) if evolution_cost is not None or rating_summaries else None,
               "models": [*base.get("models", []), *(m for s in rating_summaries for m in s.get("models", []))],
               "legacy_price": base.get("legacy_price", False), "has_usage": base.get("has_usage", False) or bool(rating_summaries)}
        for key in (*TOKEN_FIELDS, "cache_savings", "estimated_input_tokens", "missing_usage_requests"):
            row[key] = number(base.get(key)) + sum(number(s.get(key)) for s in rating_summaries)
        rows.append(row)
    totals = {key: sum(number(r.get(key)) for r in rows) for key in ("estimated_cost", *TOKEN_FIELDS, "cache_savings", "rerating_cost")}
    totals["run_count"] = len(rows)
    if rows and not any(r["estimated_cost"] is not None for r in rows):
        totals["estimated_cost"] = None
    by_model, by_provider = {}, {}
    for run in rows:
        for model in run["models"]:
            for mapping, key, ids in ((by_model, (model["provider"], model["model"]), {"model": model["model"], "provider": model["provider"]}),
                                      (by_provider, model["provider"], {"provider": model["provider"]})):
                target = mapping.setdefault(key, {**ids, **{k: 0.0 for k in ("estimated_cost", *TOKEN_FIELDS, "cache_savings")}})
                for field in ("estimated_cost", *TOKEN_FIELDS, "cache_savings"):
                    target[field] += number(model.get(field))
    return {"status": "success", "currency": "USD", "basis": "recorded_estimate", "totals": totals,
            "coverage": {"runs_with_recorded_cost": sum(r["evolution_cost"] is not None for r in rows),
                         "runs_missing_cost": sum(r["evolution_cost"] is None for r in rows),
                         "runs_missing_usage": sum(not r["has_usage"] for r in rows),
                         "legacy_price_runs": sum(r["legacy_price"] for r in rows),
                         "estimated_input_tokens": sum(r["estimated_input_tokens"] for r in rows),
                         "missing_usage_requests": sum(r["missing_usage_requests"] for r in rows),
                         "unpriced_models": sorted({m["model"] for r in rows for m in r["models"] if not m["pricing_known"]}),
                         "invoice_available": False, "truncated": truncated},
            "runs": sorted(rows, key=lambda r: str(r.get("updated_at") or ""), reverse=True),
            "by_model": list(by_model.values()), "by_provider": list(by_provider.values()),
            "dashboards": [{"provider": "google_generative_ai", "label": "Google Cloud billing", "url": "https://console.cloud.google.com/billing"},
                           {"provider": "openai", "label": "OpenAI usage", "url": "https://platform.openai.com/usage"}],
            "notes": ["Recorded estimates use captured token prices, not provider invoices. Credits, taxes, discounts, cache storage, infrastructure and local hardware costs are not available from generation responses.",
                      "Historical records without price snapshots retain their saved cost. Embedding tokens may include local estimates when provider usage was absent.",
                      "Standalone template design requests are outside recorded evolution/rerating totals.",
                      "Only this owner's recorded runs and rerating requests are included. In-flight work appears after its next checkpoint. New billing records survive run deletion; previously deleted or unowned legacy exports cannot be recovered."]}
