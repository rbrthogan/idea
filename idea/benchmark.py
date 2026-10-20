"""Offline, matched-seed comparison of exported runs and supplied holdout evidence.

This module never generates candidates or calls evaluators. Acquisition costs
and independent holdout measurements must be supplied by the experimenter.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


STRATEGIES = ("evolution", "independent_generation", "iterative_refinement")


def _finite(value: Any, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a finite number") from None
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError(f"{field} must be finite{' and nonnegative' if nonnegative else ''}")
    return number


def _exported_candidates(export: dict) -> dict[str, Any]:
    candidates = {}
    revisions = [candidate for generation in export.get("history", []) for candidate in generation]
    revisions += export.get("population", []) + export.get("quality_archive", [])
    for candidate in revisions:
        if not isinstance(candidate, dict) or not candidate.get("id"):
            continue
        identity = str(candidate["id"])
        content = candidate.get("idea")
        if identity in candidates and candidates[identity] != content:
            raise ValueError(f"Export reuses candidate ID {identity!r} for different content")
        candidates[identity] = content
    if not candidates:
        raise ValueError("Export contains no candidate revisions")
    return candidates


def _standard_error(values: list[float]) -> float | None:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None


def order_swap_report(judgments: list[dict]) -> dict:
    """Resolve presentation labels back to candidate identity before comparing."""
    agreements = 0
    first_position_wins = 0
    decisive = 0
    seen = set()
    if not isinstance(judgments, list):
        raise ValueError("Order-swap judgments must be a list")
    for judgment in judgments:
        if not isinstance(judgment, dict):
            raise ValueError("Each order-swap judgment must be an object")
        pair_id = judgment.get("pair_id")
        if not pair_id or pair_id in seen:
            raise ValueError("Order-swap pair IDs must be present and unique")
        seen.add(pair_id)
        first, second = judgment.get("candidate_a"), judgment.get("candidate_b")
        if not first or not second or first == second:
            raise ValueError("Order-swap judgments require two distinct candidate IDs")
        ab, ba = judgment.get("winner_ab"), judgment.get("winner_ba")
        if ab not in {"A", "B", "tie"} or ba not in {"A", "B", "tie"}:
            raise ValueError("Order-swap winners must be A, B, or tie")
        identity_ab = first if ab == "A" else second if ab == "B" else None
        identity_ba = second if ba == "A" else first if ba == "B" else None
        agreements += identity_ab == identity_ba
        for winner in (ab, ba):
            if winner != "tie":
                decisive += 1
                first_position_wins += winner == "A"
    return {"pairs": len(judgments), "agreements": agreements,
            "agreement_rate": agreements / len(judgments) if judgments else None,
            "first_position_win_fraction": first_position_wins / decisive if decisive else None}


def compare_manifest(manifest: dict, *, budget_usd: float, base_dir: Path = Path(".")) -> dict:
    if not isinstance(manifest, dict):
        raise ValueError("Benchmark manifest must be a JSON object")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark manifest schema version")
    budget = _finite(budget_usd, "budget_usd", nonnegative=True)
    if budget == 0:
        raise ValueError("budget_usd must be positive")
    evaluator = manifest.get("evaluator") or {}
    if not isinstance(evaluator, dict):
        raise ValueError("Evaluator definition must be an object")
    if not evaluator.get("id") or not evaluator.get("version") or evaluator.get("holdout") is not True:
        raise ValueError("Explicit evaluator id, version, and holdout=true are required")
    direction = evaluator.get("direction")
    if direction not in {"maximize", "minimize"}:
        raise ValueError("Evaluator direction must be maximize or minimize")
    if manifest.get("spend_basis") != "generation_selection_artifacts":
        raise ValueError("Spend must include generation, selection, and artifacts")
    bounds = evaluator.get("score_bounds")
    if bounds is not None:
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError("score_bounds must contain lower and upper bounds")
        bounds = [_finite(value, "score_bounds") for value in bounds]
        if bounds[0] >= bounds[1]:
            raise ValueError("score_bounds must be increasing")

    results = {}
    runs = manifest.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("Runs must be a list")
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("Each run must be an object")
        strategy = run.get("strategy")
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy!r}")
        seed = run.get("seed")
        template = run.get("template_id")
        if type(seed) is not int or not isinstance(template, str) or not template.strip():
            raise ValueError("Each run requires an integer seed and template_id")
        key = (template, seed, strategy)
        if key in results:
            raise ValueError(f"Duplicate template/seed/strategy {key!r}")
        export = run.get("export")
        if export is None:
            path = run.get("export_path")
            if not isinstance(path, str):
                raise ValueError("Each run requires export or export_path")
            export = json.loads((base_dir / path).read_text())
        if not isinstance(export, dict):
            raise ValueError("Run export must be a JSON object")
        config = export.get("config") or {}
        if config.get("random_seed") is not None and config["random_seed"] != seed:
            raise ValueError("Manifest seed does not match exported run")
        if config.get("idea_type") and config["idea_type"] != template:
            raise ValueError("Manifest template does not match exported run")
        candidates = _exported_candidates(export)
        spend = _finite((export.get("token_counts") or {}).get("cost", {}).get("total_cost"),
                        "export.token_counts.cost.total_cost", nonnegative=True)
        if spend + 1e-9 < budget:
            raise ValueError(f"Run {key!r} did not cover the common spend ceiling")
        eligible = []
        seen_ids = set()
        previous_spend = 0.0
        measurements = run.get("measurements", [])
        if not isinstance(measurements, list):
            raise ValueError("Measurements must be a list")
        for observation in measurements:
            if not isinstance(observation, dict):
                raise ValueError("Each measurement must be an object")
            identity = str(observation.get("candidate_id") or "")
            if identity not in candidates or identity in seen_ids:
                raise ValueError("Measurement candidate IDs must exist in the export and be unique")
            seen_ids.add(identity)
            cost = _finite(observation.get("cumulative_cost_usd"), "cumulative_cost_usd", nonnegative=True)
            if cost < previous_spend or cost > spend + 1e-9:
                raise ValueError("Cumulative spend must be ordered and no greater than exported spend")
            previous_spend = cost
            if type(observation.get("feasible")) is not bool:
                raise ValueError("Each measurement requires an explicit feasible boolean")
            score = _finite(observation.get("holdout_score"), "holdout_score")
            if bounds is not None and not bounds[0] <= score <= bounds[1]:
                raise ValueError("Holdout score is outside the evaluator's bounds")
            if cost <= budget + 1e-9:
                eligible.append({"candidate_id": identity, "cost": cost, "score": score,
                                 "feasible": observation["feasible"]})
        # Selective holdout measurements would bias best-of-run comparisons.
        if seen_ids != set(candidates):
            raise ValueError("Supply holdout measurements for every exported candidate revision")
        feasible = [observation for observation in eligible if observation["feasible"]]
        winner = (max if direction == "maximize" else min)(feasible,
                        key=lambda observation: observation["score"]) if feasible else None
        results[key] = {"template_id": template, "seed": seed, "strategy": strategy,
                        "eligible_candidates": len(eligible), "feasible_candidates": len(feasible),
                        "feasibility_rate": len(feasible)/len(eligible) if eligible else None,
                        "best_holdout_score": winner["score"] if winner else None,
                        "best_candidate_id": winner["candidate_id"] if winner else None}
    if not results:
        raise ValueError("No benchmark runs supplied")
    groups = {(template, seed) for template, seed, _ in results}
    for template, seed in groups:
        if any((template, seed, strategy) not in results for strategy in STRATEGIES):
            raise ValueError("Every template/seed requires all three strategies")

    comparisons = []
    sign = 1 if direction == "maximize" else -1
    for template in sorted({template for template, _ in groups}):
        seeds = sorted(seed for group_template, seed in groups if group_template == template)
        for baseline in STRATEGIES[1:]:
            deltas, unscorable = [], []
            for seed in seeds:
                evolved = results[(template, seed, "evolution")]["best_holdout_score"]
                control = results[(template, seed, baseline)]["best_holdout_score"]
                if evolved is None or control is None:
                    unscorable.append(seed)
                else:
                    deltas.append(sign*(evolved-control))
            comparisons.append({"template_id": template, "baseline": baseline,
                "paired_seeds": len(deltas), "unscorable_seeds": unscorable,
                "mean_paired_improvement": statistics.mean(deltas) if deltas else None,
                "paired_standard_error": _standard_error(deltas)})
    return {"schema_version": 1, "evaluator": evaluator, "spend_basis": manifest["spend_basis"],
            "budget_ceiling_usd": budget, "runs": [results[key] for key in sorted(results)], "comparisons": comparisons,
            "order_swap": order_swap_report(manifest.get("order_swap_judgments", [])),
            "interpretation": "Positive paired differences favor evolution under this supplied evaluator; "
                              "this report does not establish live model or algorithm superiority."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--budget", type=float, required=True, help="Common spend ceiling in USD")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = compare_manifest(json.loads(args.manifest.read_text()), budget_usd=args.budget,
                                  base_dir=args.manifest.parent)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
