#!/usr/bin/env python3
"""Annotate copied candidates or a saved history without running evolution.

Provider credentials come only from the environment. To load the local .env
without executing it, use ``uv run --env-file .env python scripts/observe_genes.py``.
The input export and remote storage are never changed.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idea.config import DEFAULT_MODELS_BY_PROVIDER, LLM_MODELS, THINKING_MODEL_CONFIG, get_model_provider
from idea.genes import empty_gene_catalogue, reconcile_gene_catalogue, valid_gene_observation
from idea.llm import Breeder
from idea.models import candidate_from_wire, candidate_to_wire
from idea.prompts.loader import get_prompts


def select_candidates(export: dict, limit: int, generation: int | None = None) -> tuple[list[dict], int]:
    """Read current population or an explicit history snapshot, de-duplicating revisions."""
    if limit < 1:
        raise ValueError("--limit must be at least 1")
    history = export.get("history", [])
    if generation is not None:
        if generation < 0 or not isinstance(history, list) or generation >= len(history):
            raise ValueError("--generation must name an existing zero-based history snapshot")
        source = history[generation]
        original_generation = generation
    elif isinstance(export.get("population"), list) and export["population"]:
        source = export["population"]
        original_generation = int(export.get("current_generation") or 0)
    elif isinstance(history, list) and history:
        source = history[-1]
        original_generation = len(history) - 1
    else:
        raise ValueError("Export has no population or history to annotate")
    if not isinstance(source, list):
        raise ValueError("Selected population must be a list")

    selected, seen = [], set()
    for raw in source:
        if not isinstance(raw, dict):
            continue
        candidate = copy.deepcopy(raw)
        # Viewer history exports flatten the document, unlike checkpoint records.
        if "idea" not in candidate:
            if not isinstance(candidate.get("content"), str):
                continue
            candidate["idea"] = {"title": candidate.pop("title", None), "content": candidate.pop("content")}
        document = candidate["idea"]
        content = document.get("content", "") if isinstance(document, dict) else document
        if not isinstance(content, str) or not content.strip():
            continue
        digest = hashlib.sha256(json.dumps(document, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        identity = str(candidate.get("id") or f"observation-{digest}")
        revision = (identity, digest)
        if revision in seen:
            continue
        seen.add(revision)
        candidate["id"] = identity
        selected.append(candidate_from_wire(candidate))
        if len(selected) >= limit:
            break
    if not selected:
        raise ValueError("Selected population has no readable candidate documents")
    return selected, original_generation


def candidate_revision(raw: dict) -> tuple[tuple[str, str], dict] | None:
    """Identify the phenotype alone; scores and other snapshot metadata may vary."""
    if not isinstance(raw, dict):
        return None
    candidate = copy.deepcopy(raw)
    document = candidate.get("idea", candidate)
    while isinstance(document, dict) and "idea" in document:
        document = document["idea"]
    if isinstance(document, str):
        title, content = None, document
    elif isinstance(document, dict):
        title, content = document.get("title"), document.get("content")
    else:
        title, content = getattr(document, "title", None), getattr(document, "content", None)
    if not isinstance(content, str) or not content.strip():
        return None
    digest = hashlib.sha256(json.dumps({"title": title, "content": content},
                                      sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    identity = str(candidate.get("id") or f"observation-{digest}")
    candidate["id"] = identity
    candidate["idea"] = {"title": title, "content": content}
    return (identity, digest), candidate_from_wire(candidate)


def spread_revisions(revisions: list[tuple[str, str]], limit: int | None) -> list[tuple[str, str]]:
    """Choose evenly spaced first appearances, including both ends when possible."""
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")
    if limit is None or limit >= len(revisions):
        return revisions[:]
    if limit == 1:
        return revisions[:1]
    return [revisions[round(index * (len(revisions) - 1) / (limit - 1))]
            for index in range(limit)]


def reusable_observations(export: dict | None) -> dict[tuple[str, str], list[dict]]:
    """Index potential raw observations; the active breeder validates fingerprints."""
    indexed = {}
    if not isinstance(export, dict):
        return indexed
    snapshots = [export.get("population", [])]
    if isinstance(export.get("history"), list):
        snapshots.extend(export["history"])
    for snapshot in snapshots:
        if not isinstance(snapshot, list):
            continue
        for raw in snapshot:
            revision = candidate_revision(raw)
            observation = raw.get("gene_observation") if isinstance(raw, dict) else None
            if revision is not None and isinstance(observation, dict):
                indexed.setdefault(revision[0], []).append(observation)
    return indexed


def reset_mapping(observation: dict) -> dict:
    raw = copy.deepcopy(observation)
    for gene in raw.get("genes", []):
        gene["gene_id"] = None
    raw["mapping_status"] = "pending"
    raw.pop("mapping_revision", None)
    raw.pop("mapping_error", None)
    return raw


def usage_payload(breeder) -> dict:
    usage = breeder.get_token_usage_summary()
    costs = {key: usage.get(key, 0.0) for key in (
        "input_cost", "uncached_input_cost", "cached_input_cost", "cache_write_input_cost",
        "output_cost", "thought_cost",
    )}
    costs.update(total_cost=usage.get("cost", 0.0), currency="USD")
    return {
        "breeder": usage,
        "total": usage.get("total", 0),
        **{f"total_{key}": usage.get(key, 0) for key in (
            "input", "output", "thoughts", "cached_input", "cache_write_input", "uncached_input", "tool_input",
        )},
        "models": {"breeder": breeder.model_name},
        "cost": costs,
    }


def write_progress(output: Path, result: dict) -> None:
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def observe_export(export: dict, *, input_path: Path, output_path: Path, candidates: list[dict],
                   original_generation: int, breeder, template_data: dict,
                   max_budget: float) -> dict:
    """Write a separate observation export; the injected provider permits offline testing."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output must be different from the input export")
    if not math.isfinite(max_budget) or max_budget < 0:
        raise ValueError("--max-budget must be a finite non-negative amount")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the path before scheduling any provider request; never overwrite an artifact.
    with output_path.open("x") as handle:
        handle.write("{}\n")

    config = copy.deepcopy(export.get("config") or {})
    idea_type = str(config.get("idea_type") or "airesearch")
    source_config = copy.deepcopy(config)
    config.update(pop_size=len(candidates), generations=1, model_type=breeder.model_name,
                  max_budget=max_budget)
    for config_key, attribute in (("creative_temp", "temperature"), ("top_p", "top_p"),
                                  ("thinking_level", "thinking_level"), ("thinking_budget", "thinking_budget"),
                                  ("random_seed", "random_seed")):
        if hasattr(breeder, attribute):
            config[config_key] = getattr(breeder, attribute)
    # The output is a fresh, single-snapshot observation experiment. Starting a
    # new catalogue avoids mixing source-generation counts into generation zero.
    catalogue = empty_gene_catalogue()
    result = {
        "evolution_id": f"gene-observation-{uuid.uuid4()}",
        "name": f"{export.get('name') or input_path.stem} — gene observation",
        "status": "in_progress", "observation_only": True, "current_generation": 0,
        "original_generation": original_generation,
        "config": config, "template_data": copy.deepcopy(template_data),
        "quality_archive": [],
        "gene_observation_run": {
            "version": 1, "source_path": str(input_path.resolve()),
            "source_evolution_id": export.get("evolution_id"),
            "source_config": source_config,
            "source_catalogue_revision": (export.get("gene_catalogue") or {}).get("revision")
            if isinstance(export.get("gene_catalogue"), dict) else None,
            "original_generation": original_generation,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": breeder.provider, "model": breeder.model_name,
            "max_budget_usd": max_budget, "selection": "first unique revisions in source order",
            "source_scores_preserved": True,
            "scope": "Gene extraction and catalogue matching only; no births, judging or selection.",
        },
    }
    breeder.cancel_check = lambda: float(breeder.total_cost) >= max_budget

    def save(status: str) -> None:
        population = [candidate_to_wire(candidate) for candidate in candidates]
        result.update(status=status, population=population, history=[copy.deepcopy(population)],
                      gene_catalogue=copy.deepcopy(catalogue), token_counts=usage_payload(breeder))
        result["gene_observation_run"]["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_progress(output_path, result)

    save("in_progress")
    try:
        for index, candidate in enumerate(candidates):
            if breeder.cancel_check():
                break
            breeder.observe_genes(candidate, idea_type)
            save("in_progress")
            print(f"Observed {index + 1}/{len(candidates)}; estimated spend ${breeder.total_cost:.6f}", flush=True)
        # This can still resolve exact known aliases after the budget stops calls.
        catalogue = reconcile_gene_catalogue(candidates, catalogue, breeder, idea_type, 0)
        complete = all(
            candidate.get("gene_observation", {}).get("status") == "ok"
            and candidate.get("gene_observation", {}).get("mapping_status") == "ok"
            for candidate in candidates
        )
        result["gene_observation_run"]["budget_reached"] = breeder.cancel_check()
        save("complete" if complete else "paused" if breeder.cancel_check() else "error")
    except BaseException as exc:
        result["gene_observation_run"]["error_type"] = type(exc).__name__
        save("paused" if isinstance(exc, KeyboardInterrupt) else "error")
        raise
    return result


def observe_history(export: dict, *, input_path: Path, output_path: Path, breeder,
                    template_data: dict, max_budget: float, limit: int | None = None,
                    reuse_export: dict | None = None, reuse_path: Path | None = None) -> dict:
    """Annotate copied snapshots chronologically without rerunning evolution.

    Only annotation fields change on source candidates. Repeated survivors share
    one raw extraction; each snapshot is matched against the catalogue as it
    existed at that generation. Provider fingerprints include the active prompt
    and model settings, so stale pilot observations cannot silently be reused.
    """
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output must be different from the input export")
    if not math.isfinite(max_budget) or max_budget < 0:
        raise ValueError("--max-budget must be a finite non-negative amount")
    history = export.get("history")
    if not isinstance(history, list) or not history or any(not isinstance(row, list) for row in history):
        raise ValueError("--history requires a nonempty list of population snapshots")
    result = copy.deepcopy(export)
    candidates, appearances, stages = {}, {}, []

    def register(snapshot: list, generation: int) -> list[tuple[dict, tuple[str, str]]]:
        records = []
        for raw in snapshot:
            revision = candidate_revision(raw)
            if revision is None:
                continue
            key, candidate = revision
            candidates.setdefault(key, candidate)
            appearances.setdefault(key, []).append(raw)
            # All canonical IDs belong to the newly rebuilt catalogue. Missing
            # pilot coverage must remain missing even if the input has old IDs.
            raw.pop("gene_observation", None)
            records.append((raw, key))
        return records

    for generation, snapshot in enumerate(result["history"]):
        stages.append((generation, register(snapshot, generation)))
    if isinstance(result.get("population"), list) and result["population"]:
        population_records = register(result["population"], len(history) - 1)
        if {key for _, key in population_records} != {key for _, key in stages[-1][1]}:
            # Some checkpoints include an additional completed current population.
            generation = max(len(history) - 1, int(export.get("current_generation") or 0))
            stages.append((generation, population_records))
    else:
        result["population"] = copy.deepcopy(result["history"][-1])
        population_records = register(result["population"], len(history) - 1)
    if not candidates:
        raise ValueError("History has no readable candidate documents")
    selected = set(spread_revisions(list(candidates), limit))
    reuse_pool = reusable_observations(export)
    for key, observations in reusable_observations(reuse_export).items():
        reuse_pool.setdefault(key, [])[:0] = observations
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as handle:
        handle.write("{}\n")

    idea_type = str((export.get("config") or {}).get("idea_type") or "airesearch")
    catalogue = empty_gene_catalogue()
    raw_cache, latest_mappings, attempted, reused = {}, {}, set(), set()
    source_token_counts = copy.deepcopy(export.get("token_counts"))
    result.update(evolution_id=f"gene-observation-{uuid.uuid4()}",
                  name=f"{export.get('name') or input_path.stem} — history gene observation",
                  status="in_progress", observation_only=True,
                  template_data=copy.deepcopy(template_data))
    result["gene_observation_run"] = {
        "version": 1, "mode": "history", "source_path": str(input_path.resolve()),
        "source_evolution_id": export.get("evolution_id"),
        "source_config": copy.deepcopy(export.get("config") or {}),
        "source_token_counts": source_token_counts,
        "source_catalogue_revision": (export.get("gene_catalogue") or {}).get("revision")
        if isinstance(export.get("gene_catalogue"), dict) else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": breeder.provider, "model": breeder.model_name,
        "max_budget_usd": max_budget,
        "selection": "all unique revisions" if limit is None else "evenly spaced unique revisions by first appearance",
        "selected_unique_revisions": len(selected), "total_unique_revisions": len(candidates),
        "total_history_snapshots": len(history), "completed_snapshots": 0,
        "selected_candidate_revisions": [{"id": key[0], "phenotype_digest": key[1]}
                                         for key in candidates if key in selected],
        "reuse_source_path": str(reuse_path.resolve()) if reuse_path else None,
        "source_scores_preserved": True, "generation_coverage": [],
        "scope": "Gene extraction and catalogue matching only; no births, judging or selection.",
    }
    metadata = result["gene_observation_run"]
    breeder.cancel_check = lambda: float(breeder.total_cost) >= max_budget

    def save(status="in_progress"):
        observed = sum(observation.get("status") == "ok" for observation in raw_cache.values())
        mapped = sum(observation.get("mapping_status") == "ok" for observation in latest_mappings.values())
        metadata.update(updated_at=datetime.now(timezone.utc).isoformat(),
                        attempted_extractions=len(attempted), reused_extractions=len(reused),
                        observed_unique_revisions=observed, mapped_unique_revisions=mapped,
                        selected_coverage=observed / len(selected),
                        total_coverage=observed / len(candidates),
                        budget_reached=breeder.cancel_check())
        result.update(status=status, gene_catalogue=copy.deepcopy(catalogue), token_counts=usage_payload(breeder))
        write_progress(output_path, result)

    save()
    try:
        for stage_index, (generation, records) in enumerate(stages):
            metadata.update(active_generation=generation, phase="extracting")
            for _, key in records:
                if key not in selected or key in raw_cache:
                    continue
                candidate = candidates[key]
                candidate.pop("gene_observation", None)
                fingerprint_method = getattr(breeder, "_genotype_prompt_and_fingerprint", None)
                if callable(fingerprint_method):
                    _, fingerprint = fingerprint_method(candidate, idea_type)
                    prior = next((observation for observation in reuse_pool.get(key, [])
                                  if valid_gene_observation(observation, fingerprint)), None)
                else:
                    prior = None
                if prior is not None:
                    observation = reset_mapping(prior)
                    reused.add(key)
                elif breeder.cancel_check():
                    continue
                else:
                    attempted.add(key)
                    breeder.observe_genes(candidate, idea_type)
                    observation = reset_mapping(candidate.get("gene_observation") or {
                        "status": "failed", "genes": [], "error": "Gene extraction returned no observation."})
                raw_cache[key] = observation
                for occurrence in appearances[key]:
                    occurrence["gene_observation"] = copy.deepcopy(observation)
                save()
                print(f"Observed {len(raw_cache)}/{len(selected)} selected revisions "
                      f"({len(candidates)} total; {len(reused)} reused); generation {generation}; "
                      f"estimated spend ${breeder.total_cost:.6f}", flush=True)
            metadata["phase"] = "matching"
            runtime = [candidate_revision(raw)[1] for raw, _ in records]
            catalogue = reconcile_gene_catalogue(runtime, catalogue, breeder, idea_type, generation)
            for (raw, key), candidate in zip(records, runtime):
                observation = candidate.get("gene_observation")
                if isinstance(observation, dict):
                    raw["gene_observation"] = copy.deepcopy(observation)
                    latest_mappings[key] = observation
            unique = {key for _, key in records}
            metadata["generation_coverage"].append({
                "generation": generation, "total_unique_revisions": len(unique),
                "selected_unique_revisions": len(unique & selected),
                "observed_unique_revisions": sum(raw_cache.get(key, {}).get("status") == "ok" for key in unique),
                "mapped_unique_revisions": sum(latest_mappings.get(key, {}).get("mapping_status") == "ok" for key in unique),
            })
            metadata["completed_snapshots"] = min(stage_index + 1, len(history))
            save()
            print(f"Matched snapshot {stage_index + 1}/{len(stages)} (generation {generation}); "
                  f"{len(catalogue['genes'])} catalogue genes; estimated spend ${breeder.total_cost:.6f}", flush=True)
        for raw, key in population_records:
            if key in latest_mappings:
                raw["gene_observation"] = copy.deepcopy(latest_mappings[key])
        complete = all(raw_cache.get(key, {}).get("status") == "ok" for key in selected) and all(
            raw.get("gene_observation", {}).get("mapping_status") == "ok"
            for _, records in stages for raw, key in records if key in selected)
        metadata["phase"] = "finished"
        save("complete" if complete else "paused" if breeder.cancel_check() else "error")
    except BaseException as exc:
        metadata["error_type"] = type(exc).__name__
        save("paused" if isinstance(exc, KeyboardInterrupt) else "error")
        raise
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Exported evolution JSON")
    parser.add_argument("--output", type=Path, required=True, help="New local JSON file; must not exist")
    parser.add_argument("--limit", type=int, help="Maximum unique revisions (default: 4 for one population, all for --history)")
    parser.add_argument("--history", action="store_true", help="Preserve and annotate all history snapshots chronologically")
    parser.add_argument("--reuse-observations", type=Path, help="Reuse fingerprint-valid raw observations from a previous history pilot")
    parser.add_argument("--generation", type=int, help="Zero-based history snapshot; defaults to current/final population")
    parser.add_argument("--provider", choices=tuple(DEFAULT_MODELS_BY_PROVIDER), help="Defaults to model provider, otherwise Gemini")
    parser.add_argument("--model", choices=[model["id"] for model in LLM_MODELS], help="Defaults to the provider's inexpensive app default")
    parser.add_argument("--thinking-level", help="Defaults to the model's lowest supported level")
    parser.add_argument("--max-budget", type=float, default=0.25, help="Stop new requests at this recorded USD estimate (default: 0.25)")
    args = parser.parse_args(argv)
    try:
        if args.input.resolve() == args.output.resolve() or args.output.exists():
            raise ValueError("--output must be a new file, different from the input")
        if not math.isfinite(args.max_budget) or args.max_budget < 0:
            raise ValueError("--max-budget must be finite and non-negative")
        export = json.loads(args.input.read_text())
        if not isinstance(export, dict):
            raise ValueError("Input must be an evolution export object")
        if args.history and args.generation is not None:
            raise ValueError("--history and --generation cannot be combined")
        if args.reuse_observations is not None and not args.history:
            raise ValueError("--reuse-observations requires --history")
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be at least 1")
        if args.history:
            candidates, generation = None, None
        else:
            candidates, generation = select_candidates(export, args.limit or 4, args.generation)
        reuse_export = json.loads(args.reuse_observations.read_text()) if args.reuse_observations else None
        if reuse_export is not None and not isinstance(reuse_export, dict):
            raise ValueError("Reuse file must be an observation export object")
        provider = args.provider or (get_model_provider(args.model) if args.model else "google_generative_ai")
        model = args.model or DEFAULT_MODELS_BY_PROVIDER[provider]
        if get_model_provider(model) != provider:
            raise ValueError("--provider and --model must identify the same provider")
        key_name = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
        api_key = os.environ.get(key_name, "").strip()
        if not api_key:
            raise ValueError(f"Set {key_name} in the environment, or use uv run --env-file .env")
        levels = THINKING_MODEL_CONFIG.get(model, {}).get("allowed_levels", [])
        thinking_level = args.thinking_level or (levels[0] if levels else None)
        if thinking_level is not None and thinking_level not in levels:
            raise ValueError("--thinking-level is unsupported by the selected model")
        idea_type = str((export.get("config") or {}).get("idea_type") or "airesearch")
        template_data = export.get("template_data")
        if not isinstance(template_data, dict):
            wrapper = get_prompts(idea_type)
            template_data = wrapper.template.model_dump(mode="json")
    except (ValueError, OSError, AttributeError) as exc:
        # Only local input validation belongs in this branch. Provider exception
        # messages can include request data, so later failures report their type.
        parser.exit(2, f"Cannot observe genes: {exc}\n")
    try:
        breeder = Breeder(provider=provider, model_name=model, api_key=api_key,
                          temperature=0.0, top_p=0.95, thinking_level=thinking_level, random_seed=0)
        breeder.register_custom_template(idea_type, template_data)
        if args.history:
            result = observe_history(export, input_path=args.input, output_path=args.output,
                                     breeder=breeder, template_data=template_data,
                                     max_budget=args.max_budget, limit=args.limit,
                                     reuse_export=reuse_export, reuse_path=args.reuse_observations)
        else:
            result = observe_export(export, input_path=args.input, output_path=args.output,
                                    candidates=candidates, original_generation=generation,
                                    breeder=breeder, template_data=template_data, max_budget=args.max_budget)
    except KeyboardInterrupt:
        parser.exit(130, "Stopped; any completed annotations remain in the output file.\n")
    except Exception as exc:
        parser.exit(1, f"Observation failed ({type(exc).__name__}); inspect the output for completed annotations.\n")
    cost = result["token_counts"]["cost"]["total_cost"]
    print(f"{result['status']}: {len(result['gene_catalogue']['genes'])} catalogue genes; "
          f"{result['token_counts']['total']} tokens; estimated cost ${cost:.6f}")
    print(f"Output: {args.output.resolve()}")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
