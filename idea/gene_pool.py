"""Generation-boundary gene consolidation, separate from immutable extraction.

Each pass starts from committed ingredients plus new observations. History and
raw provenance stay local; a validated revision supplies breeding factors.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from idea.gene_prompts import consolidation_gene_prompt
from idea.genes import empty_gene_catalogue, normalize_gene_text, parse_genotype, valid_gene_observation


class GenePoolMember(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: StrictInt
    remainder: list[str]


class GenePoolChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Standalone splits can put all factors in the remainder; no sentinel needed.
    shared: list[str]
    members: list[GenePoolMember] = Field(min_length=1)
    reason: str = Field(min_length=1)


class GenePoolResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    changes: list[GenePoolChange]


def valid_gene_projection(candidate: Any) -> bool:
    if not isinstance(candidate, dict):
        return False
    raw, projection = candidate.get("gene_observation"), candidate.get("gene_projection")
    if (not isinstance(raw, dict) or not isinstance(raw.get("fingerprint"), str)
            or not valid_gene_observation(raw, raw.get("fingerprint"))):
        return False
    if not isinstance(projection, dict) or (
        projection.get("version") != 1 or projection.get("status") != "ok"
        or projection.get("raw_fingerprint") != raw.get("fingerprint")
        or type(projection.get("revision")) is not int or projection["revision"] < 1
    ):
        return False
    try:
        terms = parse_genotype(projection.get("genotype"))
        records = projection.get("genes")
        sources = set(parse_genotype(raw["genotype"]))
        return isinstance(records, list) and len(terms) == len(records) and all(
            isinstance(g, dict) and g.get("text") == term
            and isinstance(g.get("gene_id"), str) and bool(g["gene_id"])
            and isinstance(g.get("source_terms"), list) and bool(g["source_terms"])
            and all(isinstance(t, str) and t in sources for t in g["source_terms"])
            for term, g in zip(terms, records)
        ) and set(t for g in records for t in g["source_terms"]) == sources
    except (TypeError, ValueError):
        return False


def parse_pool_rules(response: str, terms: list[str]) -> dict:
    parsed = GenePoolResponse.model_validate_json(response)
    # Omitted labels carry forward deterministically; the model only proposes edits.
    rules = {term: {'ingredients': [term], 'reason': 'Carried forward unchanged', 'unresolved': ''}
             for term in terms}
    seen = set()
    for change in parsed.changes:
        if not change.reason.strip():
            raise ValueError('Missing change reason')
        for member in change.members:
            if member.index < 0 or member.index >= len(terms) or member.index in seen:
                raise ValueError('Duplicate or unknown change index')
            seen.add(member.index)
            ingredients = [t.strip() for t in [*change.shared, *member.remainder]]
            if (not ingredients
                    or any(not t or ';' in t or ',' in t or '\n' in t for t in ingredients)
                    or any(t == 'ABSORBED' for t in ingredients)
                    or len({normalize_gene_text(t) for t in ingredients}) != len(ingredients)):
                raise ValueError('Invalid pool ingredients')
            rules[terms[member.index]] = {'ingredients': ingredients,
                'reason': change.reason.strip(), 'unresolved': ''}
    return rules


def _observed_pool(population, history):
    references, pool = [], {}
    for generation, candidates in enumerate([*(history or []), population]):
        for candidate in candidates:
            if not isinstance(candidate, dict) or not candidate.get("id"):
                continue
            raw = candidate.get("gene_observation")
            fingerprint = raw.get("fingerprint") if isinstance(raw, dict) else None
            if not isinstance(fingerprint, str) or not valid_gene_observation(raw, fingerprint):
                continue
            key = hashlib.sha256(json.dumps([str(candidate["id"]), fingerprint]).encode()).hexdigest()
            references.append((candidate, key, generation))
            pool.setdefault(key, {"candidate_id": key, "genotype": raw["genotype"]})
    return references, list(pool.values())


def _working_pool(references, population, projections=None):
    """Compact model input, with raw-source composition kept outside the prompt.

    Prefer the newest valid committed copy of a candidate revision. Historical
    copies supply vocabulary only, not repeated genotypes or past explanations.
    Uncommitted observations are admitted once, including legacy backfills.
    """
    current_objects = {id(c) for c in population}
    candidates, ranks, current = {}, {}, set()
    for candidate, key, _ in references:
        if id(candidate) in current_objects:
            current.add(key)
        rank = candidate['gene_projection']['revision'] if valid_gene_projection(candidate) else 0
        if key not in candidates or rank > ranks[key]:
            candidates[key] = candidate
            ranks[key] = rank
    pool, context = [], []
    for key, candidate in candidates.items():
        projection = (projections or {}).get(key)
        if projection is None and valid_gene_projection(candidate):
            projection = candidate['gene_projection']
        if projection is not None:
            sources = {g['text']: list(g['source_terms']) for g in projection['genes']}
            genotype = projection['genotype']
        else:
            genotype = candidate['gene_observation']['genotype']
            sources = {t: [t] for t in parse_genotype(genotype)}
        item = {'candidate_id': key, 'genotype': genotype, 'sources': sources}
        pool.append(item)
        if key in current or projection is None:
            context.append({'candidate_id': key, 'genotype': genotype,
                            'state': 'committed' if projection is not None else 'new'})
    terms = list(dict.fromkeys(t for item in pool for t in parse_genotype(item['genotype'])))
    return pool, context, terms


def consolidate_gene_pool(population, history, catalogue, manager, idea_type, generation):
    """Commit a complete mapping atomically, or retain the last good projections.

    Raw observations, scores and historical parent inputs are never rewritten.
    Historical projections are explicitly views through the latest vocabulary.
    """
    original = copy.deepcopy(catalogue) if isinstance(catalogue, dict) else empty_gene_catalogue()
    references, _ = _observed_pool(population, history)
    if not references:
        return original
    previous = original.get("consolidation", {})
    if not isinstance(previous, dict):
        previous = {}
    try:
        pool, context, terms = _working_pool(references, population)
        def signature(context, terms):
            return hashlib.sha256(json.dumps({
                "policy": consolidation_gene_prompt(context, terms), "domain": idea_type,
                "model": getattr(manager, "model_name", None),
                "provider": getattr(manager, "provider", None),
                "thinking_level": getattr(manager, "thinking_level", None),
            }, sort_keys=True).encode()).hexdigest()
        input_signature = signature(context, terms)
        # Compare against the committed output, not the pre-commit raw input.
        # Every reference must already have that projection before skipping work.
        if (previous.get('status') == 'ok' and previous.get('committed_signature') == input_signature
                and all(valid_gene_projection(c) and c['gene_projection']['revision'] == previous['revision']
                        for c, _, _ in references)):
            return original
        cancel = getattr(manager, "cancel_check", None)
        event = getattr(manager, "cancel_event", None)
        if (callable(cancel) and cancel()) or (event is not None and event.is_set()):
            raise RuntimeError("Consolidation cancelled")
        response = manager.generate_text(consolidation_gene_prompt(context, terms),
                                         response_schema=GenePoolResponse)
        rules = parse_pool_rules(response, terms)
        revision = int(previous.get("revision", 0)) + 1

        result = copy.deepcopy(original)
        entries = result.setdefault("genes", [])
        by_label = {normalize_gene_text(g["label"]): g for g in entries}
        by_id = {g["id"]: g for g in entries}
        if len(by_id) != len(entries) or len(by_label) != len(entries):
            raise ValueError("Invalid catalogue identities")
        projections = {}
        first_seen = {}
        for _, key, observed_generation in references:
            first_seen[key] = min(first_seen.get(key, generation), observed_generation, generation)
        for item in pool:
            records = {}
            for term in parse_genotype(item["genotype"]):
                rule = rules[term]
                for label in rule["ingredients"]:
                    normalized = normalize_gene_text(label)
                    gene = by_label.get(normalized)
                    if gene is None:
                        identity = "gene_" + hashlib.sha256(normalized.encode()).hexdigest()
                        if identity in by_id:
                            raise ValueError("Gene identity collision")
                        gene = {"id": identity, "label": label, "aliases": [],
                                "first_seen_generation": first_seen[item["candidate_id"]],
                                "last_seen_generation": generation,
                                "current_candidate_count": 0, "current_candidate_ids": []}
                        entries.append(gene)
                        by_label[normalized] = gene
                        by_id[identity] = gene
                    gene["first_seen_generation"] = min(gene.get("first_seen_generation", generation),
                                                         first_seen[item["candidate_id"]])
                    record = records.setdefault(gene["id"], {"text": gene["label"], "gene_id": gene["id"],
                        "source_terms": [], "decisions": []})
                    for raw_term in item['sources'][term]:
                        if raw_term not in record['source_terms']:
                            record["source_terms"].append(raw_term)
                        record["decisions"].append({"source": raw_term, "input": term,
                            "reason": rule["reason"], "unresolved": rule["unresolved"]})
            projections[item["candidate_id"]] = {"version": 1, "status": "ok", "revision": revision,
                "genotype": "; ".join(r["text"] for r in records.values()), "genes": list(records.values())}

        staged = []
        for candidate, key, _ in references:
            projection = {**copy.deepcopy(projections[key]),
                          "raw_fingerprint": candidate["gene_observation"]["fingerprint"]}
            if not valid_gene_projection({**candidate, "gene_projection": projection}):
                raise ValueError("Invalid projected genotype")
            staged.append((candidate, projection))
        current_objects = {id(c) for c in population}
        carriers = {g["id"]: set() for g in entries}
        for candidate, projection in staged:
            if id(candidate) in current_objects:
                for gene in projection["genes"]:
                    carriers[gene["gene_id"]].add(str(candidate["id"]))
        for gene in entries:
            gene["current_candidate_ids"] = sorted(carriers[gene["id"]])
            gene["current_candidate_count"] = len(carriers[gene["id"]])
            if carriers[gene["id"]]:
                gene["last_seen_generation"] = generation
        _, committed_context, committed_terms = _working_pool(references, population, projections)
        metadata = {"version": 2, "status": "ok", "revision": revision,
                    "generation": generation, "pool_signature": input_signature, "rules": rules,
                    "committed_signature": signature(committed_context, committed_terms),
                    "input_representation": "committed_plus_new",
                    "prompt_candidates": len(context), "input_terms": len(terms),
                    "model": getattr(manager, "model_name", None),
                    "provider": getattr(manager, "provider", None), "candidate_revisions": len(pool)}
        result["revision"] = int(result.get("revision", 0)) + 1
        result.setdefault("consolidation_revisions", []).append(copy.deepcopy(metadata))
        result.update(consolidation=metadata, mapping_status="ok", last_reconciled_generation=generation)
        result.pop("mapping_error", None)
        # No candidate has been mutated before all mappings/projections validate.
        for candidate, projection in staged:
            candidate["gene_projection"] = projection
        return result
    except Exception:
        original["consolidation"] = {**previous, "status": "failed",
            "error": "Gene consolidation failed; saved factors and raw observations are retained."}
        original["mapping_status"] = "failed"
        original["mapping_error"] = original["consolidation"]["error"]
        return original
