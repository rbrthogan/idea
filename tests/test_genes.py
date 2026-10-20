import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from idea.genes import (
    empty_gene_catalogue,
    normalize_genotype_response,
    parse_gene_matches,
    parse_genotype,
    reconcile_gene_catalogue,
)
from idea.llm import Breeder, Oracle
from idea.models import CandidateRecord, Idea, candidate_from_wire, candidate_to_wire


def breeder_with_output(output="hidden bids; shared clock"):
    with patch("idea.llm.LLMWrapper._setup_provider", return_value=None):
        breeder = Breeder()
    breeder._resolve_prompts = Mock(return_value=SimpleNamespace(GENOTYPE_ENCODE_PROMPT="Extract freely: {idea_content}"))
    breeder.generate_text = Mock(return_value=output)
    return breeder


def observed(identity, genes):
    genotype = "; ".join(genes)
    return {"id": identity, "idea": "Design", "gene_observation": {
        "version": 1, "status": "ok", "fingerprint": "content-identity", "genotype": genotype,
        "genes": [{"text": text, "gene_id": None} for text in parse_genotype(genotype)],
        "mapping_status": "pending",
    }}


def matching(*groups):
    return json.dumps({"groups": [{"existing_gene_id": identity, "input_indices": indices}
                                  for identity, indices in groups]})


def test_parse_keeps_raw_spelling_and_only_normalizes_presentation_duplicates():
    assert parse_genotype(" Hidden   bids; hidden bids; SECRET bids; delayed feedback ") == [
        "Hidden   bids", "SECRET bids", "delayed feedback"
    ]
    with pytest.raises(ValueError):
        parse_genotype(" ; ; ")


def test_observation_roundtrip_reuses_exact_genotype_in_breeding_without_catalogue_prompt():
    breeder = breeder_with_output("Genotype: Hidden   bids; shared clock")
    candidate = {"id": "one", "idea": "Complete design"}
    observation = breeder.observe_genes(candidate, "game")
    assert observation["status"] == "ok"
    assert observation["genotype"] == "Hidden   bids; shared clock"
    assert observation["genes"] == [{"text": "Hidden   bids", "gene_id": None},
                                    {"text": "shared clock", "gene_id": None}]
    assert breeder.generate_text.call_count == 1
    prompt = breeder.generate_text.call_args.args[0]
    assert prompt.startswith("Extract freely: Complete design\n")
    assert "Shared gene representation:" in prompt

    restored = candidate_from_wire(candidate_to_wire(candidate))
    restored["gene_observation"]["genes"][0]["gene_id"] = "canonical-is-annotation-only"
    restored["gene_observation"]["mapping_status"] = "failed"
    fresh_breeder = breeder_with_output("must not be generated")
    assert fresh_breeder.encode_to_genotype(restored, "game") == "Hidden   bids; shared clock"
    assert fresh_breeder.observe_genes(restored, "game")["status"] == "ok"
    fresh_breeder.generate_text.assert_not_called()


@pytest.mark.parametrize("response", [
    "Title: A design\nContent: Full prose; with semicolons.\n\nGenotype: hidden bids; shared clock",
    "Phenotype: Full prose; with semicolons.\n\n**Genotype:** hidden bids; shared clock",
    "Full prose.\n### **Genotype**: hidden bids; shared clock",
    "Genotype: earlier draft\nFull prose.\n__Genotype:__ hidden bids; shared clock",
    "```text\nFull prose.\nGenotype: hidden bids; shared clock\n```",
    "Full prose.\nGenotype:\n```text\nhidden bids; shared clock\n```",
])
def test_only_explicit_final_genotype_section_is_observed_and_bred(response):
    breeder = breeder_with_output(response)
    candidate = {"id": "one", "idea": "Complete design"}
    observation = breeder.observe_genes(candidate, "game")
    assert observation["status"] == "ok"
    assert observation["genotype"] == "hidden bids; shared clock"
    assert [gene["text"] for gene in observation["genes"]] == ["hidden bids", "shared clock"]
    assert breeder.encode_to_genotype(candidate, "game") == "hidden bids; shared clock"
    assert breeder.generate_text.call_count == 1


def test_direct_semicolon_response_is_not_rewritten_or_split_on_embedded_label():
    text = "Hidden   bids;\n shared clock; a display labelled Genotype: to identify the player"
    assert normalize_genotype_response(text) == text


@pytest.mark.parametrize("response", [
    "Phenotype: Full design; many rules", "Title: Game\nContent: Full design; many rules",
    "Full prose\nGenotype: ", "Genotype: ; ;",
])
def test_unseparated_source_echo_or_empty_genotype_is_not_observed(response):
    breeder = breeder_with_output(response)
    candidate = {"id": "one", "idea": "Complete design"}
    assert breeder.observe_genes(candidate, "game")["status"] == "failed"
    assert not breeder._genotype_cache
    assert candidate["idea"] == "Complete design"


def test_legacy_observation_containing_source_echo_is_not_reused():
    breeder = breeder_with_output("hidden bids; shared clock")
    candidate = {"id": "one", "idea": "Complete design"}
    _, fingerprint = breeder._genotype_prompt_and_fingerprint(candidate, "game")
    echoed = "Title: Game\nContent: Full design; many rules\n\nGenotype: hidden bids; shared clock"
    candidate["gene_observation"] = {
        "version": 1, "status": "ok", "fingerprint": fingerprint, "genotype": echoed,
        "genes": [{"text": text, "gene_id": "obsolete"} for text in parse_genotype(echoed)],
    }
    assert breeder.observe_genes(candidate, "game")["genotype"] == "hidden bids; shared clock"
    assert breeder.generate_text.call_count == 1
    assert all(gene["gene_id"] is None for gene in candidate["gene_observation"]["genes"])


def test_fingerprint_uses_only_phenotype_across_wire_formatted_and_flat_shapes():
    breeder = breeder_with_output()
    formatted = {"id": "a", "idea": Idea(title="Title", content="Body")}
    wire = {"id": "b", "idea": {"title": "Title", "content": "Body"}, "fitness": 0.7}
    flat = {"id": "c", "title": "Title", "content": "Body", "ratings": {"auto": 1500}}
    fingerprints = [breeder.observe_genes(candidate, "game")["fingerprint"]
                    for candidate in (formatted, wire, flat)]
    record = CandidateRecord.from_candidate(formatted)
    fingerprints.append(breeder.observe_genes(record, "game")["fingerprint"])
    fingerprints.append(breeder.observe_genes({"idea": {"idea": wire["idea"]}}, "game")["fingerprint"])
    assert len(set(fingerprints)) == 1
    before = flat["gene_observation"]["fingerprint"]
    flat["gene_observation"]["mapping_status"] = "failed"
    assert breeder.observe_genes(flat, "game")["fingerprint"] == before
    assert breeder.generate_text.call_count == 1


@pytest.mark.parametrize("change", ["content", "prompt", "model", "temperature"])
def test_observation_invalidates_when_content_or_extraction_configuration_changes(change):
    breeder = breeder_with_output()
    candidate = {"id": "one", "idea": "Original"}
    old = copy.deepcopy(breeder.observe_genes(candidate, "game"))
    if change == "content":
        candidate["idea"] = "Revised"
    elif change == "prompt":
        breeder._resolve_prompts.return_value.GENOTYPE_ENCODE_PROMPT = "New extraction: {idea_content}"
    elif change == "model":
        breeder.model_name = "another-model"
    else:
        breeder.temperature = 0.123
    breeder.generate_text.return_value = "new mechanism"
    new = breeder.observe_genes(candidate, "game")
    assert new["fingerprint"] != old["fingerprint"]
    assert new["genes"] == [{"text": "new mechanism", "gene_id": None}]
    assert old["genotype"] == "hidden bids; shared clock"
    assert breeder.generate_text.call_count == 2


def test_extraction_failure_is_sanitized_preserves_candidate_and_can_retry():
    breeder = breeder_with_output()
    breeder.generate_text.side_effect = RuntimeError("secret-api-key and raw provider payload")
    candidate = {"id": "one", "idea": "Untouched", "fitness": 0.7}
    observation = breeder.observe_genes(candidate, "game")
    assert observation["status"] == "failed"
    assert observation["genes"] == []
    assert "secret" not in json.dumps(observation)
    assert candidate["idea"] == "Untouched" and candidate["fitness"] == 0.7
    breeder.generate_text.side_effect = None
    assert breeder.observe_genes(candidate, "game")["status"] == "ok"


def test_new_equivalents_share_canonical_gene_and_counts_are_distinct_candidates():
    candidates = [observed("a", ["hidden bids", "secret bids", "shared clock"]),
                  observed("b", ["secret bids"]), observed("a", ["hidden bids"])]
    old_observation = candidates[0]["gene_observation"]
    matcher = SimpleNamespace(generate_text=Mock(return_value=matching((None, [0, 1]), (None, [2]))))
    catalogue = reconcile_gene_catalogue(candidates, empty_gene_catalogue(), matcher, "game", 0)
    assert catalogue["revision"] == 1
    assert len(catalogue["genes"]) == 2
    hidden, clock = catalogue["genes"]
    assert hidden["label"] == "hidden bids" and hidden["aliases"] == ["secret bids"]
    assert hidden["current_candidate_count"] == 2
    assert hidden["current_candidate_ids"] == ["a", "b"]
    assert clock["current_candidate_count"] == 1
    assert candidates[0]["gene_observation"]["genes"][0]["gene_id"] == hidden["id"]
    assert candidates[0]["gene_observation"]["genes"][1]["gene_id"] == hidden["id"]
    assert old_observation["genes"][0]["gene_id"] is None  # shared historical metadata is untouched
    assert candidates[0]["gene_observation"]["genotype"] == "hidden bids; secret bids; shared clock"


def test_existing_alias_matches_and_exact_normalization_needs_no_llm_call():
    seed = [observed("a", ["hidden bids"])]
    matcher = SimpleNamespace(generate_text=Mock())
    initial = reconcile_gene_catalogue(seed, empty_gene_catalogue(), matcher, "game", 0)
    identity = initial["genes"][0]["id"]
    matcher.generate_text.return_value = matching((identity, [0]))
    candidate = observed("b", ["secret bids"])
    second = reconcile_gene_catalogue([candidate], initial, matcher, "game", 1)
    assert second["genes"][0]["id"] == identity
    assert second["genes"][0]["aliases"] == ["secret bids"]
    assert second["genes"][0]["first_seen_generation"] == 0
    assert second["genes"][0]["last_seen_generation"] == 1
    matcher.generate_text.reset_mock()
    third = reconcile_gene_catalogue([observed("c", [" SECRET  bids "])], second, matcher, "game", 2)
    matcher.generate_text.assert_not_called()
    assert third["revision"] == second["revision"]
    assert third["genes"][0]["current_candidate_ids"] == ["c"]
    assert initial["genes"][0]["aliases"] == []


@pytest.mark.parametrize("response", [
    matching((None, [0])),  # omitted term
    matching((None, [0, 0]), (None, [1])),  # duplicate
    matching((None, [0, 2])),  # out of range
    matching(("unknown", [0, 1])),
    matching((None, []), (None, [0, 1])),
    '{"groups":[{"existing_gene_id":null,"input_indices":[true,1]}]}',
    '{"groups":[{"existing_gene_id":null,"input_indices":[0,1],"invented":1}]}',
    "not json",
])
def test_invalid_matching_partition_is_rejected_without_partial_additions(response):
    with pytest.raises((ValueError, TypeError)):
        parse_gene_matches(response, 2, set())
    candidate = observed("a", ["one mechanism", "another mechanism"])
    original = empty_gene_catalogue()
    matcher = SimpleNamespace(generate_text=Mock(return_value=response))
    result = reconcile_gene_catalogue([candidate], original, matcher, "game", 1)
    assert result["genes"] == [] and result["revision"] == 0
    assert original == empty_gene_catalogue()
    observation = candidate["gene_observation"]
    assert observation["status"] == "ok" and observation["mapping_status"] == "failed"
    assert all(gene["gene_id"] is None for gene in observation["genes"])


def test_mapping_failure_retries_without_reextracting_or_losing_raw_genes():
    candidate = observed("a", ["hidden bids", "shared clock"])
    matcher = SimpleNamespace(generate_text=Mock(side_effect=RuntimeError("secret credential")))
    failed = reconcile_gene_catalogue([candidate], empty_gene_catalogue(), matcher, "game", 0)
    assert "secret" not in json.dumps(failed)
    matcher.generate_text.side_effect = None
    matcher.generate_text.return_value = matching((None, [0]), (None, [1]))
    successful = reconcile_gene_catalogue([candidate], failed, matcher, "game", 1)
    assert len(successful["genes"]) == 2
    assert candidate["gene_observation"]["mapping_status"] == "ok"
    assert candidate["gene_observation"]["genotype"] == "hidden bids; shared clock"
    assert "mapping_error" not in candidate["gene_observation"]


def test_catalogue_has_no_population_scaled_cap_and_retains_absent_genes():
    candidate = observed("a", [f"mechanism {i}" for i in range(8)])
    matcher = SimpleNamespace(generate_text=Mock(return_value=matching(*[(None, [i]) for i in range(8)])))
    catalogue = reconcile_gene_catalogue([candidate], empty_gene_catalogue(), matcher, "game", 0)
    assert len(catalogue["genes"]) == 8  # deliberately exceeds three times the population
    absent = reconcile_gene_catalogue([], catalogue, matcher, "game", 1)
    assert len(absent["genes"]) == 8
    assert all(gene["current_candidate_count"] == 0 for gene in absent["genes"])
    assert all(gene["last_seen_generation"] == 0 for gene in absent["genes"])


def test_invalid_catalogue_does_not_destroy_extraction_or_raise():
    candidate = observed("a", ["hidden bids"])
    invalid = {"version": 999, "revision": 0, "genes": []}
    result = reconcile_gene_catalogue([candidate], invalid, SimpleNamespace(), "game", 0)
    assert result["mapping_status"] == "failed"
    assert candidate["gene_observation"]["status"] == "ok"
    assert candidate["gene_observation"]["genes"] == [{"text": "hidden bids", "gene_id": None}]


@pytest.mark.parametrize("gene_count", [1, 2])
def test_cancelled_annotation_keeps_unknown_genes_and_makes_no_matching_call(gene_count):
    candidate = observed("a", [f"mechanism {i}" for i in range(gene_count)])
    matcher = SimpleNamespace(cancel_check=lambda: True, generate_text=Mock())
    result = reconcile_gene_catalogue([candidate], empty_gene_catalogue(), matcher, "game", 0)
    matcher.generate_text.assert_not_called()
    assert result["genes"] == []
    assert result["mapping_status"] == "failed"
    assert candidate["gene_observation"]["status"] == "ok"
    assert candidate["gene_observation"]["mapping_status"] == "failed"


def test_observation_metadata_never_enters_oracle_prompt_or_history_identity():
    with patch("idea.llm.LLMWrapper._setup_provider", return_value=None):
        oracle = Oracle()
    oracle._resolve_prompts = Mock(return_value=SimpleNamespace(
        ORACLE_MAIN_PROMPT="Read designs: {history_text}", ORACLE_INSTRUCTION="",
        ORACLE_FORMAT_INSTRUCTIONS="",
    ))
    flat = {"id": "a", "title": "Clock", "content": "Players negotiate every tick."}
    annotated = {**flat, "gene_observation": {
        "genes": [{"text": "CANONICAL_GENE_MUST_NOT_LEAK", "gene_id": "private-catalogue-id"}],
        "mapping_status": "ok",
    }}
    plain_prompt = oracle._build_analysis_prompt([[flat]], "game")
    assert oracle._build_analysis_prompt([[annotated]], "game") == plain_prompt
    duplicate_prompt = oracle._build_analysis_prompt([[flat, annotated]], "game")
    assert duplicate_prompt.count("Players negotiate every tick.") == 1
    assert "CANONICAL_GENE_MUST_NOT_LEAK" not in duplicate_prompt
    assert "private-catalogue-id" not in duplicate_prompt
    assert oracle._extract_idea_content(annotated) == oracle._extract_idea_content(
        {"idea": Idea(title=flat["title"], content=flat["content"])}
    )
