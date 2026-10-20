"""Offline checks of the production extraction → consolidation → breeding path."""
import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from idea.gene_pool import consolidate_gene_pool, valid_gene_projection, parse_pool_rules
from idea.genes import empty_gene_catalogue
from idea.models import candidate_from_wire, candidate_to_wire
from idea.evolution_orchestrator import EvolutionOrchestrator
from tests.test_genes import breeder_with_output, observed
from tests.test_gene_lifecycle import engine_for


def manager_for(*factors):
    return SimpleNamespace(model_name="test", provider="test", thinking_level="medium",
        generate_text=Mock(return_value=json.dumps({"changes": [
            {"shared": ingredients, "members": [{"index": i, "remainder": []}], "reason": "Retains each independent trait"}
            for i, ingredients in enumerate(factors)]})))


def test_grouped_changes_extract_shared_core_and_retain_each_remainder():
    terms = ['comic desert heist', 'tragic desert pilgrimage', 'body horror']
    response = {'changes': [{'shared': ['desert setting'], 'members': [
        {'index': 0, 'remainder': ['comic tone', 'heist']},
        {'index': 1, 'remainder': ['tragic tone', 'pilgrimage']}],
        'reason': 'Common setting with different moods and activities'}]}
    rules = parse_pool_rules(json.dumps(response), terms)
    assert rules[terms[0]]['ingredients'] == ['desert setting', 'comic tone', 'heist']
    assert rules[terms[1]]['ingredients'] == ['desert setting', 'tragic tone', 'pilgrimage']
    assert rules[terms[2]]['ingredients'] == ['body horror']


def test_standalone_split_needs_no_placeholder_and_retains_both_factors():
    response = {'changes': [{'shared': [], 'members': [
        {'index': 0, 'remainder': ['absurdism', 'satire']}], 'reason': 'Separate reusable ingredients'}]}
    rules = parse_pool_rules(json.dumps(response), ['absurdist satire', 'horror'])
    assert rules['absurdist satire']['ingredients'] == ['absurdism', 'satire']
    assert rules['horror']['ingredients'] == ['horror']


@pytest.mark.parametrize('bad', ['duplicate_member', 'duplicate_group', 'unknown', 'negative',
                                'empty', 'duplicate_factor', 'separator', 'boolean', 'placeholder'])
def test_invalid_grouped_changes_are_rejected(bad):
    group = {'shared': ['setting'], 'members': [{'index': 0, 'remainder': ['mood']}], 'reason': 'Shared core'}
    response = {'changes': [group]}
    if bad == 'duplicate_member': group['members'] *= 2
    if bad == 'duplicate_group': response['changes'] *= 2
    if bad == 'unknown': group['members'][0]['index'] = 1
    if bad == 'negative': group['members'][0]['index'] = -1
    if bad == 'boolean': group['members'][0]['index'] = True
    if bad == 'empty':
        group['shared'] = []
        group['members'][0]['remainder'] = []
    if bad == 'placeholder': group['shared'] = ['ABSORBED']
    if bad == 'duplicate_factor': group['members'][0]['remainder'] = ['setting']
    if bad == 'separator': group['shared'] = ['setting; mood']
    with pytest.raises(ValueError):
        parse_pool_rules(json.dumps(response), ['source'])


def test_global_pool_factors_history_and_all_islands_without_changing_raw_or_scores():
    a = {**observed("a", ["space western"]), "island_id": 0, "fitness": .7}
    b = {**observed("b", ["dinosaur western"]), "island_id": 1, "fitness": .3}
    history = [[copy.deepcopy(a)], [copy.deepcopy(a), copy.deepcopy(b)]]
    before = copy.deepcopy([a, b])
    manager = manager_for(["space setting", "western"], ["dinosaurs", "western"])
    catalogue = consolidate_gene_pool([a, b], history, {}, manager, "drabble", 1)
    assert catalogue["consolidation"]["candidate_revisions"] == 2
    assert catalogue["consolidation"]["revision"] == 1
    western = next(g for g in catalogue["genes"] if g["label"] == "western")
    assert western["current_candidate_ids"] == ["a", "b"]
    for candidate, original in zip([a, b], before):
        assert valid_gene_projection(candidate)
        assert {k: v for k, v in candidate.items() if k != "gene_projection"} == original
        assert candidate_from_wire(candidate_to_wire(candidate))["gene_projection"] == candidate["gene_projection"]
    assert history[0][0]["gene_projection"] == a["gene_projection"]
    assert a["gene_projection"]["genes"][1]["source_terms"] == ["space western"]
    assert catalogue["genes"][0]["aliases"] == []  # A source bundle is not a synonym.

    # Resume retains projections and skips a paid pass on an unchanged pool.
    restored = json.loads(json.dumps(catalogue))
    again = consolidate_gene_pool([a, b], history, restored, manager, "drabble", 1)
    assert again == catalogue
    manager.generate_text.assert_called_once()


def test_revised_vocabulary_reprojects_history_but_retains_actual_breeding_inputs():
    a = observed("a", ["space western"])
    history = [[copy.deepcopy(a)]]
    manager = manager_for(["space setting", "western"])
    cat = consolidate_gene_pool([a], history, {}, manager, "drabble", 0)
    a["breeding_genotypes"] = [{"genotype": "space setting; western", "gene_pool_revision": 1}]
    b = observed("b", ["lunar setting"])
    manager.generate_text = manager_for(["outer space"], ["western"], ["moon setting"]).generate_text
    cat = consolidate_gene_pool([a, b], history, cat, manager, "drabble", 1)
    assert cat["consolidation"]["revision"] == 2
    assert len(cat["consolidation_revisions"]) == 2
    assert history[0][0]["gene_projection"]["genotype"] == "outer space; western"
    assert a["breeding_genotypes"] == [{"genotype": "space setting; western", "gene_pool_revision": 1}]
    assert a["gene_observation"]["genotype"] == "space western"
    prompt = manager.generate_text.call_args.args[0]
    assert 'space western' not in prompt
    assert 'Previous mapping:' not in prompt
    assert 'Retains each independent trait' not in prompt
    assert 'space setting; western' in prompt
    assert 'lunar setting' in prompt
    # Composition retains original source coverage after splitting and renaming.
    assert valid_gene_projection(a)
    assert all(g['source_terms'] == ['space western'] for g in a['gene_projection']['genes'])
    manager.generate_text.reset_mock()
    assert consolidate_gene_pool([a, b], history, cat, manager, 'drabble', 1) == cat
    manager.generate_text.assert_not_called()


@pytest.mark.parametrize("failure", ["missing", "duplicate", "unknown", "empty", "placeholder", "provider", "cancel"])
def test_failed_pass_is_atomic_and_preserves_previous_revision(failure):
    a = observed("a", ["space western"])
    manager = manager_for(["space setting", "western"])
    cat = consolidate_gene_pool([a], [], {}, manager, "drabble", 0)
    b = observed("b", ["dinosaur western"])
    before = copy.deepcopy([a, b])
    manager.generate_text = manager_for(["space setting"], ["western"], ["dinosaurs", "western"]).generate_text
    response = json.loads(manager.generate_text.return_value)
    if failure == "missing": del response["changes"][0]["members"][0]["index"]
    if failure == "duplicate": response["changes"][1]["members"][0]["index"] = 0
    if failure == "unknown": response["changes"][1]["members"][0]["index"] = 10
    if failure == "empty": response["changes"][1]["shared"] = []
    if failure == "placeholder": response["changes"][1]["shared"] = ['ABSORBED']
    manager.generate_text.return_value = json.dumps(response)
    if failure == "provider": manager.generate_text.side_effect = RuntimeError("secret provider payload")
    if failure == "cancel": manager.cancel_check = lambda: True
    result = consolidate_gene_pool([a, b], [], cat, manager, "drabble", 1)
    assert result["mapping_status"] == "failed"
    assert result["consolidation"]["revision"] == 1
    assert [a, b] == before and cat["mapping_status"] == "ok"
    assert "secret" not in json.dumps(result)
    if failure == "cancel": manager.generate_text.assert_not_called()


def test_prompt_uses_committed_labels_not_historical_genotypes_or_decisions():
    a = observed('old', ['space western'])
    b = observed('survivor', ['dinosaur western'])
    history = [[copy.deepcopy(a), copy.deepcopy(b)]]
    cat = consolidate_gene_pool([a, b], history, {},
        manager_for(['space setting', 'western'], ['dinosaurs', 'western']), 'drabble', 0)
    retired = observed('unused', ['discarded bundle'])
    # Unused catalogue entries and archived explanations must not enter the prompt.
    cat['genes'].append({'id': 'retired', 'label': 'discarded bundle', 'aliases': []})
    cat['consolidation']['rules']['space western']['reason'] = 'OLD-JUSTIFICATION-MARKER'
    child = observed('new', ['lunar western'])
    manager = manager_for(['outer space'], ['western'], ['dinosaurs'], ['moon setting', 'western'])
    cat = consolidate_gene_pool([b, child], history, cat, manager, 'drabble', 1)
    assert cat['mapping_status'] == 'ok'
    prompt = manager.generate_text.call_args.args[0]
    assert 'space setting' in prompt  # Historical vocabulary remains revisable.
    assert 'space setting; western' not in prompt  # But not the old candidate's genotype.
    assert 'dinosaurs; western' in prompt  # Current survivor uses committed factors.
    assert 'lunar western' in prompt  # New raw evidence is integrated once.
    for crud in ['space western', 'dinosaur western', 'discarded bundle', 'OLD-JUSTIFICATION-MARKER']:
        assert crud not in prompt
    assert cat['consolidation']['prompt_candidates'] == 2
    assert history[0][0]['gene_projection']['genotype'] == 'outer space; western'
    assert all(valid_gene_projection(c) for c in [*history[0], b, child])


def test_merge_composes_sources_without_duplicate_source_terms():
    a = observed('a', ['space western', 'western genre'])
    cat = consolidate_gene_pool([a], [], {}, manager_for(['space setting', 'western'], ['western']), 'drabble', 0)
    b = observed('b', ['new setting'])
    cat = consolidate_gene_pool([a, b], [], cat,
        manager_for(['combined mechanism'], ['combined mechanism'], ['new setting']), 'drabble', 1)
    assert cat['mapping_status'] == 'ok'
    assert a['gene_projection']['genotype'] == 'combined mechanism'
    assert a['gene_projection']['genes'][0]['source_terms'] == ['space western', 'western genre']
    assert valid_gene_projection(a)


def test_changed_raw_fingerprint_uses_new_evidence_instead_of_old_projection():
    a = observed('a', ['space western'])
    cat = consolidate_gene_pool([a], [], {}, manager_for(['space setting', 'western']), 'drabble', 0)
    old_projection = copy.deepcopy(a['gene_projection'])
    a = observed('a', ['ocean mystery'])
    a['gene_projection'] = old_projection
    manager = manager_for(['ocean setting', 'mystery'])
    cat = consolidate_gene_pool([a], [], cat, manager, 'drabble', 1)
    assert cat['mapping_status'] == 'ok'
    prompt = manager.generate_text.call_args.args[0]
    assert 'ocean mystery' in prompt and 'space setting' not in prompt
    assert valid_gene_projection(a)


def test_omitted_change_preserves_original_phrase_and_stale_projection_is_invalid():
    a = observed("a", ["weapon-metabolizing flora"])
    manager = manager_for(["metal-eating plants"])
    manager.generate_text.return_value = json.dumps({"changes": []})
    consolidate_gene_pool([a], [], {}, manager, "drabble", 0)
    assert a["gene_projection"]["genotype"] == "weapon-metabolizing flora"
    assert valid_gene_projection(a)
    a["gene_observation"]["fingerprint"] = "changed"
    assert not valid_gene_projection(a)


def test_breeding_reuses_saved_factors_while_extraction_stays_raw():
    encoder = breeder_with_output("space western")
    parent = {"id": "a", "idea": "A complete story"}
    encoder.observe_genes(parent, "drabble")
    consolidate_gene_pool([parent], [], {}, manager_for(["space setting", "western"]), "drabble", 0)
    breeder = breeder_with_output()
    breeder.gene_encoder = encoder
    helper = SimpleNamespace(generate_context_from_parents=Mock(return_value="context"),
                             generate_idea_from_context=Mock(return_value=("child", "direction")))
    breeder._get_thread_ideator = lambda: helper
    result = breeder.breed([parent], "drabble")
    assert result["parent_genotypes"] == [{"parent_id": "a", "genotype": "space setting; western",
        "raw_genotype": "space western", "source": "consolidated", "gene_pool_revision": 1}]
    assert helper.generate_context_from_parents.call_args.args[0] == ["space setting; western"]
    assert encoder.encode_to_genotype(parent, "drabble") == "space western"
    encoder.generate_text.assert_called_once()
    # A changed phenotype cannot use the old projection, even if raw words coincide.
    parent["idea"] = "Changed story"
    assert breeder.breed([parent], "drabble")["parent_genotypes"][0]["source"] == "raw"


def test_orchestrator_observes_then_consolidates_and_respects_budget():
    candidates = [{"id": "a", "idea": "Story A", "island_id": 0},
                  {"id": "b", "idea": "Story B", "island_id": 1}]
    encoder = breeder_with_output("space western")
    engine = engine_for(candidates, breeder_with_output())
    engine.gene_manager = encoder
    encoder.generate_text = Mock(side_effect=["space western", "space western",
        manager_for(["space setting", "western"]).generate_text.return_value])
    orchestrator = EvolutionOrchestrator(engine, AsyncMock())
    asyncio.run(orchestrator._observe_population_genes(0))
    assert all(valid_gene_projection(c) for c in candidates)
    assert encoder.generate_text.call_count == 3
    assert engine.gene_catalogue["genes"][0]["current_candidate_count"] == 2
    assert engine.breeder.generate_text.call_count == 0
    engine.check_budget = lambda: True
    asyncio.run(orchestrator._observe_population_genes(0))
    assert encoder.generate_text.call_count == 3
    assert engine.stop_requested
