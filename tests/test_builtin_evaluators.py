import asyncio

import pytest

from idea.evaluation.builtins import (
    CHARACTER_COUNT_RUNNER_REF,
    CharacterCountRunner,
    create_builtin_computed_runner,
    is_builtin_computed_runner,
)
from idea.evaluation.components import ComputedEvaluator
from idea.evaluation.models import ComputedComponentConfig, EvaluationStatus
from idea.viewer import _web_evaluation_component_available
from idea.evolution import EvolutionEngine


def _config(**overrides):
    payload = {
        "id": "length",
        "kind": "computed",
        "runner_ref": CHARACTER_COUNT_RUNNER_REF,
        "metric_names": ["character_count"],
    }
    payload.update(overrides)
    return ComputedComponentConfig.model_validate(payload)


def test_character_count_runner_counts_configured_text_without_execution():
    runner = CharacterCountRunner()
    result = asyncio.run(
        runner.run(
            {"idea": {"title": "Prime", "content": "2, 3, 5"}},
            candidate_id="candidate-1",
            generation=2,
            config=_config(),
        )
    )

    assert result.status == "ok"
    assert result.metrics == {"character_count": 7.0}
    assert result.artifacts == []


def test_character_count_runner_can_exclude_whitespace_and_include_title():
    result = asyncio.run(
        CharacterCountRunner().run(
            {"idea": {"title": "A B", "content": " C\nD "}},
            candidate_id="candidate-1",
            generation=None,
            config=_config(
                parameters={
                    "field": "title_and_content",
                    "include_whitespace": False,
                }
            ),
        )
    )

    assert result.metrics == {"character_count": 4.0}


def test_invalid_builtin_configuration_becomes_explicit_failed_evidence():
    config = _config(parameters={"field": "everything"})
    population = [
        {"id": "candidate-1", "idea": {"title": "A", "content": "B"}}
    ]

    result = asyncio.run(
        ComputedEvaluator(config).evaluate_population(
            population,
            CharacterCountRunner(),
            evaluation_generation=1,
        )
    )

    assert result.scores == {}
    assert result.batch.records[0].status == EvaluationStatus.FAILED
    assert "field must be" in result.batch.records[0].error
    assert population[0]["evaluations"]["length"]["status"] == "failed"


def test_builtin_runner_registry_is_an_explicit_allow_list():
    assert is_builtin_computed_runner(CHARACTER_COUNT_RUNNER_REF)
    assert isinstance(
        create_builtin_computed_runner(CHARACTER_COUNT_RUNNER_REF),
        CharacterCountRunner,
    )
    assert not is_builtin_computed_runner("package.module:callable")
    assert create_builtin_computed_runner("package.module:callable") is None


def test_web_allows_only_reviewed_builtin_computed_runners():
    assert _web_evaluation_component_available(_config())
    assert not _web_evaluation_component_available(
        _config(runner_ref="external.simulation")
    )


def test_engine_resolves_allow_listed_builtin_without_injected_runner():
    engine = EvolutionEngine(
        pop_size=2,
        generations=1,
        evaluation_config={
            "components": [
                {
                    "id": "length",
                    "kind": "computed",
                    "runner_ref": CHARACTER_COUNT_RUNNER_REF,
                    "metric_names": ["character_count"],
                }
            ],
            "fitness": {
                "metrics": [
                    {
                        "component_id": "length",
                        "metric_name": "character_count",
                        "normalization": "bounds",
                        "bounds": [0, 10],
                    }
                ]
            },
        },
    )
    population = [
        {"id": "short", "idea": {"content": "12"}},
        {"id": "long", "idea": {"content": "12345678"}},
    ]

    fitness = asyncio.run(
        engine._score_population_fitness(
            population,
            {0: 1500.0, 1: 1500.0},
        )
    )

    assert engine.unsupported_evaluation_components == []
    assert fitness[0]["fitness"] == pytest.approx(0.2)
    assert fitness[1]["fitness"] == pytest.approx(0.8)
