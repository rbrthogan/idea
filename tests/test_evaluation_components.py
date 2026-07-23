import asyncio

from idea.evaluation import (
    ComputedComponentConfig,
    ComputedEvaluator,
    ComputedResultManifest,
    EmbeddingDiversityComponentConfig,
    EmbeddingDiversityEvaluator,
    EvaluationRecord,
    EvaluationStatus,
    LLMPairwiseComponentConfig,
    LLMPairwiseEvaluator,
    attach_evaluation_record,
    candidate_identifier,
)


class _Critic:
    def get_tournament_ranks(self, ideas, _idea_type, **kwargs):
        kwargs["details"].append({"round": 1, "pairs": []})
        return {index: 1500.0 + index for index, _idea in enumerate(ideas)}


class _LegacyCritic:
    def get_tournament_ranks(
        self,
        ideas,
        _idea_type,
        _rounds,
        _progress_callback,
        details,
        _full_rounds,
        should_stop=None,
    ):
        assert callable(should_stop)
        details.append({"round": 1, "pairs": []})
        return {index: 1400.0 + index for index, _idea in enumerate(ideas)}


class _RoundsCritic:
    def __init__(self):
        self.rounds = None

    def get_tournament_ranks(self, ideas, _idea_type, **kwargs):
        self.rounds = kwargs["rounds"]
        return {index: 1500.0 for index, _idea in enumerate(ideas)}


class _ExplodingCritic:
    def get_tournament_ranks(self, _ideas, _idea_type, **_kwargs):
        raise RuntimeError("judge backend unavailable")


def _run_pairwise(critic):
    population = [
        {"id": "idea-a", "idea": "A"},
        {"id": "idea-b", "idea": "B"},
    ]
    details = []
    result = LLMPairwiseEvaluator(
        LLMPairwiseComponentConfig(id="quality")
    ).evaluate_population(
        critic,
        population,
        "test",
        rounds=1,
        progress_callback=None,
        details=details,
        full_tournament_rounds=1,
        should_stop=lambda: False,
        comparison_cache={},
        evaluation_generation=2,
    )
    return population, result


def test_pairwise_adapter_returns_typed_records_and_attaches_them():
    population, result = _run_pairwise(_Critic())

    assert result.scores == {0: 1500.0, 1: 1501.0}
    assert result.details == [{"round": 1, "pairs": []}]
    assert result.batch.generation == 2
    assert result.batch.records[0].metrics == {"quality_elo": 1500.0}
    assert result.batch.records[0].component_kind.value == "llm_pairwise"
    assert result.batch.records[0].component_version == "1"
    assert result.batch.records[0].scope.value == "selection_pool"
    assert result.batch.records[0].generation == 2
    assert population[0]["evaluations"]["quality"]["status"] == "ok"


def test_pairwise_adapter_preserves_legacy_critic_signature_fallback():
    _population, result = _run_pairwise(_LegacyCritic())

    assert result.scores == {0: 1400.0, 1: 1401.0}


def test_pairwise_adapter_honors_component_rounds_override():
    critic = _RoundsCritic()
    population = [{"id": "idea-a", "idea": "A"}]

    LLMPairwiseEvaluator(
        LLMPairwiseComponentConfig(id="quality", rounds=4)
    ).evaluate_population(
        critic,
        population,
        "test",
        rounds=1,
        progress_callback=None,
        details=[],
        full_tournament_rounds=1,
        should_stop=lambda: False,
        comparison_cache={},
        evaluation_generation=2,
    )

    assert critic.rounds == 4


def test_pairwise_adapter_converts_component_exception_to_failed_records():
    population, result = _run_pairwise(_ExplodingCritic())

    assert result.scores == {}
    assert [record.status for record in result.batch.records] == [
        EvaluationStatus.FAILED,
        EvaluationStatus.FAILED,
    ]
    assert result.batch.records[0].error == "judge backend unavailable"
    assert population[0]["evaluations"]["quality"]["status"] == "failed"


def test_diversity_adapter_records_failures_instead_of_coercing_to_zero():
    population = [
        {"id": "idea-a", "idea": "A"},
        {"id": "idea-b", "idea": "B"},
    ]

    async def scores(_population):
        return {0: 0.25, 1: float("nan")}

    result = asyncio.run(
        EmbeddingDiversityEvaluator(
            EmbeddingDiversityComponentConfig(id="diversity")
        ).evaluate_population(
            population,
            scores,
            evaluation_generation=3,
        )
    )

    assert result.scores == {0: 0.25}
    assert result.batch.records[0].status == EvaluationStatus.OK
    assert result.batch.records[1].status == EvaluationStatus.FAILED
    assert population[1]["evaluations"]["diversity"]["status"] == "failed"


def test_diversity_adapter_converts_component_exception_to_failed_records():
    population = [
        {"id": "idea-a", "idea": "A"},
        {"id": "idea-b", "idea": "B"},
    ]

    async def unavailable(_population):
        raise RuntimeError("embedding backend unavailable")

    result = asyncio.run(
        EmbeddingDiversityEvaluator(
            EmbeddingDiversityComponentConfig(id="diversity")
        ).evaluate_population(
            population,
            unavailable,
            evaluation_generation=3,
        )
    )

    assert result.scores == {}
    assert [record.status for record in result.batch.records] == [
        EvaluationStatus.FAILED,
        EvaluationStatus.FAILED,
    ]
    assert result.batch.records[0].error == "embedding backend unavailable"
    assert population[1]["evaluations"]["diversity"]["status"] == "failed"


def test_computed_evaluator_enforces_declared_timeout_as_failed_record():
    class SlowRunner:
        async def run(self, _candidate, **_kwargs):
            await asyncio.sleep(0.05)
            return ComputedResultManifest(status="ok", metrics={"score": 1.0})

    population = [{"id": "idea-a", "idea": "A"}]
    result = asyncio.run(
        ComputedEvaluator(
            ComputedComponentConfig(
                id="simulation",
                runner_ref="slow-v1",
                metric_names=["score"],
                timeout_seconds=0.01,
            )
        ).evaluate_population(
            population,
            SlowRunner(),
            evaluation_generation=1,
        )
    )

    assert result.scores == {}
    assert result.batch.records[0].status == EvaluationStatus.FAILED
    assert "0.01-second timeout" in result.batch.records[0].error


def test_candidate_identifier_is_stable_and_safe_for_unstructured_candidates():
    first = candidate_identifier({"idea": {"content": "x"}}, 0)
    second = candidate_identifier({"idea": {"content": "x"}}, 0)

    assert first == second
    assert first.startswith("candidate-0-")


def test_candidate_identifier_ignores_attached_evaluation_and_runtime_metadata():
    candidate = {"idea": {"content": "x"}}
    original = candidate_identifier(candidate, 0)
    attach_evaluation_record(
        candidate,
        EvaluationRecord(
            candidate_id=original,
            component_id="quality",
            status=EvaluationStatus.OK,
            metrics={"score": 0.8},
        ),
    )
    candidate.update(
        {
            "selection_metrics": {"fitness": 0.8},
            "generation_metrics": {"0": {"fitness": 0.8}},
            "survival_metrics": {"age": 1},
            "ratings": {"auto": 1600},
            "island_id": 2,
        }
    )

    assert candidate_identifier(candidate, 0) == original
    assert candidate["_evaluation_candidate_id"] == original
