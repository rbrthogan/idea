import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

from idea.evolution_orchestrator import EvolutionOrchestrator
from idea.evolution_persistence import EvolutionSerializer
from idea.models import (
    CANDIDATE_RECORD_VERSION,
    CANDIDATE_RECORD_VERSION_KEY,
    CandidateCodecError,
    CandidateRecord,
    Idea,
    UnsupportedCandidateRecordVersion,
    candidate_from_wire,
    candidate_to_wire,
    merge_candidate_records,
)
from idea.ratings import extract_title_content


class _ArtifactPayload(BaseModel):
    id: uuid.UUID
    media_type: str
    metadata: dict = Field(default_factory=dict)


class _EvaluationPayload(BaseModel):
    component_id: str
    artifact_ids: list[uuid.UUID]
    evidence: dict = Field(default_factory=dict)


def _rich_candidate():
    candidate_id = uuid.uuid4()
    parent_id = uuid.uuid4()
    artifact_id = uuid.uuid4()
    provenance_event_id = uuid.uuid4()
    artifact = _ArtifactPayload(
        id=artifact_id,
        media_type="image/png",
        metadata={"dimensions": [1200, 900]},
    )
    evaluation = _EvaluationPayload(
        component_id="visual_judge",
        artifact_ids=[artifact_id],
        evidence={"scores": [{"name": "legibility", "value": 0.8}]},
    )
    return {
        "id": candidate_id,
        "idea": Idea(title="Residue Atlas", content="Map primes by two residues."),
        "parent_ids": [parent_id],
        "artifacts": [artifact],
        "evaluations": {"visual_judge": evaluation},
        "provenance": {
            "events": [
                {
                    "event_id": provenance_event_id,
                    "agent": "artifact-producer",
                }
            ]
        },
        "future_extension": {
            "nested": [{"enabled": True, "weights": [0.75, 0.25]}]
        },
    }


def test_candidate_record_round_trips_nested_extensions_without_whitelisting():
    candidate = _rich_candidate()
    record = CandidateRecord.from_candidate(candidate)

    assert set(record.extensions) == {
        "artifacts",
        "evaluations",
        "provenance",
        "future_extension",
    }
    assert isinstance(record.extensions["artifacts"][0], _ArtifactPayload)

    wire = record.to_wire_dict()
    # This is strict JSON, not json.dumps(default=str).
    json.dumps(wire, allow_nan=False)
    assert wire[CANDIDATE_RECORD_VERSION_KEY] == CANDIDATE_RECORD_VERSION
    assert wire["id"] == str(candidate["id"])
    assert wire["parent_ids"] == [str(candidate["parent_ids"][0])]
    assert wire["artifacts"][0]["id"] == str(candidate["artifacts"][0].id)
    assert wire["evaluations"]["visual_judge"]["artifact_ids"] == [
        str(candidate["artifacts"][0].id)
    ]
    assert wire["provenance"]["events"][0]["event_id"] == str(
        candidate["provenance"]["events"][0]["event_id"]
    )
    assert wire["future_extension"] == candidate["future_extension"]

    restored = CandidateRecord.from_wire_dict(wire).to_candidate_dict()
    assert restored["id"] == candidate["id"]
    assert restored["parent_ids"] == candidate["parent_ids"]
    assert restored["idea"] == candidate["idea"]
    assert restored["artifacts"] == wire["artifacts"]
    assert restored["evaluations"] == wire["evaluations"]
    assert restored["provenance"] == wire["provenance"]
    assert restored["future_extension"] == candidate["future_extension"]
    assert candidate_to_wire(restored) == wire


def test_evolution_serializer_uses_versioned_recursive_candidate_codec():
    candidate = _rich_candidate()

    wire = EvolutionSerializer.serialize_idea(candidate)
    restored = EvolutionSerializer.deserialize_idea(wire)

    assert wire[CANDIDATE_RECORD_VERSION_KEY] == CANDIDATE_RECORD_VERSION
    assert isinstance(restored["idea"], Idea)
    assert restored["id"] == candidate["id"]
    assert restored["artifacts"] == wire["artifacts"]
    assert restored["evaluations"] == wire["evaluations"]
    assert restored["future_extension"] == candidate["future_extension"]


def test_legacy_candidate_wire_without_version_still_decodes():
    candidate_id = uuid.uuid4()
    legacy = {
        "id": str(candidate_id),
        "idea": {"title": "Legacy", "content": "Still supported"},
        "parent_ids": [],
        "unknown_metadata": {"nested": [1, 2, 3]},
    }

    restored = candidate_from_wire(legacy)

    assert restored["id"] == candidate_id
    assert restored["idea"] == Idea(title="Legacy", content="Still supported")
    assert restored["unknown_metadata"] == {"nested": [1, 2, 3]}


def test_direct_idea_remains_compatible_with_persistence_codec():
    idea = Idea(title="Standalone", content="Legacy direct model")

    wire = EvolutionSerializer.serialize_idea(idea)
    restored = EvolutionSerializer.deserialize_idea(wire)

    assert wire == {"title": "Standalone", "content": "Legacy direct model"}
    assert restored == idea


def test_merge_candidate_records_preserves_extensions_and_new_content():
    candidate = _rich_candidate()
    transformed = {
        "id": candidate["id"],
        "idea": Idea(title="Refined Atlas", content="Use CRT coordinates."),
        "parent_ids": candidate["parent_ids"],
        "formatter_note": "schema-recovered",
    }

    merged = merge_candidate_records(candidate, transformed)

    assert merged["idea"] == transformed["idea"]
    assert merged["formatter_note"] == "schema-recovered"
    assert merged["artifacts"] is candidate["artifacts"]
    assert merged["evaluations"] is candidate["evaluations"]
    assert merged["provenance"] is candidate["provenance"]
    assert merged["future_extension"] is candidate["future_extension"]
    # Runtime dictionaries remain unwrapped and do not gain the wire version.
    assert CANDIDATE_RECORD_VERSION_KEY not in merged


def test_orchestrator_format_boundary_carries_extension_metadata_forward():
    candidate = _rich_candidate()

    class _Critic:
        def refine(self, value, _idea_type):
            return {
                "id": value["id"],
                "idea": "Refined mathematical body",
                "parent_ids": value["parent_ids"],
            }

    class _Formatter:
        def format_idea(self, value, _idea_type):
            return {
                "id": value["id"],
                "idea": Idea(
                    title="Formatted refinement",
                    content="Refined mathematical body",
                ),
                "parent_ids": value["parent_ids"],
            }

    engine = SimpleNamespace(
        critic=_Critic(),
        formatter=_Formatter(),
        idea_type="prime_visualization",
    )
    orchestrator = EvolutionOrchestrator(engine, progress_callback=lambda _payload: None)

    result = asyncio.run(
        orchestrator._prepare_idea_for_population(candidate, refine=True)
    )

    assert result["idea"].title == "Formatted refinement"
    assert result["artifacts"] is candidate["artifacts"]
    assert result["evaluations"] is candidate["evaluations"]
    assert result["provenance"] is candidate["provenance"]
    assert result["future_extension"] is candidate["future_extension"]


def test_candidate_record_is_compatible_with_rating_extraction():
    record = CandidateRecord.from_candidate(_rich_candidate())

    assert extract_title_content(record) == {
        "title": "Residue Atlas",
        "content": "Map primes by two residues.",
    }


def test_candidate_codec_rejects_future_versions_and_unsafe_values():
    with pytest.raises(UnsupportedCandidateRecordVersion):
        CandidateRecord.from_candidate(
            {
                CANDIDATE_RECORD_VERSION_KEY: 999,
                "id": "candidate",
                "idea": "future",
            }
        )

    with pytest.raises(CandidateCodecError, match="unsupported candidate metadata"):
        candidate_to_wire(
            {
                "id": "candidate",
                "idea": "unsafe",
                "parent_ids": [],
                "extension": object(),
            }
        )
