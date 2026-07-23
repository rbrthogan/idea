"""Shared idea and candidate-record boundaries.

Runtime candidates remain ordinary dictionaries for compatibility with the
evolution engine.  ``CandidateRecord`` gives those dictionaries one explicit,
versioned boundary without nesting or whitelisting their extension metadata.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


CANDIDATE_RECORD_VERSION = 1
CANDIDATE_RECORD_VERSION_KEY = "candidate_record_version"
_CANDIDATE_CORE_KEYS = frozenset(
    {CANDIDATE_RECORD_VERSION_KEY, "id", "idea", "parent_ids"}
)


class CandidateCodecError(ValueError):
    """A candidate value cannot be represented safely on the JSON wire."""


class UnsupportedCandidateRecordVersion(CandidateCodecError):
    """A persisted candidate uses a newer or otherwise unsupported schema."""


class Idea(BaseModel):
    """Represents the human-readable part of a candidate."""

    title: Optional[str] = None
    content: str


def encode_json_safe(value: Any) -> Any:
    """Recursively convert supported runtime values to strict JSON values.

    UUIDs use their canonical strings and Pydantic models use their public
    field representation.  Unsupported objects fail explicitly instead of
    falling back to ``str(value)``, which previously hid metadata loss.
    """

    if isinstance(value, CandidateRecord):
        return value.to_wire_dict()
    if isinstance(value, Enum):
        return encode_json_safe(value.value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return encode_json_safe(
            value.model_dump(mode="python", by_alias=True, exclude_none=False)
        )
    if is_dataclass(value) and not isinstance(value, type):
        return encode_json_safe(asdict(value))
    if isinstance(value, Mapping):
        encoded: Dict[str, Any] = {}
        for raw_key, item in value.items():
            if isinstance(raw_key, str):
                key = raw_key
            elif isinstance(raw_key, (str, int, float, bool)) or raw_key is None:
                if isinstance(raw_key, float) and not math.isfinite(raw_key):
                    raise CandidateCodecError("mapping keys must be finite")
                key = str(raw_key)
            else:
                raise CandidateCodecError(
                    f"mapping key {raw_key!r} is not JSON-compatible"
                )
            if key in encoded:
                raise CandidateCodecError(
                    f"mapping keys collide after JSON conversion: {key!r}"
                )
            encoded[key] = encode_json_safe(item)
        return encoded
    if isinstance(value, (list, tuple)):
        return [encode_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [encode_json_safe(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CandidateCodecError("non-finite floats are not JSON-safe")
        return value
    raise CandidateCodecError(
        f"unsupported candidate metadata type: {type(value).__name__}"
    )


def decode_json_safe(value: Any) -> Any:
    """Recursively validate and copy a JSON-wire value.

    JSON has no type marker that distinguishes an arbitrary UUID string from
    normal text.  Candidate identity fields are restored by
    ``CandidateRecord.from_wire_dict``; UUIDs inside extension metadata remain
    canonical strings so their meaning is not guessed.
    """

    if isinstance(value, BaseModel) or is_dataclass(value):
        return decode_json_safe(encode_json_safe(value))
    if isinstance(value, Mapping):
        decoded: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CandidateCodecError("JSON object keys must be strings")
            decoded[key] = decode_json_safe(item)
        return decoded
    if isinstance(value, (list, tuple)):
        return [decode_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CandidateCodecError("non-finite floats are not JSON-safe")
        return value
    raise CandidateCodecError(
        f"unsupported JSON-wire value: {type(value).__name__}"
    )


def _restore_uuid(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError):
        return value


def _restore_idea(value: Any) -> Any:
    if not isinstance(value, Mapping) or "content" not in value:
        return value
    # Constructing Idea from a richer mapping would discard its unknown keys.
    # Leave that mapping intact so extension data wins over convenience.
    if not set(value).issubset({"title", "content"}):
        return dict(value)
    return Idea.model_validate(value)


class CandidateRecord(BaseModel):
    """Versioned view over the app's existing flat candidate dictionary.

    Known identity/lineage fields are typed explicitly.  Every other top-level
    field lives in ``extensions`` internally and is flattened again by
    ``to_candidate_dict``/``to_wire_dict``.  This keeps the existing dict wire
    shape while preventing newly introduced metadata from being ignored.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    schema_version: Literal[CANDIDATE_RECORD_VERSION] = CANDIDATE_RECORD_VERSION
    id: Any = None
    idea: Any
    parent_ids: List[Any] = Field(default_factory=list)
    extensions: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_candidate(cls, candidate: Mapping[str, Any]) -> "CandidateRecord":
        if not isinstance(candidate, Mapping):
            raise CandidateCodecError("candidate must be a mapping")
        if "idea" not in candidate:
            raise CandidateCodecError("candidate mapping is missing 'idea'")

        version = candidate.get(
            CANDIDATE_RECORD_VERSION_KEY,
            CANDIDATE_RECORD_VERSION,
        )
        if version != CANDIDATE_RECORD_VERSION:
            raise UnsupportedCandidateRecordVersion(
                f"unsupported candidate record version: {version!r}"
            )

        raw_parent_ids = candidate.get("parent_ids", [])
        if raw_parent_ids is None:
            raw_parent_ids = []
        if not isinstance(raw_parent_ids, (list, tuple)):
            raise CandidateCodecError("candidate parent_ids must be a list")

        extensions = {
            key: value
            for key, value in candidate.items()
            if key not in _CANDIDATE_CORE_KEYS
        }
        if any(not isinstance(key, str) for key in extensions):
            raise CandidateCodecError("candidate extension keys must be strings")

        return cls(
            schema_version=CANDIDATE_RECORD_VERSION,
            id=candidate.get("id"),
            idea=candidate["idea"],
            parent_ids=list(raw_parent_ids),
            extensions=extensions,
        )

    @classmethod
    def from_wire_dict(cls, payload: Mapping[str, Any]) -> "CandidateRecord":
        decoded = decode_json_safe(payload)
        record = cls.from_candidate(decoded)
        return cls(
            schema_version=record.schema_version,
            id=_restore_uuid(record.id),
            idea=_restore_idea(record.idea),
            parent_ids=[_restore_uuid(parent_id) for parent_id in record.parent_ids],
            extensions=record.extensions,
        )

    def to_candidate_dict(self, *, include_version: bool = False) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if include_version:
            result[CANDIDATE_RECORD_VERSION_KEY] = self.schema_version
        result["id"] = self.id
        result["idea"] = self.idea
        result["parent_ids"] = list(self.parent_ids)
        result.update(self.extensions)
        return result

    def to_wire_dict(self) -> Dict[str, Any]:
        return encode_json_safe(self.to_candidate_dict(include_version=True))


def merge_candidate_records(source: Any, transformed: Any) -> Any:
    """Overlay a transformation while carrying every source extension field.

    Formatters and refiners are allowed to replace ``idea`` and other fields
    they return.  Metadata they know nothing about remains on the candidate.
    Non-candidate inputs retain their historical pass-through behavior.
    """

    if isinstance(source, CandidateRecord):
        source_payload: Any = source.to_candidate_dict(include_version=True)
    else:
        source_payload = source

    if not isinstance(source_payload, Mapping) or "idea" not in source_payload:
        return transformed

    include_version = CANDIDATE_RECORD_VERSION_KEY in source_payload
    merged = dict(source_payload)
    if isinstance(transformed, CandidateRecord):
        transformed_payload: Any = transformed.to_candidate_dict(include_version=True)
        include_version = True
    elif isinstance(transformed, Mapping):
        transformed_payload = transformed
        include_version = include_version or (
            CANDIDATE_RECORD_VERSION_KEY in transformed_payload
        )
    else:
        transformed_payload = {"idea": transformed}

    merged.update(transformed_payload)
    return CandidateRecord.from_candidate(merged).to_candidate_dict(
        include_version=include_version
    )


def candidate_to_wire(value: Any) -> Any:
    """Encode a candidate-shaped value without changing its flat wire shape."""

    if isinstance(value, CandidateRecord):
        return value.to_wire_dict()
    if isinstance(value, Mapping) and "idea" in value:
        return CandidateRecord.from_candidate(value).to_wire_dict()
    return encode_json_safe(value)


def candidate_from_wire(value: Any) -> Any:
    """Decode a candidate wire value to the legacy dict/Idea runtime shapes."""

    if isinstance(value, Mapping) and (
        "idea" in value or CANDIDATE_RECORD_VERSION_KEY in value
    ):
        return CandidateRecord.from_wire_dict(value).to_candidate_dict()

    decoded = decode_json_safe(value)
    if (
        isinstance(decoded, Mapping)
        and "content" in decoded
        and set(decoded).issubset({"title", "content"})
    ):
        return Idea.model_validate(decoded)
    return decoded


__all__ = [
    "CANDIDATE_RECORD_VERSION",
    "CANDIDATE_RECORD_VERSION_KEY",
    "CandidateCodecError",
    "CandidateRecord",
    "Idea",
    "UnsupportedCandidateRecordVersion",
    "candidate_from_wire",
    "candidate_to_wire",
    "decode_json_safe",
    "encode_json_safe",
    "merge_candidate_records",
]
