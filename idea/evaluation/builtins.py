"""Small, reviewed computed scorers that are safe to run in-process.

Built-in runners never execute candidate-supplied source, import arbitrary
callables, access the network, or launch subprocesses.  They exist both as
useful low-cost metrics and as executable examples of the generic computed
evaluator contract.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .models import ComputedComponentConfig
from .script_contract import ComputedResultManifest


CHARACTER_COUNT_RUNNER_REF = "builtin.character_count"


def _idea_text_fields(candidate: Any) -> Dict[str, str]:
    idea = candidate.get("idea", candidate) if isinstance(candidate, Mapping) else candidate
    if hasattr(idea, "model_dump"):
        idea = idea.model_dump()
    elif hasattr(idea, "dict"):
        idea = idea.dict()

    if isinstance(idea, Mapping):
        return {
            "title": str(idea.get("title") or ""),
            "content": str(idea.get("content") or ""),
        }
    return {"title": "", "content": str(idea or "")}


class CharacterCountRunner:
    """Count Unicode characters in one configured part of an idea.

    The component must declare exactly one metric name.  Parameters are:

    - ``field``: ``content`` (default), ``title``, or ``title_and_content``;
    - ``include_whitespace``: whether whitespace code points count (default true).

    The resulting finite number already follows the evaluator-wide convention:
    a larger value is fitter.  Whether length is a sensible fitness ingredient
    is intentionally left to the run configuration.
    """

    _FIELDS = frozenset({"content", "title", "title_and_content"})
    _PARAMETERS = frozenset({"field", "include_whitespace"})

    async def run(
        self,
        candidate: Any,
        *,
        candidate_id: str,
        generation: Optional[int],
        config: ComputedComponentConfig,
    ) -> ComputedResultManifest:
        del candidate_id, generation

        if len(config.metric_names) != 1:
            raise ValueError(
                f"{CHARACTER_COUNT_RUNNER_REF} requires exactly one metric name"
            )

        unknown_parameters = sorted(set(config.parameters) - self._PARAMETERS)
        if unknown_parameters:
            raise ValueError(
                "unsupported character-count parameters: "
                + ", ".join(unknown_parameters)
            )

        field = str(config.parameters.get("field", "content"))
        if field not in self._FIELDS:
            raise ValueError(
                "character-count field must be content, title, or title_and_content"
            )
        include_whitespace = config.parameters.get("include_whitespace", True)
        if not isinstance(include_whitespace, bool):
            raise ValueError("include_whitespace must be a boolean")

        fields = _idea_text_fields(candidate)
        if field == "title_and_content":
            text = fields["title"] + "\n" + fields["content"]
        else:
            text = fields[field]
        if not include_whitespace:
            text = "".join(character for character in text if not character.isspace())

        metric_name = config.metric_names[0]
        return ComputedResultManifest(
            status="ok",
            metrics={metric_name: float(len(text))},
            summary=f"Counted Unicode characters in {field.replace('_', ' ')}.",
        )


def create_builtin_computed_runner(runner_ref: str) -> Optional[Any]:
    """Resolve an allow-listed in-process runner without dynamic imports."""

    if runner_ref == CHARACTER_COUNT_RUNNER_REF:
        return CharacterCountRunner()
    return None


def is_builtin_computed_runner(runner_ref: str) -> bool:
    return runner_ref == CHARACTER_COUNT_RUNNER_REF


__all__ = [
    "CHARACTER_COUNT_RUNNER_REF",
    "CharacterCountRunner",
    "create_builtin_computed_runner",
    "is_builtin_computed_runner",
]
