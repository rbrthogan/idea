"""Compile template guidance once and render named slots without reparsing data.

Requirements, output guidance, model responses and candidate text are literal
values. Braces inside them must never become a second round of prompt slots.
"""

import json
import re
from typing import Any


_SLOT = re.compile(r"\{\{|\}\}|\{([a-zA-Z_][a-zA-Z_0-9]*)\}")
_GUIDANCE_SLOT = re.compile(r"(?<!\{)\{(requirements|requirements_digest)\}(?!\})")
_CANDIDATE_STAGES = {"generate", "refine", "breed"}
_GENERATION_PROTOCOL = "Return only the candidate, using Title: for its title and Content: for its complete body."
_FORMAT_PROTOCOL = (
    "Extract the existing title and complete body into the requested title/content JSON schema. "
    "Preserve the body's wording, facts, structure, code, and punctuation. "
    "Do not shorten, expand, or rewrite the body to meet word counts or other requirements. "
    "If a title is absent, supply a suitable short title. Return only the schema fields."
)
_COMPARISON_PROTOCOL = "Finish with exactly one verdict: Result: A, Result: B, or Result: tie."
_DIRECTION_PROTOCOL = (
    "Return only one concise generation direction for a later model to follow. "
    "Do not write the candidate, a candidate title, commentary, or Markdown headings."
)


def _render_source(source: str, values: dict[str, Any], escaped_depth=None) -> str:
    escaped_depth = [0] if escaped_depth is None else escaped_depth
    def substitute(match):
        if match.group(0) == "{{":
            escaped_depth[0] += 1
            return "{"
        if match.group(0) == "}}":
            if escaped_depth[0]:
                escaped_depth[0] -= 1
                return "}"
            return "}}"
        name = match.group(1)
        if name not in values:
            raise KeyError(name)
        return str(values[name])
    return _SLOT.sub(substitute, source)


class CompiledPrompt(str):
    """A string-compatible prompt retaining which pieces are literal guidance."""

    def __new__(cls, parts):
        value = super().__new__(cls, "".join(text for _is_source, text in parts))
        value.parts = tuple(parts)
        return value

    def format(self, **values):
        escaped_depth = [0]
        return "".join(_render_source(text, values, escaped_depth) if is_source else text
                       for is_source, text in self.parts)


def render_prompt(prompt: str, **values) -> str:
    if isinstance(prompt, CompiledPrompt):
        return prompt.format(**values)
    return _render_source(str(prompt), values)


def bind_prompt(source: str, **literal_values) -> CompiledPrompt:
    """Bind author-owned metadata now while retaining later candidate slots."""
    names = "|".join(re.escape(name) for name in literal_values)
    pattern = re.compile(r"(?<!\{)\{(" + names + r")\}(?!\})")
    source_parts = source.parts if isinstance(source, CompiledPrompt) else ((True, source),)
    parts = []
    for is_source, text in source_parts:
        if not is_source:
            parts.append((False, text))
            continue
        cursor = 0
        for match in pattern.finditer(text):
            parts.append((True, text[cursor:match.start()]))
            parts.append((False, str(literal_values[match.group(1)])))
            cursor = match.end()
        parts.append((True, text[cursor:]))
    return CompiledPrompt(parts)


def requirements_text(template) -> str:
    structured = getattr(template, "requirements", None) or []
    parts = ["\n".join(f"- {item}" for item in structured)] if structured else []
    legacy = getattr(template, "special_requirements", None)
    if legacy:
        # Preserve legacy prose verbatim; no guessing which lines are rules.
        parts.append(legacy)
    return "\n\n".join(parts)


def output_text(template) -> str:
    output = getattr(template, "output", None)
    if output is None:
        return ""
    get = output.get if isinstance(output, dict) else lambda key, default=None: getattr(output, key, default)
    lines = []
    if get("title"):
        lines.append(f"Title guidance: {get('title')}")
    if get("content"):
        lines.append(f"Body guidance: {get('content')}")
    count = get("word_count")
    if count:
        count_get = count.get if isinstance(count, dict) else lambda key: getattr(count, key, None)
        minimum, maximum = count_get("min"), count_get("max")
        if minimum is not None and minimum == maximum:
            lines.append(f"Body length: exactly {minimum} words, excluding the title.")
        elif minimum is not None and maximum is not None:
            lines.append(f"Body length: {minimum}–{maximum} words, excluding the title.")
        elif minimum is not None:
            lines.append(f"Body length: at least {minimum} words, excluding the title.")
        elif maximum is not None:
            lines.append(f"Body length: at most {maximum} words, excluding the title.")
    return "\n".join(lines)


def compile_prompt(source: str, template, stage: str, *, inject: bool = True) -> CompiledPrompt:
    """Compile guidance for one stage, keeping that stage's response protocol."""
    requirements = requirements_text(template)
    output = output_text(template)
    parts = []
    requirements_included = False
    source_parts = source.parts if isinstance(source, CompiledPrompt) else ((True, source),)
    for is_source, text in source_parts:
        if not is_source:
            parts.append((False, text))
            continue
        cursor = 0
        for match in _GUIDANCE_SLOT.finditer(text):
            parts.append((True, text[cursor:match.start()]))
            # Both legacy aliases mean the same complete contract, included once.
            parts.append((False, "" if requirements_included or stage == "format" else requirements))
            requirements_included = True
            cursor = match.end()
        parts.append((True, text[cursor:]))
    if not inject:
        return CompiledPrompt(parts)

    if stage in {"context", "genotype"}:
        # Concepts are inputs to a later task. Appending that task's narrative,
        # title, or length instructions can turn extraction into reconstruction.
        # Explicit author-written requirements placeholders above still work.
        return CompiledPrompt(parts)

    if stage == "direction":
        if requirements and not requirements_included:
            parts.append((False,
                "\n\nBackground requirements for the future candidate (not this response):\n"
                + requirements))
        parts.append((False, "\n\n" + _DIRECTION_PROTOCOL))
        return CompiledPrompt(parts)

    if stage == "format":
        parts.append((False, "\n\n" + _FORMAT_PROTOCOL))
        return CompiledPrompt(parts)

    if stage in _CANDIDATE_STAGES:
        heading = "Candidate requirements:"
        output_heading = "Candidate output guidance:"
    else:
        heading = "Requirements for the candidate (keep this stage's own response format):"
        output_heading = "Candidate output to assess or preserve (not this stage's response format):"
    if requirements and not requirements_included:
        parts.append((False, f"\n\n{heading}\n{requirements}"))
    if output:
        parts.append((False, f"\n\n{output_heading}\n{output}"))
    if stage in _CANDIDATE_STAGES:
        parts.append((False, "\n\n" + _GENERATION_PROTOCOL))
    elif stage == "comparison":
        parts.append((False, "\n\n" + _COMPARISON_PROTOCOL))
    return CompiledPrompt(parts)


def generation_prompt(prompts, direction: str) -> str:
    build = getattr(prompts, "build_generation_prompt", None)
    if callable(build):
        return build(direction)
    # Legacy Python wrappers and lightweight integrations retain their API.
    template = getattr(prompts, "template", None)
    requirements = getattr(template, "special_requirements", None)
    return f"{direction}\n\nConstraints:\n{requirements}" if requirements else direction


def add_oracle_guidance(prompt: str, template) -> str:
    """Oracle's complete rendered prompt is data, never a format template."""
    sections = []
    requirements = requirements_text(template)
    output = output_text(template)
    if requirements and requirements not in prompt:
        sections.append("Requirements for replacement candidates (retain the Oracle response protocol):\n" + requirements)
    if output and output not in prompt:
        sections.append("Output guidance for replacement candidates (not the Oracle analysis):\n" + output)
    return prompt + ("\n\n" + "\n\n".join(sections) if sections else "")


def _ambiguous_envelope_body(body: str) -> bool:
    """Reject structural ambiguity; this is not a meaning or word-count check."""
    stripped = body.strip()
    if stripped.lower() in {"none", "null", "undefined", "no response", "no response.",
                            "model response was empty.", "..."}:
        return True
    if re.fullmatch(r"\[[^\[\]\r\n]{0,200}\]|<[^<>\r\n]{0,200}>", stripped):
        return True
    if not re.search(r"\w", stripped):
        return True

    if stripped.startswith(("{", "[")):
        try:
            payload = json.loads(stripped)
        except (ValueError, RecursionError):
            payload = None
        if payload == {} or payload == []:
            return True
        if isinstance(payload, dict) and {"title", "content"}.issubset(payload):
            return True
        if isinstance(payload, list) and any(
                isinstance(item, dict) and {"title", "content"}.issubset(item) for item in payload):
            return True

    lines = body.splitlines()
    nonblank_indices = [index for index, line in enumerate(lines) if line.strip()]
    open_fence = None
    first_fence = None
    last_fence_end = None
    for index, line in enumerate(lines):
        # A second field label, even alone or indented, makes the boundary
        # unclear. Commentary labels likewise need the existing formatter.
        if re.match(r"^\s*(?:Title|Content)\s*:", line, re.IGNORECASE):
            return True
        if re.match(r"^[\s>#*_\-]*(?:Analysis|Word\s*count|Explanation|Notes)\s*:",
                    line, re.IGNORECASE):
            return True
        if open_fence is not None:
            marker, length = open_fence
            if re.fullmatch(r"\s*" + re.escape(marker) + "{" + str(length) + r",}\s*", line):
                open_fence = None
                last_fence_end = index
            continue
        fence = re.match(r"^\s*(`{3,}|~{3,})[^\r\n]*$", line)
        if fence:
            marker = fence.group(1)
            open_fence = (marker[0], len(marker))
            if first_fence is None:
                first_fence = index
    if open_fence is not None:
        return True
    # Embedded balanced code is content. A fence around the entire body may
    # instead be a response wrapper and should be interpreted by the formatter.
    return bool(nonblank_indices and first_fence == nonblank_indices[0]
                and last_fence_end == nonblank_indices[-1])


def extract_title_content_envelope(text: Any):
    """Extract an unambiguous envelope; return None for the existing fallback.

    Accepted text is preserved, not semantically validated. In particular,
    unlabeled commentary cannot reliably be distinguished from candidate prose.
    """
    if not isinstance(text, str):
        return None
    lines = text.splitlines(keepends=True)
    if not lines or not lines[0].startswith("Title:"):
        return None
    title = lines[0][len("Title:"):].strip()
    if not title or not re.search(r"\w", title) or title.startswith(("```", "~~~")):
        return None
    header = 1
    while header < len(lines) and not lines[header].strip():
        header += 1
    if header >= len(lines) or not lines[header].startswith("Content:"):
        return None
    first_body_line = lines[header][len("Content:"):]
    remaining = "".join(lines[header + 1:])
    # Remove the single separator space after the label, not indentation that
    # may be meaningful in code or preformatted content.
    inline_body = first_body_line[1:] if first_body_line.startswith(" ") else first_body_line
    body = inline_body + remaining if first_body_line.strip() else remaining
    if not body.strip() or _ambiguous_envelope_body(body):
        return None
    return {"title": title, "content": body}
