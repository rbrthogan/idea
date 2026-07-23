import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parent.parent
VIEWER_JS = ROOT / "idea" / "static" / "js" / "viewer.js"
RATER_JS = ROOT / "idea" / "static" / "js" / "rater.js"


def _extract_function_bundle(script: str, first_function: str, last_function: str) -> str:
    start = script.index(f"function {first_function}")
    end = script.index(f"function {last_function}", start)
    return script[start:end]


def _render_markdown(markdown: str) -> str:
    script = VIEWER_JS.read_text()
    render_bundle = _extract_function_bundle(script, "normalizeMarkdownStructure", "createCardPreview")
    node_script = f"""
{render_bundle}
const input = {json.dumps(markdown)};
process.stdout.write(renderMarkdown(input));
"""
    result = subprocess.run(
        ["node", "-e", node_script],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _render_rater_markdown(markdown: str) -> str:
    script = RATER_JS.read_text()
    start = script.index("function escapeHtml")
    end = script.index("async function refreshPair", start)
    render_bundle = script[start:end]
    node_script = f"""
{render_bundle}
const input = {json.dumps(markdown)};
process.stdout.write(renderMarkdown(input));
"""
    result = subprocess.run(
        ["node", "-e", node_script],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


@pytest.fixture(params=[_render_markdown, _render_rater_markdown], ids=["viewer", "rater"])
def markdown_renderer(request):
    return request.param


def test_render_markdown_repairs_inline_common_headings():
    html = _render_markdown(
        "# Overview FluxLoop is a spatial game. "
        "### Components Board: A 7x7 grid. "
        "### Setup The board begins empty. "
        "### Rules & Turn Sequence 1. The Slide: Place a token."
    )

    assert "<h1>Overview</h1>" in html
    assert "<p>FluxLoop is a spatial game.</p>" in html
    assert "<h3>Components</h3>" in html
    assert "<p>Board: A 7x7 grid.</p>" in html
    assert "<h3>Setup</h3>" in html
    assert "<p>The board begins empty.</p>" in html
    assert "<h3>Rules &amp; Turn Sequence</h3>" in html
    assert "<ol><li>The Slide: Place a token.</li>" in html
    assert "### Components" not in html


def test_render_markdown_does_not_repair_hashes_inside_code_blocks():
    html = _render_markdown(
        "```js\nconst label = '### Components Board';\n```\n"
        "### Components Board: A grid."
    )

    assert "const label = '### Components Board';" in html
    assert "<pre><code>const label = '### Components Board';</code></pre>" in html
    assert "<p>const label" not in html
    assert "<h3>Components</h3>" in html
    assert "<p>Board: A grid.</p>" in html


def test_render_markdown_repairs_headings_glued_to_body_and_lists():
    html = _render_markdown(
        "### OverviewHearthfire is a race between two opposing forces.### Components"
        "*   **Board:** A 61-hex grid.*   **Tokens:** 20 stones."
        "### Rules & Turn SequenceOn your turn, perform two actions in order:1.  **Cultivate:** Place a piece."
        "2.  **Thermal Decay:** Remove warmth."
        "### Win Condition*   **Ember:** Reach the edge.*   **Frost:** Block Ember."
        "### Emergent ComplexityThe board disintegrates from the inside out."
    )

    assert "<h3>Overview</h3>" in html
    assert "<p>Hearthfire is a race between two opposing forces.</p>" in html
    assert "<h3>Components</h3>" in html
    assert "<ul><li><strong>Board:</strong> A 61-hex grid.</li>" in html
    assert "<li><strong>Tokens:</strong> 20 stones.</li>" in html
    assert "<h3>Rules &amp; Turn Sequence</h3>" in html
    assert "<ol><li><strong>Cultivate:</strong> Place a piece.</li>" in html
    assert "<li><strong>Thermal Decay:</strong> Remove warmth.</li>" in html
    assert "<h3>Win Condition</h3>" in html
    assert "<h3>Emergent Complexity</h3>" in html
    assert "<p>The board disintegrates from the inside out.</p>" in html
    assert "OverviewHearthfire" not in html
    assert ".###" not in html


@pytest.mark.parametrize(
    "destination",
    [
        "javascript:alert(1)",
        "JaVaScRiPt:alert(1)",
        "java&#x73;cript:alert(1)",
        "java&#999999999999999999;script:alert(1)",
        "java&colon;script:alert(1)",
        "data:text/html,<svg onload=alert(1)>",
        "vbscript:msgbox(1)",
        "file:///etc/passwd",
        "blob:https://example.test/id",
        "//evil.example/payload",
        "https://example.test/\" onmouseover=\"alert(1)",
    ],
)
def test_render_markdown_degrades_unsafe_links_to_inert_text(markdown_renderer, destination):
    html = markdown_renderer(f"[open]({destination})")

    assert "<a " not in html
    assert "href=" not in html
    assert "<svg" not in html
    assert "onmouseover=\"" not in html
    assert "open" in html


@pytest.mark.parametrize(
    "destination",
    [
        "https://example.test/path?one=1&two=2",
        "http://example.test/path",
        "/evolutions/one",
        "./artifact.png",
        "../artifact.png",
        "#evaluation",
        "docs/evaluators.html",
    ],
)
def test_render_markdown_allows_http_and_relative_links(markdown_renderer, destination):
    html = markdown_renderer(f"[open]({destination})")

    assert '<a href="' in html
    assert 'target="_blank"' in html
    assert 'rel="noopener noreferrer"' in html
    assert destination.replace("&", "&amp;") in html


def test_render_markdown_keeps_generated_html_and_svg_inert(markdown_renderer):
    html = markdown_renderer(
        '<svg onload="alert(1)"><script>alert(2)</script></svg>'
        '<img src=x onerror="alert(3)">'
    )

    assert "<svg" not in html
    assert "<script" not in html
    assert "<img" not in html
    assert "&lt;svg" in html
    assert "&lt;script&gt;" in html
    assert "&lt;img" in html


def test_generated_text_sinks_use_escaping_contract():
    viewer = VIEWER_JS.read_text()
    rater = RATER_JS.read_text()

    assert "card.querySelector('.card-title').textContent = ideaData.title || 'Untitled'" in viewer
    assert "card.querySelector('.idea-preview-text').textContent = plainPreview" in viewer
    assert "${escapeHtml(sourceIdea.title || 'Untitled')}" in viewer
    assert "${escapeHtml(sourcePreview)}" in viewer
    assert "lineageModalContent.querySelector('.analysis-text').textContent = idea.oracle_analysis" in viewer
    assert "${escapeHtml(ancestor.title || 'Untitled')}" in viewer
    assert "${escapeHtml(preview)}" in viewer
    assert "${escapeHtml(item.message)}" in viewer
    assert "progressContainer.querySelector('#evolution-name').textContent" in viewer
    assert 'data-history-action="rename"' in viewer
    assert "title.textContent = originalName" in viewer
    assert "onclick=\"selectHistoryItem" not in viewer
    assert "onclick=\"event.stopPropagation(); handleHistory" not in viewer

    assert "modalContent.querySelector('.idea-details-title').textContent" in rater
    assert "querySelector('[data-idea-detail=\"auto-rating\"]').textContent" in rater
    assert "querySelector('[data-idea-detail=\"manual-rating\"]').textContent" in rater
    assert "Error: ${escapeHtml(error.message)}" in rater


@pytest.mark.parametrize(
    "script_path,next_function",
    [(VIEWER_JS, "getTournamentEntryForGeneration"), (RATER_JS, "decodeMarkdownUrlForPolicy")],
)
@pytest.mark.parametrize(
    "generated_text",
    [
        '<img src=x onerror="alert(1)">',
        '<svg onload="alert(2)"></svg>',
        '</h4><script>alert(3)</script>',
    ],
    ids=["title", "oracle-analysis", "rating-text"],
)
def test_escape_html_contract_for_generated_title_oracle_and_rating_text(
    script_path, next_function, generated_text
):
    script = script_path.read_text()
    escape_bundle = _extract_function_bundle(script, "escapeHtml", next_function)
    node_script = f"""
{escape_bundle}
process.stdout.write(escapeHtml({json.dumps(generated_text)}));
"""
    result = subprocess.run(
        ["node", "-e", node_script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "<" not in result.stdout
    assert ">" not in result.stdout
    assert "&lt;" in result.stdout
