import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).parent.parent
VIEWER_JS = ROOT / "idea" / "static" / "js" / "viewer.js"


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
