import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).parent.parent
VIEWER_HTML = ROOT / "idea" / "static" / "html" / "viewer.html"
RATER_HTML = ROOT / "idea" / "static" / "html" / "rater.html"
VIEWER_JS = ROOT / "idea" / "static" / "js" / "viewer.js"
RATER_JS = ROOT / "idea" / "static" / "js" / "rater.js"


def _node_eval(script: str) -> str:
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _bundle_between(source: str, start: str, end: str) -> str:
    start_index = source.index(start)
    return source[start_index : source.index(end, start_index)]


def test_core_view_switcher_has_roving_tabs_and_named_panels():
    html = VIEWER_HTML.read_text()
    javascript = VIEWER_JS.read_text()

    expected_tabs = {
        "cards": "generations-container",
        "tree": "evolution-tree-container",
        "ranked": "evolution-ranked-container",
    }
    for mode, panel_id in expected_tabs.items():
        tab_tag = re.search(
            rf'<button[^>]*id="evolution-view-tab-{mode}"[^>]*>', html
        )
        assert tab_tag, mode
        assert 'role="tab"' in tab_tag.group(0)
        assert f'aria-controls="{panel_id}"' in tab_tag.group(0)

        panel_tag = re.search(rf'<div[^>]*id="{panel_id}"[^>]*>', html)
        assert panel_tag, panel_id
        assert 'role="tabpanel"' in panel_tag.group(0)
        assert f'aria-labelledby="evolution-view-tab-{mode}"' in panel_tag.group(0)

    assert html.count('class="evolution-view-toggle-btn') == 3
    assert html.count('role="tab"') >= 3
    assert html.count('tabindex="0"') >= 1
    assert html.count('tabindex="-1"') >= 2
    assert "button.addEventListener('keydown', handleEvolutionViewTabKeydown)" in javascript
    assert "button.tabIndex = isActive ? 0 : -1" in javascript
    assert "targetTab.focus()" in javascript


def test_roving_tab_index_keyboard_contract_in_node():
    source = VIEWER_JS.read_text()
    bundle = _bundle_between(
        source,
        "function getRovingTabIndex",
        "function handleEvolutionViewTabKeydown",
    )
    node_script = f"""
{bundle}
const inputs = [
  [0, 'ArrowRight', 3],
  [2, 'ArrowRight', 3],
  [0, 'ArrowLeft', 3],
  [1, 'ArrowDown', 3],
  [1, 'ArrowUp', 3],
  [2, 'Home', 3],
  [0, 'End', 3],
  [1, 'Escape', 3],
  [-1, 'ArrowRight', 3],
  [0, 'ArrowRight', 0]
];
process.stdout.write(JSON.stringify(inputs.map(args => getRovingTabIndex(...args))));
"""

    assert json.loads(_node_eval(node_script)) == [1, 0, 2, 2, 0, 0, 2, None, None, None]


def test_core_dynamic_selection_controls_are_native_buttons():
    viewer_html = VIEWER_HTML.read_text()
    viewer_js = VIEWER_JS.read_text()
    rater_js = RATER_JS.read_text()

    assert '<span class="example-link"' not in viewer_html
    assert viewer_html.count('<button type="button" class="example-link"') == 5
    assert '<button type="button" class="template-list-select"' in viewer_js
    assert '<button type="button" class="history-item-select"' in viewer_js
    assert "container.querySelectorAll('.history-item-select')" in viewer_js
    assert "document.createElement('button')" in _bundle_between(
        viewer_js,
        "function createEvolutionTreeNode",
        "function createEvolutionTreeContinuation",
    )
    assert '<button type="button" class="evolution-rank-row' in viewer_js

    ranking_bundle = _bundle_between(
        rater_js,
        "function updateRankingsTableContent",
        "function getIdeaRawDiversityScore",
    )
    assert "row.addEventListener('click'" not in ranking_bundle
    assert "row.querySelector('.idea-details-trigger')" in ranking_bundle
    assert '<button type="button" class="idea-details-trigger">' in rater_js


def test_rating_sort_headers_are_buttons_with_programmatic_direction():
    html = RATER_HTML.read_text()

    assert '<caption class="visually-hidden">Idea rankings.' in html
    assert '<th class="sortable-header"' not in html
    assert html.count('<button type="button" class="sortable-header"') == 8
    assert html.count('scope="col" aria-sort="none"') == 8
    assert html.count('class="sort-arrow inactive" aria-hidden="true"') == 8


def test_sort_indicator_updates_aria_sort_in_node():
    source = RATER_JS.read_text()
    bundle = _bundle_between(
        source,
        "function updateSortIndicators",
        "function sortTable",
    )
    node_script = f"""
let currentSortColumn = 'fitness';
let currentSortDirection = 'desc';
function makeHeader(column) {{
  const th = {{ attributes: {{}}, setAttribute(name, value) {{ this.attributes[name] = value; }} }};
  const arrow = {{ className: '' }};
  return {{
    dataset: {{ sort: column }},
    querySelector() {{ return arrow; }},
    closest() {{ return th; }},
    th,
    arrow
  }};
}}
const headers = ['rank', 'fitness', 'title'].map(makeHeader);
global.document = {{ querySelectorAll() {{ return headers; }} }};
{bundle}
updateSortIndicators();
process.stdout.write(JSON.stringify(headers.map(header => ({{
  sort: header.th.attributes['aria-sort'],
  arrow: header.arrow.className
}}))));
"""

    assert json.loads(_node_eval(node_script)) == [
        {"sort": "none", "arrow": "sort-arrow inactive"},
        {"sort": "descending", "arrow": "sort-arrow active desc"},
        {"sort": "none", "arrow": "sort-arrow inactive"},
    ]


def test_progress_and_pairing_updates_have_bounded_live_regions():
    viewer_html = VIEWER_HTML.read_text()
    viewer_js = VIEWER_JS.read_text()
    rater_html = RATER_HTML.read_text()

    assert 'id="statusText" role="status" aria-live="polite" aria-atomic="true"' in viewer_html
    assert 'id="evolution-status-text" role="status" aria-live="polite" aria-atomic="true"' in viewer_js
    assert 'aria-label="Evolution progress" aria-valuetext="0% complete"' in viewer_js
    assert 'id="progress-percentage" aria-hidden="true"' in viewer_js
    assert 'id="evolution-activity-log"' in viewer_js
    assert 'id="evolution-activity-log"' in viewer_js and 'id="evolution-activity-log" aria-live=' not in viewer_js

    assert 'id="manualSwissStatus"' in rater_html
    assert 'role="status" aria-live="polite"' in rater_html
    assert 'aria-label="Automated rating progress"' in rater_html
    assert 'id="ratingStatusAnnouncement" class="visually-hidden" role="status"' in rater_html


def test_rating_progress_announces_deciles_not_every_poll_in_node():
    source = RATER_JS.read_text()
    announcer_bundle = _bundle_between(
        source,
        "function announceRatingStatus",
        "function normalizeGenerationValue",
    )
    progress_bundle = _bundle_between(
        source,
        "function updateProgressBar",
        "function stopAutoRatingProgressPolling",
    )
    node_script = f"""
const writes = [];
let announcementText = '';
const announcement = {{}};
Object.defineProperty(announcement, 'textContent', {{
  get() {{ return announcementText; }},
  set(value) {{ announcementText = value; writes.push(value); }}
}});
const attributes = {{ 'aria-valuenow': '0' }};
const progress = {{
  style: {{}},
  dataset: {{}},
  offsetHeight: 1,
  getAttribute(name) {{ return attributes[name] || ''; }},
  setAttribute(name, value) {{ attributes[name] = String(value); }}
}};
global.document = {{
  getElementById(id) {{
    if (id === 'ratingProgress') return progress;
    if (id === 'ratingStatusAnnouncement') return announcement;
    return null;
  }}
}};
console.log = () => {{}};
console.error = () => {{}};
{announcer_bundle}
{progress_bundle}
[1, 8, 10, 19, 20, 100].forEach(updateProgressBar);
process.stdout.write(JSON.stringify({{
  writes,
  valueNow: attributes['aria-valuenow'],
  valueText: attributes['aria-valuetext']
}}));
"""
    result = json.loads(_node_eval(node_script))

    assert result == {
        "writes": [
            "Automated rating 0% complete",
            "Automated rating 10% complete",
            "Automated rating 20% complete",
            "Automated rating 100% complete",
        ],
        "valueNow": "100",
        "valueText": "100% complete",
    }


def test_core_dialogs_have_programmatic_names_and_named_close_buttons():
    html = VIEWER_HTML.read_text()
    rater_html = RATER_HTML.read_text()
    viewer_js = VIEWER_JS.read_text()

    modal_labels = {
        "contactModal": "contactModalLabel",
        "deleteKeyModal": "deleteKeyModalLabel",
        "saveDataModal": "saveDataModalLabel",
        "ideaModal": "ideaModalLabel",
        "welcomeModal": "welcomeModalLabel",
        "templateEditModal": "templateEditModalTitle",
    }
    for modal_id, label_id in modal_labels.items():
        modal_tag = re.search(rf'<div[^>]*id="{modal_id}"[^>]*>', html)
        assert modal_tag, modal_id
        assert f'aria-labelledby="{label_id}"' in modal_tag.group(0)
        assert f'id="{label_id}"' in html

    assert "modal.setAttribute('aria-labelledby', 'alreadyRunningModalLabel')" in viewer_js
    assert 'id="alreadyRunningModalLabel"' in viewer_js
    assert 'id="alreadyRunningModalDescription"' in viewer_js
    assert '<button type="button" class="btn-close" data-bs-dismiss="modal"></button>' not in html
    assert 'aria-label="Copy raw idea markdown"' in html
    assert 'aria-label="Copy raw idea markdown"' in rater_html


def test_programmatic_dialog_restores_the_latest_focus_origin_in_node():
    source = VIEWER_JS.read_text()
    bundle = _bundle_between(
        source,
        "function rememberModalFocusOrigin",
        "// Update the showIdeaModal",
    )
    node_script = f"""
const body = {{}};
const focusCalls = [];
const first = {{ isConnected: true, disabled: false, focus(options) {{ focusCalls.push(['first', options]); }} }};
const second = {{ isConnected: true, disabled: false, focus(options) {{ focusCalls.push(['second', options]); }} }};
const listeners = [];
const modal = {{
  dataset: {{}},
  addEventListener(name, listener) {{ listeners.push([name, listener]); }}
}};
global.document = {{ body, activeElement: first }};
{bundle}
rememberModalFocusOrigin(modal);
document.activeElement = second;
rememberModalFocusOrigin(modal);
listeners[0][1]();
process.stdout.write(JSON.stringify({{
  listenerNames: listeners.map(([name]) => name),
  focusCalls
}}));
"""

    assert json.loads(_node_eval(node_script)) == {
        "listenerNames": ["hidden.bs.modal"],
        "focusCalls": [["second", {"preventScroll": True}]],
    }


def test_changed_browser_javascript_remains_syntax_valid():
    for path in (VIEWER_JS, RATER_JS):
        subprocess.run(
            ["node", "--check", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
