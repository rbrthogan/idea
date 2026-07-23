import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).parent.parent
VIEWER_HTML = ROOT / "idea" / "static" / "html" / "viewer.html"
RATER_HTML = ROOT / "idea" / "static" / "html" / "rater.html"
ADMIN_HTML = ROOT / "idea" / "static" / "html" / "admin.html"
AUTH_JS = ROOT / "idea" / "static" / "js" / "auth_logic.js"
VIEWER_PY = ROOT / "idea" / "viewer.py"


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


def test_api_error_responses_do_not_serialize_server_tracebacks():
    source = VIEWER_PY.read_text()

    assert '"traceback": traceback.format_exc()' not in source
    assert '"details": error_details' not in source


def test_settings_status_never_returns_the_decrypted_api_key():
    source = VIEWER_PY.read_text()
    endpoint = _bundle_between(
        source,
        'async def get_settings_status',
        '@app.post("/api/contact")',
    )

    assert '"masked_key": masked_key' in endpoint
    assert '"api_key": api_key' not in endpoint
    assert "fullApiKey" not in (ROOT / "idea" / "static" / "js" / "viewer.js").read_text()
    assert 'id="copyApiKeyBtn"' not in VIEWER_HTML.read_text()


def test_auth_url_policies_block_active_images_and_cross_origin_token_leaks():
    source = AUTH_JS.read_text()
    bundle = _bundle_between(
        source,
        "function getSafeUserImageUrl",
        "// 3. Create Login Modal",
    )
    node_script = f"""
global.window = {{ location: {{ origin: 'https://app.example' }} }};
{bundle}
const imageInputs = [
  '/avatar.png',
  'https://images.example/avatar.png',
  'http://images.example/avatar.png',
  'javascript:alert(1)',
  'data:image/svg+xml,<svg onload=alert(1)>',
  'vbscript:msgbox(1)',
  'blob:https://app.example/id',
  '//evil.example/avatar.png'
];
const requestInputs = [
  '/api/evolution',
  'https://app.example/api/evolution',
  'https://evil.example/?next=https://app.example',
  'https://app.example.evil.example/api',
  'https://app.example@evil.example/api',
  '//evil.example/api',
  'javascript:alert(1)'
];
process.stdout.write(JSON.stringify({{
  images: imageInputs.map(value => getSafeUserImageUrl(value)),
  internal: requestInputs.map(value => isInternalRequestUrl(value))
}}));
"""
    result = json.loads(_node_eval(node_script))

    assert result["images"][:2] == [
        "https://app.example/avatar.png",
        "https://images.example/avatar.png",
    ]
    assert result["images"][2:] == ["/static/img/logo.png"] * 6
    assert result["internal"] == [True, True, False, False, False, False, False]


def test_auth_user_fields_are_assigned_through_dom_properties():
    source = AUTH_JS.read_text()

    assert 'src="${photoURL}"' not in source
    assert "${name}" not in source
    assert "${user.email}" not in source
    assert "querySelector('.user-avatar').src = photoURL" in source
    assert "querySelector('.user-display-name').textContent = name" in source
    assert "querySelector('.user-email').textContent = email" in source
    assert "const isInternal = isInternalRequestUrl(url)" in source
    assert ".includes(window.location.origin)" not in source


def test_viewer_generated_template_and_message_sinks_use_text_nodes():
    source = VIEWER_HTML.read_text()

    assert "onclick=\"event.stopPropagation(); editTemplateMainPage('${id}')\"" not in source
    assert "row.querySelector('.template-name').textContent = templateName" in source
    assert "descriptionElement.textContent = descriptionForDisplay" in source
    assert "descriptionElement.title = description" in source
    assert "content.querySelector('.template-preview-name').textContent" in source
    assert "content.querySelector('.template-preview-description').textContent" in source
    assert "content.querySelector('.template-preview-item-type').textContent" in source
    assert "content.querySelector('.template-preview-criteria').textContent" in source
    assert "textSpan.textContent = message" in source
    assert "document.createTextNode(String(message ?? ''))" in source
    assert "insertAdjacentHTML" not in source
    assert "input.value = String(value ?? '')" in source
    assert 'value="${value}"' not in source
    assert "encodeURIComponent(templateId)" in source
    assert "encodeURIComponent(currentEditingTemplateId)" in source


def test_viewer_alert_classes_are_allow_listed_in_node():
    source = VIEWER_HTML.read_text()
    bundle = _bundle_between(
        source,
        "function normalizeTemplateAlertType",
        "function showStatus",
    )
    node_script = f"""
{bundle}
const values = ['success', 'error', 'warning', 'danger', 'info', 'danger onclick=alert(1)', ''];
process.stdout.write(JSON.stringify(values.map(value => normalizeTemplateAlertType(value))));
"""

    assert json.loads(_node_eval(node_script)) == [
        "success",
        "danger",
        "warning",
        "danger",
        "info",
        "info",
        "info",
    ]


def test_viewer_preview_normalizer_handles_malicious_generated_values_in_node():
    source = VIEWER_HTML.read_text()
    bundle = _bundle_between(
        source,
        "function getTemplatePreviewDisplay",
        "function showTemplatePreview",
    )
    payload = {
        "name": '<img src=x onerror="alert(1)">',
        "description": "</p><svg onload=alert(2)>",
        "metadata": {"item_type": "</small><script>alert(3)</script>"},
        "comparison_criteria": ["safe", "<iframe srcdoc=x>", "third"],
    }
    node_script = f"""
{bundle}
process.stdout.write(JSON.stringify(getTemplatePreviewDisplay({json.dumps(payload)})));
"""
    result = json.loads(_node_eval(node_script))

    assert result["name"] == payload["name"]
    assert result["description"] == payload["description"]
    assert result["itemType"] == payload["metadata"]["item_type"]
    assert result["criteria"] == "safe, <iframe srcdoc=x>..."


def test_viewer_preview_mounts_malicious_generated_values_as_text_in_node():
    source = VIEWER_HTML.read_text()
    bundle = _bundle_between(
        source,
        "function getTemplatePreviewDisplay",
        "function hideTemplatePreview",
    )
    payload = {
        "name": '<img src=x onerror="alert(1)">',
        "description": "</p><svg onload=alert(2)>",
        "metadata": {"item_type": "</small><script>alert(3)</script>"},
        "comparison_criteria": ["<iframe srcdoc=x>", "safe"],
    }
    node_script = f"""
const fields = Object.fromEntries([
  '.template-preview-name',
  '.template-preview-description',
  '.template-preview-item-type',
  '.template-preview-criteria'
].map(key => [key, {{ textContent: '' }}]));
const content = {{
  innerHTML: '',
  querySelector(selector) {{ return fields[selector]; }}
}};
const preview = {{ style: {{ display: 'none' }} }};
global.document = {{
  getElementById(id) {{ return id === 'templatePreview' ? preview : content; }}
}};
{bundle}
showTemplatePreview({json.dumps(payload)});
process.stdout.write(JSON.stringify({{
  shell: content.innerHTML,
  texts: Object.fromEntries(Object.entries(fields).map(([key, value]) => [key, value.textContent])),
  display: preview.style.display
}}));
"""
    result = json.loads(_node_eval(node_script))

    assert payload["name"] not in result["shell"]
    assert payload["description"] not in result["shell"]
    assert result["texts"][".template-preview-name"] == payload["name"]
    assert result["texts"][".template-preview-description"] == payload["description"]
    assert result["texts"][".template-preview-item-type"] == payload["metadata"]["item_type"]
    assert result["display"] == "block"


def test_admin_user_rows_use_text_content_for_all_server_fields():
    source = ADMIN_HTML.read_text()

    assert "tbody.innerHTML = data.users.map" not in source
    assert "tbody.replaceChildren(...data.users.map(createAdminUserRow))" in source
    assert "userIdCode.textContent = display.userId" in source
    assert "evolutionBadge.textContent = display.evolutionCount" in source
    assert "templateBadge.textContent = display.templateCount" in source
    assert "activityCell.textContent = display.lastActivity" in source


def test_admin_user_display_normalizer_accepts_malicious_text_without_markup_execution():
    source = ADMIN_HTML.read_text()
    bundle = _bundle_between(
        source,
        "function normalizeAdminUserDisplay",
        "function createAdminUserRow",
    )
    payload = {
        "user_id": '<img/onerror="alert(1)">',
        "evolution_count": "</span><script>alert(2)</script>",
        "template_count": "<svg onload=alert(3)>",
        "has_api_key": True,
        "last_activity": "not-a-date",
    }
    node_script = f"""
{bundle}
process.stdout.write(JSON.stringify(normalizeAdminUserDisplay({json.dumps(payload)})));
"""
    result = json.loads(_node_eval(node_script))

    assert result == {
        "userId": "<img/onerror...",
        "evolutionCount": payload["evolution_count"],
        "templateCount": payload["template_count"],
        "hasApiKey": True,
        "lastActivity": "--",
    }


def test_admin_user_row_creates_only_fixed_elements_for_malicious_server_text():
    source = ADMIN_HTML.read_text()
    bundle = _bundle_between(
        source,
        "function normalizeAdminUserDisplay",
        "// Initialize when auth is ready",
    )
    payload = {
        "user_id": '<img/onerror="alert(1)">',
        "evolution_count": "</span><script>alert(2)</script>",
        "template_count": "<svg onload=alert(3)>",
        "has_api_key": True,
        "last_activity": "not-a-date",
    }
    node_script = f"""
class FakeElement {{
  constructor(tagName) {{
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.className = '';
    this.textContent = '';
  }}
  append(...children) {{ this.children.push(...children); }}
  appendChild(child) {{ this.children.push(child); return child; }}
}}
global.document = {{
  createElement(tagName) {{ return new FakeElement(tagName); }},
  createTextNode(value) {{ return {{ tagName: '#text', children: [], textContent: String(value) }}; }}
}};
{bundle}
const row = createAdminUserRow({json.dumps(payload)});
const nodes = [];
function visit(node) {{
  nodes.push({{ tagName: node.tagName, textContent: node.textContent }});
  (node.children || []).forEach(visit);
}}
visit(row);
process.stdout.write(JSON.stringify(nodes));
"""
    nodes = json.loads(_node_eval(node_script))
    tags = {node["tagName"] for node in nodes}
    texts = [node["textContent"] for node in nodes]

    assert not {"IMG", "SCRIPT", "SVG", "IFRAME"} & tags
    assert "<img/onerror..." in texts
    assert payload["evolution_count"] in texts
    assert payload["template_count"] in texts


def test_inline_scripts_remain_valid_javascript():
    for path in (VIEWER_HTML, RATER_HTML, ADMIN_HTML):
        source = path.read_text()
        scripts = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)</script>", source)
        javascript = "\n".join(re.sub(r"\{\{[\s\S]*?\}\}", "null", script) for script in scripts)
        subprocess.run(
            ["node", "--check"],
            input=javascript,
            check=True,
            capture_output=True,
            text=True,
        )
