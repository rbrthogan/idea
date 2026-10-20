"""Behavioral regression checks for shared settings request and save recovery."""
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "idea/static/js/workbench.js"


def run_shared(body, *, mobile=False):
    script = """
global.window = { currentUser: { uid: 'first-user' }, matchMedia: () => ({ matches: MOBILE, addEventListener() {} }) };
const readyCallbacks = [];
const nodes = {};
global.document = { addEventListener(type, listener) { if (type === 'DOMContentLoaded') readyCallbacks.push(listener); }, getElementById(id) { return nodes[id] || null; } };
const makeNode = (value = '') => ({ value, textContent: '', hidden: false, disabled: false, style: { setProperty(name, value) { this[name] = value; } }, dataset: {}, attributes: {}, listeners: {},
    setAttribute(name, value) { this.attributes[name] = value; }, removeAttribute(name) { delete this.attributes[name]; },
    addEventListener(type, listener) { this.listeners[type] = listener; }, focus() { this.focused = true; }
});
""".replace('MOBILE', json.dumps(mobile)) + SHARED.read_text() + "\n(async () => {\n" + body + "\n})().catch(error => { console.error(error); process.exit(1); });"
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


SIDEBAR_FIXTURE = """
const classes = (initial = []) => {
    const values = new Set(initial);
    return { contains: name => values.has(name), add: name => values.add(name), remove: name => values.delete(name),
        toggle(name, force) { const on = force ?? !values.has(name); if (on) values.add(name); else values.delete(name); } };
};
const element = (id, initial = []) => ({ ...makeNode(), id, classList: classes(initial), isConnected: true, offsetParent: {},
    hasAttribute(name) { return name in this.attributes; },
    toggleAttribute(name, on) { if (on) this.attributes[name] = ''; else delete this.attributes[name]; },
    querySelector() { return null; }, contains(node) { return this === node; },
    focus() { document.activeElement = this; }, scrollIntoView() { this.scrolled = true; }
});
['controlsPanel', 'sidebarToggle', 'sidebarBackdrop', 'setupApiKeyBtn', 'apiKeyInput', 'apiKeyProviderSelect', 'providerSelect', 'replaceApiKeyBtn'].forEach(id => nodes[id] = element(id));
nodes.providerSelect.value = 'openai';
nodes['view-main'] = element('view-main', ['active']);
nodes['view-settings'] = element('view-settings');
const content = element('content');
document.body = { dataset: {}, classList: classes() };
document.querySelector = selector => selector === '.sidebar-view.active' ? [nodes['view-main'], nodes['view-settings']].find(node => node.classList.contains('active')) : null;
document.querySelectorAll = selector => selector === '.sidebar-view' ? [nodes['view-main'], nodes['view-settings']] : selector === '.main-content, .evolution-content' ? [content] : [];
global.localStorage = { getItem() { return 'true'; }, setItem() {} };
readyCallbacks[0]();
document.activeElement = nodes.setupApiKeyBtn;
let finishStatus;
window.showSidebarView = name => {
    window.IdeaWorkbench.selectSidebarView(name);
    return new Promise(resolve => finishStatus = resolve);
};
"""


@pytest.mark.parametrize('mobile', [True, False])
def test_banner_reveals_closed_settings_drawer_and_focuses_selected_provider(mobile):
    viewer = (ROOT / 'idea/static/js/viewer.js').read_text()
    start = viewer.index("    const setupBtn = document.getElementById('setupApiKeyBtn');")
    banner_binding = viewer[start:viewer.index('    // Nav buttons', start)]
    result = run_shared(SIDEBAR_FIXTURE + banner_binding + """
const initiallyInert = nodes.controlsPanel.hasAttribute('inert');
nodes['view-settings'].scrollTop = 500;
nodes.setupApiKeyBtn.listeners.click();
const opened = { inert: nodes.controlsPanel.hasAttribute('inert'), expanded: nodes.sidebarToggle.attributes['aria-expanded'], show: nodes.controlsPanel.classList.contains('show'), collapsed: nodes.controlsPanel.classList.contains('collapsed'), settingsHidden: nodes['view-settings'].hidden, scrollTop: nodes['view-settings'].scrollTop, provider: nodes.apiKeyProviderSelect.value };
finishStatus(); await new Promise(resolve => setImmediate(resolve));
const focused = document.activeElement.id;
const scrolled = nodes.apiKeyInput.scrolled;
window.IdeaWorkbench.closeSidebar();
process.stdout.write(JSON.stringify({ initiallyInert, opened, focused, scrolled, restored: document.activeElement.id }));
""", mobile=mobile)
    assert result['initiallyInert'] is True
    assert result['opened'] == {
        'inert': False, 'expanded': 'true', 'show': mobile, 'collapsed': False,
        'settingsHidden': False, 'scrollTop': 0, 'provider': 'openai',
    }
    assert result['focused'] == 'apiKeyInput'
    assert result['scrolled'] is True
    assert result['restored'] == 'setupApiKeyBtn'


@pytest.mark.parametrize('departure', ['close', 'navigate', 'interact', 'account'])
def test_late_settings_status_never_steals_focus_after_user_leaves(departure):
    result = run_shared(SIDEBAR_FIXTURE + """
window.IdeaWorkbench.openProviderSettings();
""" + {
        'close': "window.IdeaWorkbench.closeSidebar();",
        'navigate': "window.IdeaWorkbench.selectSidebarView('main');",
        'interact': "nodes.apiKeyProviderSelect.focus();",
        'account': "window.currentUser = {uid:'other-user'}; nodes.setupApiKeyBtn.focus();",
    }[departure] + """
const before = document.activeElement.id;
finishStatus(); await new Promise(resolve => setImmediate(resolve));
process.stdout.write(JSON.stringify({before, after: document.activeElement.id, scrolled: !!nodes.apiKeyInput.scrolled}));
""", mobile=True)
    assert result['after'] == result['before']
    assert result['scrolled'] is False


def test_connected_provider_focuses_replace_action_and_replacement_input():
    result = run_shared(SIDEBAR_FIXTURE + """
nodes.apiKeyInput.offsetParent = null;
['apiKeyInputContainer', 'apiKeyConfiguredContainer', 'cancelApiKeyEdit'].forEach(id => nodes[id] = element(id));
window.IdeaWorkbench.openProviderSettings('google_generative_ai');
finishStatus(); await new Promise(resolve => setImmediate(resolve));
const action = document.activeElement.id;
window.IdeaWorkbench.editProviderKey();
process.stdout.write(JSON.stringify({action, focused: document.activeElement.id, input: nodes.apiKeyInputContainer.style.display, provider: nodes.apiKeyProviderSelect.value, inert: nodes.controlsPanel.hasAttribute('inert')}));
""", mobile=True)
    assert result == {
        'action': 'replaceApiKeyBtn', 'focused': 'apiKeyInput', 'input': 'block',
        'provider': 'google_generative_ai', 'inert': False,
    }


def test_provider_status_ignores_previous_accounts_inflight_response():
    result = run_shared("""
const requests = [];
global.fetch = () => new Promise(resolve => requests.push(resolve));
const app = window.IdeaWorkbench;
const old = app.loadProviderSettings().catch(error => error.message);
window.currentUser = { uid: 'second-user' };
const fresh = app.loadProviderSettings();
requests[1]({ ok: true, json: async () => ({ api_keys: { openai: { configured: false } } }) });
await fresh;
requests[0]({ ok: true, json: async () => ({ api_keys: { openai: { configured: true } } }) });
const oldError = await old;
process.stdout.write(JSON.stringify({ oldError, snapshot: app.providerSettingsSnapshot(), requests: requests.length }));
""")
    assert result["requests"] == 2
    assert "session changed" in result["oldError"]
    assert result["snapshot"]["api_keys"]["openai"]["configured"] is False


def test_failed_provider_status_explains_failure_and_recovers_on_retry():
    result = run_shared("""
nodes.providerSelect = makeNode('openai');
nodes.selectedProviderStatus = makeNode();
nodes.configureSelectedProvider = makeNode();
global.fetch = async () => { throw new Error('Offline'); };
try { await window.IdeaWorkbench.loadProviderSettings(); } catch (_) {}
const failed = { message: nodes.selectedProviderStatus.textContent, state: nodes.selectedProviderStatus.dataset.state, action: nodes.configureSelectedProvider.textContent, hidden: nodes.configureSelectedProvider.hidden };
global.fetch = async () => ({ ok: true, json: async () => ({ api_keys: { openai: { configured: true } } }) });
await window.IdeaWorkbench.loadProviderSettings();
process.stdout.write(JSON.stringify({ failed, recovered: nodes.selectedProviderStatus.textContent, actionHidden: nodes.configureSelectedProvider.hidden }));
""")
    assert result["failed"] == {
        "message": "Could not check provider key",
        "state": "error",
        "action": "Check settings",
        "hidden": False,
    }
    assert result["recovered"] == "OpenAI key ready"
    assert result["actionHidden"] is True


def test_superseded_refresh_cannot_overwrite_newer_provider_status():
    result = run_shared("""
const requests = [];
global.fetch = () => new Promise(resolve => requests.push(resolve));
const app = window.IdeaWorkbench;
const first = app.loadProviderSettings().catch(error => error.message);
const second = app.loadProviderSettings(true);
requests[1]({ ok: true, json: async () => ({ api_keys: { openai: { configured: true } } }) });
await second;
requests[0]({ ok: true, json: async () => ({ api_keys: { openai: { configured: false } } }) });
await first;
process.stdout.write(JSON.stringify(app.providerSettingsSnapshot()));
""")
    assert result["api_keys"]["openai"]["configured"] is True


def test_shared_key_save_locks_provider_then_reports_saved_but_refresh_failed():
    result = run_shared("""
['apiKeyInput', 'saveSettingsBtn', 'apiKeyProviderSelect', 'cancelApiKeyEdit', 'settingsStatus'].forEach(id => nodes[id] = makeNode());
nodes.apiKeyInput.value = 'test-placeholder';
nodes.apiKeyProviderSelect.value = 'openai';
nodes.saveSettingsBtn.textContent = 'Save key';
document.body = { dataset: { sharedKeySettings: 'true' }, classList: { toggle() {} } };
document.querySelector = () => null;
document.querySelectorAll = () => [];
readyCallbacks[0]();
let resolveSave;
let count = 0;
global.fetch = async () => { if (++count === 1) return new Promise(resolve => resolveSave = resolve); throw new Error('Offline'); };
const saving = nodes.saveSettingsBtn.listeners.click();
const pending = { disabled: nodes.saveSettingsBtn.disabled, providerLocked: nodes.apiKeyProviderSelect.disabled, inputLocked: nodes.apiKeyInput.disabled, label: nodes.saveSettingsBtn.textContent };
resolveSave({ ok: true, json: async () => ({ status: 'success' }) });
await saving;
process.stdout.write(JSON.stringify({ pending, value: nodes.apiKeyInput.value, disabled: nodes.saveSettingsBtn.disabled, providerLocked: nodes.apiKeyProviderSelect.disabled, status: nodes.settingsStatus.textContent, type: nodes.settingsStatus.className, hidden: nodes.settingsStatus.hidden, display: nodes.settingsStatus.style.display }));
""")
    assert result["pending"] == {"disabled": True, "providerLocked": True, "inputLocked": True, "label": "Saving…"}
    assert result["value"] == ""
    assert result["disabled"] is False
    assert result["providerLocked"] is False
    assert "was saved" in result["status"]
    assert "could not be refreshed" in result["status"]
    assert "warning" in result["type"]
    assert result["hidden"] is False
    assert result["display"] == "block"


@pytest.mark.parametrize('flow', ['viewer', 'shared'])
@pytest.mark.parametrize('provider', ['google_generative_ai', 'openai'])
@pytest.mark.parametrize('outcome', ['ready', 'unreadable', 'missing', 'unknown', 'failed'])
def test_key_save_only_reports_success_after_fresh_selected_provider_confirmation(flow, provider, outcome):
    viewer = (ROOT / 'idea/static/js/viewer.js').read_text()
    start = viewer.index('let settingsStatusVersion')
    viewer_flow = viewer[start:viewer.index('/**\n * Delete API Key', start)]
    result = run_shared(
        'const flow=' + json.dumps(flow) + ', provider=' + json.dumps(provider)
        + ', outcome=' + json.dumps(outcome) + ';\n' + """
['view-settings', 'apiKeyInput', 'saveSettingsBtn', 'apiKeyProviderSelect', 'cancelApiKeyEdit',
 'settingsStatus', 'apiKeyInputContainer', 'apiKeyConfiguredContainer'].forEach(id => nodes[id] = makeNode());
nodes.apiKeyInput.value = 'synthetic-key';
nodes.apiKeyProviderSelect.value = provider;
nodes.saveSettingsBtn.textContent = nodes.saveSettingsBtn.innerHTML = 'Save key';
document.body = { dataset: { sharedKeySettings: 'true' }, classList: { toggle() {} } };
document.querySelector = () => null;
document.querySelectorAll = () => [];
const other = provider === 'openai' ? 'google_generative_ai' : 'openai';
// A prior successful check or a ready other provider cannot establish this save.
window.IdeaWorkbench.setProviderSettings({api_keys:{[provider]:{configured:true}}});
const status = {api_keys:{[other]:{configured:true}}};
if (outcome !== 'unknown') status.api_keys[provider] = {
 configured: outcome === 'ready',
 error_code: outcome === 'unreadable' ? 'stored_key_unreadable' : undefined,
 message: outcome === 'unreadable' ? 'Saved key cannot be opened. Check server configuration.' : undefined
};
const calls = [];
global.fetch = async (url, options) => {
 calls.push({url, method:options?.method || 'GET', body:options?.body && JSON.parse(options.body)});
 if (options?.method === 'POST') return {ok:true,json:async()=>({status:'success'})};
 if (outcome === 'failed') throw new Error('Offline');
 return {ok:true,json:async()=>status};
};
function showStatusInSettings(message, type) {
 nodes.settingsStatus.textContent = message;
 nodes.settingsStatus.className = 'alert-' + type;
 nodes.settingsStatus.hidden = false;
}
const loadExecutionSettings = force => window.IdeaWorkbench.loadProviderSettings(force);
const apiErrorMessage = (data, fallback) => data.message || fallback;
""" + viewer_flow + """
if (flow === 'shared') { readyCallbacks[0](); await nodes.saveSettingsBtn.listeners.click(); }
else await saveSettings();
process.stdout.write(JSON.stringify({
 calls, message:nodes.settingsStatus.textContent, type:nodes.settingsStatus.className,
 hidden:nodes.settingsStatus.hidden, value:nodes.apiKeyInput.value,
 input:nodes.apiKeyInputContainer.style.display, configured:nodes.apiKeyConfiguredContainer.style.display
}));
"""
    )
    assert [call['method'] for call in result['calls']] == ['POST', 'GET']
    assert result['calls'][0]['body'] == {'provider': provider, 'api_key': 'synthetic-key'}
    assert result['hidden'] is False and result['value'] == ''
    assert ('success' in result['type']) is (outcome == 'ready')
    if outcome == 'ready':
        assert 'API key saved.' in result['message']
        assert result['configured'] == 'block'
    else:
        assert 'warning' in result['type']
        if outcome == 'unreadable':
            assert result['message'] == 'Saved key cannot be opened. Check server configuration.'
        elif outcome == 'failed':
            assert 'could not be refreshed' in result['message']
        else:
            assert 'no usable' in result['message']


def test_sidebar_back_restores_the_control_that_opened_settings():
    result = run_shared("""
let current = 'view-main';
const opener = makeNode(); opener.isConnected = true;
const classes = () => ({ toggle() {}, remove() {} });
['main', 'advanced'].forEach(name => nodes[`view-${name}`] = { ...makeNode(), id: `view-${name}`, classList: classes(), contains: node => name === 'main' && node === opener, toggleAttribute() {} });
nodes.controlsPanel = { classList: classes() };
document.querySelector = () => nodes[current];
document.querySelectorAll = () => [nodes['view-main'], nodes['view-advanced']];
document.activeElement = opener;
window.IdeaWorkbench.selectSidebarView('advanced');
current = 'view-advanced';
document.activeElement = nodes['view-advanced'];
window.IdeaWorkbench.selectSidebarView('main');
process.stdout.write(JSON.stringify({ restored: opener.focused, advancedHidden: nodes['view-advanced'].hidden, mainHidden: nodes['view-main'].hidden }));
""")
    assert result == {"restored": True, "advancedHidden": True, "mainHidden": False}


def test_sidebar_escape_does_not_steal_modal_keyboard_event_after_modal_hides():
    result = run_shared("""
let keydown;
document.addEventListener = (type, listener) => { if (type === 'keydown') keydown = listener; };
document.body = { dataset: {}, classList: { toggle() {} } };
document.querySelector = () => null;
document.querySelectorAll = () => [];
readyCallbacks[0]();
let prevented = false;
keydown({ key: 'Escape', target: { closest: () => ({}) }, preventDefault() { prevented = true; } });
process.stdout.write(JSON.stringify({ prevented }));
""")
    assert result == {"prevented": False}


def test_provider_panel_waits_for_status_before_offering_key_entry():
    result = run_shared("""
['apiKeyInput', 'saveSettingsBtn', 'apiKeyProviderSelect', 'apiKeyInputContainer', 'apiKeyConfiguredContainer', 'settingsStatus', 'providerSelect'].forEach(id => nodes[id] = makeNode());
nodes.providerSelect.value = 'openai';
let resolveStatus;
global.fetch = () => new Promise(resolve => resolveStatus = resolve);
window.IdeaWorkbench.openProviderSettings();
const waiting = { hiddenInput: nodes.apiKeyInputContainer.style.display, saveLocked: nodes.saveSettingsBtn.disabled, providerLocked: nodes.apiKeyProviderSelect.disabled, message: nodes.settingsStatus.textContent };
resolveStatus({ ok: true, json: async () => ({ api_keys: { openai: { configured: true, masked_key: 'masked' } } }) });
await new Promise(resolve => setImmediate(resolve));
process.stdout.write(JSON.stringify({ waiting, configured: nodes.apiKeyConfiguredContainer.style.display, saveLocked: nodes.saveSettingsBtn.disabled, providerLocked: nodes.apiKeyProviderSelect.disabled, statusHidden: nodes.settingsStatus.hidden }));
""")
    assert result["waiting"] == {"hiddenInput": "none", "saveLocked": True, "providerLocked": True, "message": "Checking saved provider keys…"}
    assert result["configured"] == "block"
    assert result["saveLocked"] is False
    assert result["providerLocked"] is False
    assert result["statusHidden"] is True


def test_unreadable_provider_has_explicit_replacement_form_without_any_mutation():
    result = run_shared("""
['apiKeyInput', 'saveSettingsBtn', 'apiKeyProviderSelect', 'apiKeyInputContainer', 'apiKeyConfiguredContainer', 'settingsStatus', 'providerSelect', 'selectedProviderStatus', 'configureSelectedProvider'].forEach(id => nodes[id] = makeNode());
nodes.providerSelect.value = 'google_generative_ai';
const calls = [];
global.fetch = async (url, options) => { calls.push({url, method: options?.method || 'GET'}); return {ok:true, json:async()=>({api_keys:{google_generative_ai:{configured:false,source:'unreadable',error_code:'stored_key_unreadable',message:'Saved key cannot be opened. Enter a replacement. The saved key has not been changed.'},openai:{configured:true}}})}; };
window.IdeaWorkbench.openProviderSettings();
await new Promise(resolve => setImmediate(resolve));
process.stdout.write(JSON.stringify({calls,input:nodes.apiKeyInputContainer.style.display,configured:nodes.apiKeyConfiguredContainer.style.display,saveLocked:nodes.saveSettingsBtn.disabled,message:nodes.settingsStatus.textContent,hidden:nodes.settingsStatus.hidden,readiness:nodes.selectedProviderStatus.textContent,action:nodes.configureSelectedProvider.textContent}));
""")
    assert result["calls"] == [{"url": "/api/settings/status", "method": "GET"}]
    assert result["input"] == "block" and result["configured"] == "none"
    assert result["saveLocked"] is False
    assert result["hidden"] is False and "replacement" in result["message"]
    assert result["readiness"] == "Gemini saved key unreadable"
    assert result["action"] == "Replace Gemini key"
