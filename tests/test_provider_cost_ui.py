import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "idea/static/js/workbench.js"


def run_shared(expression, setup=""):
    script = """
global.window = { currentUser: { uid: 'user-1' }, matchMedia: () => ({ matches: false, addEventListener() {} }), dispatchEvent() {} };
global.document = { addEventListener() {}, getElementById() { return null; } };
global.CustomEvent = class { constructor(type, options) { this.type = type; this.detail = options?.detail; } };
""" + setup + SHARED.read_text() + f"\nprocess.stdout.write(JSON.stringify({expression}));"
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_model_preferences_never_cross_providers_and_recover_removed_models():
    choices = run_shared("""[
        window.IdeaWorkbench.preferredModel(catalog, 'openai', { remembered: 'gemini-current', defaultModel: 'gemini-current' }),
        window.IdeaWorkbench.preferredModel(catalog, 'openai', { remembered: 'gpt-other' }),
        window.IdeaWorkbench.preferredModel(catalog, 'google_generative_ai', { remembered: 'gemini-retired', defaultModel: 'gemini-current' })
    ]""", """
const catalog = [
  { id: 'gemini-current', provider: 'google_generative_ai' },
  { id: 'gpt-default', provider: 'openai' },
  { id: 'gpt-other', provider: 'openai' }
];
""")
    assert choices == ["gpt-default", "gpt-other", "gemini-current"]


def test_switching_back_restores_each_model_and_each_independent_key_status():
    setup = """
const stored = new Map();
global.localStorage = { getItem: key => stored.get(key) ?? null, setItem: (key, value) => stored.set(key, String(value)) };
function control(value = '') {
  return {
    _value: value, dataset: {}, options: [], children: [], listeners: {}, hidden: false,
    get value() { return this.options.find(option => option.selected)?.value || this._value; },
    set value(next) { this._value = next; this.options.forEach(option => option.selected = option.value === next); },
    replaceChildren() { this.options = []; this.children = []; this._value = ''; },
    appendChild(child) { this.children.push(child); if (child.children) this.options.push(...child.children); },
    addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); },
    removeEventListener(type, listener) { this.listeners[type] = (this.listeners[type] || []).filter(value => value !== listener); },
    dispatchEvent(event) { (this.listeners[event.type] || []).forEach(listener => listener(event)); }
  };
}
const nodes = {
  providerSelect: control('google_generative_ai'), modelType: control('gemini-a'),
  selectedProviderStatus: control(), configureSelectedProvider: control()
};
document.getElementById = id => nodes[id] || null;
document.createElement = tag => tag === 'optgroup' ? { children: [], appendChild(child) { this.children.push(child); } } : { dataset: {} };
"""
    result = run_shared("""(() => {
        const app = window.IdeaWorkbench;
        app.configureModelPicker({ modelId: 'modelType', providerId: 'providerSelect', defaultModel: 'gemini-a', models: [
            { id: 'gemini-a', provider: 'google_generative_ai' },
            { id: 'gemini-b', provider: 'google_generative_ai' },
            { id: 'gpt-a', provider: 'openai' }, { id: 'gpt-b', provider: 'openai' }
        ] });
        app.setProviderSettings({ api_keys: { google_generative_ai: { configured: true }, openai: { configured: false } } });
        nodes.modelType.value = 'gemini-b';
        nodes.modelType.dispatchEvent(new Event('change'));
        nodes.providerSelect.value = 'openai';
        nodes.providerSelect.dispatchEvent(new Event('change'));
        const missingOpenAI = nodes.selectedProviderStatus.textContent;
        const openAIOptions = nodes.modelType.options.map(option => option.value);
        nodes.modelType.value = 'gpt-b';
        nodes.modelType.dispatchEvent(new Event('change'));
        app.setProviderSettings({ api_keys: { google_generative_ai: { configured: true }, openai: { configured: true } } });
        nodes.providerSelect.value = 'google_generative_ai';
        nodes.providerSelect.dispatchEvent(new Event('change'));
        const restoredGemini = nodes.modelType.value;
        nodes.providerSelect.value = 'openai';
        nodes.providerSelect.dispatchEvent(new Event('change'));
        const restoredOpenAI = nodes.modelType.value;
        const retiredWasFound = app.selectModel('modelType', 'gemini-retired', { fallback: true });
        return { missingOpenAI, openAIOptions, restoredGemini,
            restoredOpenAI, retiredWasFound, retiredFallback: nodes.modelType.value,
            readyProvider: nodes.selectedProviderStatus.textContent, setupHidden: nodes.configureSelectedProvider.hidden,
            stored: [...stored] };
    })()""", setup)
    assert result["openAIOptions"] == ["gpt-a", "gpt-b"]
    assert result["missingOpenAI"] == "OpenAI key needed"
    assert result["restoredGemini"] == "gemini-b"
    assert result["restoredOpenAI"] == "gpt-b"
    assert result["retiredWasFound"] is False
    assert result["retiredFallback"] == "gemini-b"
    assert result["readyProvider"] == "Gemini key ready"
    assert result["setupHidden"] is True
    assert all("key" not in name.lower() for name, _ in result["stored"])


def test_billing_preserves_missing_cost_and_explains_partial_estimate_coverage():
    result = run_shared("""({
        missing: window.IdeaWorkbench.money(null),
        zero: window.IdeaWorkbench.money(0),
        coverage: window.IdeaWorkbench.billingCoverageText({
            runs_missing_cost: 2, runs_missing_usage: 1, missing_usage_requests: 4, legacy_price_runs: 3,
            estimated_input_tokens: 400, unpriced_models: ['unknown-model'], truncated: true
        }, 8)
    })""")
    assert result["missing"] == "Not recorded"
    assert "0.00" in result["zero"]
    assert "2 runs have no recorded cost and are excluded" in result["coverage"]
    assert "1 run has incomplete token details" in result["coverage"]
    assert "4 requests returned no usage; their spend is unknown" in result["coverage"]
    assert "legacy price estimates" in result["coverage"]
    assert "400 input tokens were estimated locally" in result["coverage"]
    assert "unknown-model" in result["coverage"]
    assert "Only part of the saved history is included" in result["coverage"]


def test_missing_provider_usage_is_visible_even_when_other_usage_is_complete():
    text = run_shared("window.IdeaWorkbench.billingCoverageText({ runs_with_recorded_cost: 2, missing_usage_requests: 1 }, 2)")
    assert "1 request returned no usage" in text
    assert "spend is unknown" in text
    assert "2 runs have recorded cost" not in text


def test_provider_and_cost_panels_are_available_on_both_pages():
    for page, provider_id in (("viewer", "providerSelect"), ("rater", "ratingProviderSelect")):
        html = (ROOT / f"idea/static/html/{page}.html").read_text()
        assert f'id="{provider_id}"' in html
        assert 'id="selectedProviderStatus"' in html
        assert 'id="view-costs"' in html
        assert 'id="billingOverview"' in html
        assert 'id="apiKeyProviderSelect"' in html
        assert 'type="password"' in html
        assert 'id="cancelApiKeyEdit"' in html


def test_autorating_billing_warning_is_visible_as_text_and_clears_for_the_next_run():
    source = (ROOT / "idea/static/js/rater.js").read_text()
    start = source.index("function updateAutoratingBillingWarning")
    helper = source[start:source.index("function announceRatingStatus", start)]
    payload = '<img src=x onerror="alert(1)"> Usage could not be saved.'
    script = f"""
const warning = {{ hidden: true, textContent: '' }};
global.document = {{ getElementById: () => warning }};
{helper}
updateAutoratingBillingWarning({json.dumps(payload)});
const shown = {{ ...warning }};
updateAutoratingBillingWarning(null);
process.stdout.write(JSON.stringify({{ shown, reset: warning }}));
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    states = json.loads(result.stdout)
    assert states["shown"] == {"hidden": False, "textContent": payload}
    assert states["reset"] == {"hidden": True, "textContent": ""}
    html = (ROOT / "idea/static/html/rater.html").read_text()
    assert 'id="autoratingBillingWarning" class="alert alert-warning small" role="alert" hidden' in html
    assert "updateAutoratingBillingWarning(data.billing_warning)" in source
    assert "updateAutoratingBillingWarning(errorPayload.billing_warning)" in source
    progress = source[source.index("const pollProgress = async"):source.index("pollProgress();")]
    assert progress.index("updateAutoratingBillingWarning(progressData.billing_warning)") < progress.index("!progressData.is_running")
