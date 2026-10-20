"""Functional draft revision checks; all provider requests are mocked in Node."""

import json
from pathlib import Path
import subprocess

import pytest


HTML = (Path(__file__).resolve().parents[1] / "idea/static/html/viewer.html").read_text()


def bundle(start, end):
    offset = HTML.index(start)
    return HTML[offset:HTML.index(end, offset)]


HELPERS = bundle("    function templateResponseMessage", "    let templateGenerationCatalog")
DRAFT = bundle("    // Template draft request ownership", "    function getTemplatePreviewDisplay")
ACTIONS = bundle("    async function saveAndUseTemplate()", "    function selectNewTemplate")

SETUP = r"""
const assert = require('node:assert/strict');
const tick = () => new Promise(resolve => setImmediate(resolve));
function deferred() { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; }
const controls = {};
for (const id of ['ideaTypeInput', 'templateGenerationProvider', 'templateGenerationModel', 'generateTemplateBtn', 'templateRemoveAttachmentBtn', 'templateRevisionInstruction', 'reviseTemplateBtn', 'undoTemplateRevisionBtn', 'templateRevisionStatus', 'saveTemplateBtn', 'reviewEditBtn', 'templatePreview', 'templateEditModal', 'templateEditModalTitle', 'deleteTemplateMainPageBtn']) {
  controls[id] = { value: '', disabled: false, hidden: false, innerHTML: id, textContent: '', dataset: {}, style: {}, attrs: {}, classList: { contains() { return false; } }, focus() { this.focused = true; }, setAttribute(name, value) { this.attrs[name] = value; }, scrollIntoView() {} };
}
controls.ideaTypeInput.value = 'Original brief';
controls.templateRevisionInstruction.value = '  Broaden the scope.  ';
controls.templateGenerationProvider.value = 'google_generative_ai';
controls.templateGenerationModel.value = 'gemini-4-argon';
controls.templateGenerationModel.selectedOptions = [{ textContent: 'Gemini 4 Argon' }];
const initial = { name: 'Original', description: 'A complete draft', author: 'User', metadata: { item_type: 'proposals', note: 'Preserve metadata' }, requirements: ['Be practical'], output: { title: 'Short title', content: 'A testable proposal', word_count: { min: 100, max: 200 } }, comparison_criteria: ['Clarity'], prompts: { context: 'Generate contexts', specific_prompt: 'Use {context_pool}', format: 'Extract {input_text}', critique: 'Critique {idea}', refine: 'Refine {idea} {critique}', breed: 'Combine {ideas}', genotype_encode: 'Encode {idea_content}', oracle_constraints: 'Preserve optional prompt' }, custom: { retained: true } };
const initialValidation = { is_valid: true, warnings: [] };
let currentGeneratedTemplate = structuredClone(initial), currentGeneratedTemplateValidation = structuredClone(initialValidation), currentGeneratedTemplateId = 'saved-source';
let templateGenerationInFlight = false, templateEditorBusy = false, templateEditorFromGenerated = false, templateEditorLoadRequest = 0;
let currentEditingTemplateId = null, currentEditingTemplateIsSystem = false;
const events = {}, requests = [], previews = [], statuses = [], alerts = [], selected = [], opened = [];
global.document = { getElementById: id => controls[id] };
global.window = { currentUser: { uid: 'account-a' }, addEventListener: (name, callback) => { events[name] = callback; }, IdeaWorkbench: { loadProviderSettings: async () => ({ api_keys: { google_generative_ai: { configured: true }, openai: { configured: true } } }), providerName: provider => provider === 'openai' ? 'OpenAI' : 'Gemini' } };
global.fetch = (url, options) => { const request = { url, options, deferred: deferred() }; requests.push(request); return request.deferred.promise; };
function reply(index, data, ok = true) { requests[index].deferred.resolve({ ok, json: async () => data }); }
function success(name = 'Revised') { return { status: 'success', template: { ...structuredClone(initial), name }, validation: structuredClone(initialValidation) }; }
function updateTemplateGenerationReadiness() {}
function getSelectedTemplateContextFile() { return null; }
function showStatus(message, type) { statuses.push({ message, type }); }
function hideStatus() { statuses.length = 0; }
function showTemplatePreview(template, validation) { previews.push({ template: structuredClone(template), validation }); controls.templatePreview.style.display = 'block'; updateTemplateDraftControls(); }
function hideTemplatePreview() { controls.templatePreview.style.display = 'none'; }
function clearTemplateAttachment() {}
function showSimpleAlert(message, type) { alerts.push({ message, type }); }
function setTemplateFlowMode() {}
function selectNewTemplate(id, name) { selected.push({ id, name }); clearGeneratedTemplateDraft(); }
function populateTemplateEditForm(template) { opened.push(template); }
function openTemplateEditor() { controls.templateEditModal.classList.contains = () => true; updateTemplateDraftControls(); }
"""


def run_case(body):
    script = SETUP + HELPERS + DRAFT + ACTIONS + "\n(async () => {\n" + body + "\n})().catch(error => { console.error(error); process.exit(1); });"
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_revision_sends_complete_current_draft_with_model_and_blocks_competing_actions():
    run_case("""
updateTemplateDraftControls();
assert.equal(controls.reviseTemplateBtn.textContent, 'Revise with Gemini 4 Argon');
const pending = handleTemplateRevision();
await tick();
assert.equal(requests.length, 1);
assert.equal(requests[0].url, '/api/templates/revise');
assert.deepEqual(JSON.parse(requests[0].options.body), { template: initial, instruction: 'Broaden the scope.', model_id: 'gemini-4-argon', provider: 'google_generative_ai' });
for (const id of ['reviseTemplateBtn', 'generateTemplateBtn', 'reviewEditBtn', 'saveTemplateBtn', 'templateRevisionInstruction', 'templateGenerationModel', 'templateGenerationProvider']) assert.equal(controls[id].disabled, true, id);
assert.equal(controls.templatePreview.attrs['aria-busy'], 'true');
await Promise.all([handleTemplateRevision(), handleTemplateGeneration(), saveAndUseTemplate(), reviewAndEditTemplate()]);
assert.equal(requests.length, 1);
assert.equal(opened.length, 0);
reply(0, success());
await pending;
assert.equal(currentGeneratedTemplate.name, 'Revised');
assert.equal(currentGeneratedTemplateId, null);
assert.equal(controls.templateRevisionInstruction.value, '');
assert.equal(controls.saveTemplateBtn.disabled, false);
assert.equal(controls.undoTemplateRevisionBtn.hidden, false);
assert.equal(controls.templatePreview.attrs['aria-busy'], 'false');
""")


def test_next_followup_uses_latest_draft_and_undo_restores_previous_draft_validation_and_id():
    run_case("""
const first = handleTemplateRevision(); await tick(); reply(0, success('Version 2')); await first;
controls.templateRevisionInstruction.value = 'Make the audience broader';
const second = handleTemplateRevision(); await tick();
assert.equal(JSON.parse(requests[1].options.body).template.name, 'Version 2');
reply(1, success('Version 3')); await second;
undoTemplateRevision();
assert.equal(currentGeneratedTemplate.name, 'Version 2');
assert.equal(currentGeneratedTemplateId, null);
assert.deepEqual(currentGeneratedTemplateValidation, initialValidation);
assert.equal(controls.templateRevisionInstruction.value, 'Make the audience broader');
assert.equal(controls.undoTemplateRevisionBtn.hidden, true);
assert.equal(requests.length, 2);
""")


def test_undo_of_first_revision_restores_saved_identity_without_saving_or_overwriting():
    run_case("""
const pending = handleTemplateRevision(); await tick(); reply(0, success()); await pending;
undoTemplateRevision();
assert.deepEqual(currentGeneratedTemplate, initial);
assert.deepEqual(currentGeneratedTemplateValidation, initialValidation);
assert.equal(currentGeneratedTemplateId, 'saved-source');
assert.equal(controls.templateRevisionInstruction.value, '  Broaden the scope.  ');
await saveAndUseTemplate();
assert.equal(requests.length, 1);
assert.deepEqual(selected, [{ id: 'saved-source', name: 'Original' }]);
""")


def test_saving_revised_saved_template_creates_new_copy_only_after_explicit_save():
    run_case("""
const pending = handleTemplateRevision(); await tick(); reply(0, success()); await pending;
assert.equal(requests.length, 1);
const saving = saveAndUseTemplate(); await tick();
assert.equal(requests[1].url, '/api/templates/');
assert.equal(requests[1].options.method, 'POST');
assert.equal(JSON.parse(requests[1].options.body).template_data.name, 'Revised');
assert.equal(controls.reviseTemplateBtn.disabled, true);
await handleTemplateRevision();
assert.equal(requests.length, 2);
reply(1, { status: 'success', template_id: 'new-copy' }); await saving;
assert.deepEqual(selected, [{ id: 'new-copy', name: 'Revised' }]);
""")


@pytest.mark.parametrize("failure", ["server", "network", "malformed", "invalid_template"])
def test_failed_revision_preserves_draft_instruction_identity_and_unlocks_controls(failure):
    action = {
        "server": "reply(0, { status: 'error', message: 'Model unavailable' }, false);",
        "network": "requests[0].deferred.reject(new TypeError('Failed to fetch'));",
        "malformed": "requests[0].deferred.resolve({ ok: true, json: async () => { throw new SyntaxError('Unexpected token'); } });",
        "invalid_template": "reply(0, { ...success(), validation: { is_valid: false, warnings: ['Missing prompt'] } });",
    }[failure]
    run_case("""
const pending = handleTemplateRevision(); await tick();
""" + action + """
await pending;
assert.deepEqual(currentGeneratedTemplate, initial);
assert.deepEqual(currentGeneratedTemplateValidation, initialValidation);
assert.equal(currentGeneratedTemplateId, 'saved-source');
assert.equal(controls.templateRevisionInstruction.value, '  Broaden the scope.  ');
assert.equal(controls.reviseTemplateBtn.disabled, false);
assert.equal(controls.saveTemplateBtn.disabled, false);
assert.equal(previousTemplateRevision, null);
assert.equal(controls.templateRevisionStatus.dataset.state, 'danger');
""" + ("assert.match(controls.templateRevisionStatus.textContent, /processing and billing/); assert.match(controls.templateRevisionStatus.textContent, /wait before trying again/);" if failure in {"network", "malformed"} else "assert.doesNotMatch(controls.templateRevisionStatus.textContent, /processing and billing/);"))


@pytest.mark.parametrize("blocked", ["empty", "oversized", "invalid_draft", "no_key", "manual_editor"])
def test_invalid_or_busy_inputs_do_not_start_paid_revision(blocked):
    setup = {
        "empty": "controls.templateRevisionInstruction.value = '   ';",
        "oversized": "controls.templateRevisionInstruction.value = 'x'.repeat(12001);",
        "invalid_draft": "currentGeneratedTemplateValidation = { is_valid: false, warnings: ['Missing context'] };",
        "no_key": "window.IdeaWorkbench.loadProviderSettings = async () => ({ api_keys: {} });",
        "manual_editor": "controls.templateEditModal.classList.contains = () => true;",
    }[blocked]
    run_case(setup + """
await handleTemplateRevision();
assert.equal(requests.length, 0);
assert.deepEqual(currentGeneratedTemplate, initial);
assert.equal(currentGeneratedTemplateId, 'saved-source');
assert.equal(templateGenerationInFlight, false);
""")


def test_account_switch_during_key_preflight_prevents_provider_request():
    run_case("""
const keys = deferred(); window.IdeaWorkbench.loadProviderSettings = () => keys.promise;
const pending = handleTemplateRevision(); await tick();
window.currentUser = { uid: 'account-b' }; events['idea-auth-ready']();
keys.resolve({ api_keys: { google_generative_ai: { configured: true } } }); await pending;
assert.equal(requests.length, 0);
assert.equal(currentGeneratedTemplate, null);
assert.equal(currentGeneratedTemplateId, null);
assert.equal(controls.ideaTypeInput.value, '');
assert.equal(controls.templateRevisionInstruction.value, '');
assert.equal(templateGenerationInFlight, false);
""")


@pytest.mark.parametrize("action", ["discard", "sign_out", "switch_account"])
@pytest.mark.parametrize("response", ["success", "failure"])
def test_late_revision_response_cannot_restore_discarded_or_other_account_draft(action, response):
    mutation = {
        "discard": "clearGeneratedTemplateDraft();",
        "sign_out": "window.currentUser = null; events['idea-auth-cleared']();",
        "switch_account": "window.currentUser = { uid: 'account-b' }; events['idea-auth-ready']();",
    }[action]
    completion = "reply(0, success());" if response == "success" else "requests[0].deferred.reject(new TypeError('Failed to fetch'));"
    run_case("const pending = handleTemplateRevision(); await tick();\n" + mutation + """
assert.equal(templateGenerationInFlight, true);
assert.equal(controls.templatePreview.style.display, 'none');
""" + completion + """
await pending;
assert.equal(currentGeneratedTemplate, null);
assert.equal(previousTemplateRevision, null);
assert.equal(previews.length, 0);
assert.equal(controls.templateRevisionStatus.hidden, true);
assert.equal(templateGenerationInFlight, false);
""")


def test_generation_locks_revision_and_ignores_response_after_draft_discard():
    run_case("""
const pending = handleTemplateGeneration(); await tick();
await Promise.all([handleTemplateRevision(), saveAndUseTemplate(), reviewAndEditTemplate()]);
assert.equal(requests.length, 1);
assert.equal(requests[0].url, '/api/templates/generate');
assert.equal(opened.length, 0);
clearGeneratedTemplateDraft(); reply(0, success('Late generation')); await pending;
assert.equal(currentGeneratedTemplate, null);
assert.equal(previews.length, 0);
assert.equal(templateGenerationInFlight, false);
""")


def test_late_save_response_does_not_select_a_discarded_draft():
    run_case("""
currentGeneratedTemplateId = null;
const pending = saveAndUseTemplate(); await tick();
clearGeneratedTemplateDraft();
reply(0, { status: 'success', template_id: 'saved-later' }); await pending;
assert.equal(selected.length, 0);
assert.equal(alerts.length, 0);
assert.equal(currentGeneratedTemplate, null);
assert.equal(templateDraftSaveInFlight, false);
""")
