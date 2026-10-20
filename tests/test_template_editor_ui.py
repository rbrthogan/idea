"""No-provider-call checks for template generation choices and edit preservation."""
import json
import subprocess
from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / 'idea/static/html/viewer.html').read_text()


def bundle(start, end):
    offset = HTML.index(start)
    return HTML[offset:HTML.index(end, offset)]


def run_node(script):
    result = subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


GENERATION = bundle('    // Template draft request ownership', '    function getTemplatePreviewDisplay')
ERROR_HELPER = bundle('    function templateResponseMessage', '    let templateGenerationCatalog')


def generation_setup(file='null'):
    return '''
const controls = {};
for (const id of ['ideaTypeInput', 'templateGenerationProvider', 'templateGenerationModel', 'generateTemplateBtn', 'templateRemoveAttachmentBtn', 'templatePreview']) controls[id] = { value: '', disabled: false, innerHTML: 'Generate', focus() {}, scrollIntoView() {} };
controls.ideaTypeInput.value = 'A tiny science-fiction story';
controls.templateGenerationProvider.value = 'openai';
controls.templateGenerationModel.value = 'gpt-6-astra';
controls.templateGenerationModel.selectedOptions = [{ textContent: 'OpenAI Astra' }];
global.document = { getElementById: id => controls[id] };
global.window = { addEventListener() {}, IdeaWorkbench: { loadProviderSettings: async () => ({ api_keys: { openai: { configured: true } } }), providerName: () => 'OpenAI' } };
let templateGenerationInFlight = false, currentGeneratedTemplate = null, currentGeneratedTemplateId = null, currentGeneratedTemplateValidation = null;
const statuses = [], requests = [];
function showStatus(message, type) { statuses.push([message, type]); }
function hideStatus() {}
function showTemplatePreview(template) { currentGeneratedTemplate = template; }
function updateTemplateGenerationReadiness() {}
function getSelectedTemplateContextFile() { return FILE; }
global.fetch = async (url, options) => { requests.push({ url, options }); return { ok: true, json: async () => ({ status: 'success', template: { name: 'Draft' } }) }; };
'''.replace('FILE', file)


def test_generation_uses_explicit_best_model_and_blocks_duplicate_shortcuts():
    result = run_node(generation_setup() + ERROR_HELPER + GENERATION + '''
(async () => {
  await Promise.all([handleTemplateGeneration(), handleTemplateGeneration()]);
  process.stdout.write(JSON.stringify({ requests, brief: controls.ideaTypeInput.value, disabled: controls.generateTemplateBtn.disabled, template: currentGeneratedTemplate }));
})().catch(error => { console.error(error); process.exit(1); });
''')
    assert len(result['requests']) == 1
    assert json.loads(result['requests'][0]['options']['body']) == {
        'idea_type_suggestion': 'A tiny science-fiction story', 'model_id': 'gpt-6-astra', 'provider': 'openai',
    }
    assert result['brief'] == 'A tiny science-fiction story'
    assert result['disabled'] is False
    assert result['template']['name'] == 'Draft'


def test_openai_attachment_is_preserved_and_rejected_before_any_provider_request():
    result = run_node(generation_setup("{ name: 'source.pdf' }") + ERROR_HELPER + GENERATION + '''
(async () => {
 await handleTemplateGeneration();
 process.stdout.write(JSON.stringify({ requests, statuses, file: getSelectedTemplateContextFile(), brief: controls.ideaTypeInput.value }));
})().catch(error => { console.error(error); process.exit(1); });
''')
    assert result['requests'] == []
    assert result['file']['name'] == 'source.pdf'
    assert 'still selected' in result['statuses'][0][0]
    assert result['brief'] == 'A tiny science-fiction story'


def test_failed_generation_retains_previous_draft_and_entered_brief():
    result = run_node(generation_setup() + ERROR_HELPER + GENERATION + '''
currentGeneratedTemplate = { name: 'Previous draft' };
fetch = async () => ({ ok: false, json: async () => ({ detail: [{ msg: 'Model unavailable' }] }) });
(async () => {
 await handleTemplateGeneration();
 process.stdout.write(JSON.stringify({ template: currentGeneratedTemplate, brief: controls.ideaTypeInput.value, disabled: controls.generateTemplateBtn.disabled, statuses }));
})().catch(error => { console.error(error); process.exit(1); });
''')
    assert result['template']['name'] == 'Previous draft'
    assert result['brief'] == 'A tiny science-fiction story'
    assert result['disabled'] is False
    assert result['statuses'][-1] == ['Model unavailable', 'danger']


def test_template_editor_sends_optional_prompts_and_metadata_without_loss():
    save = bundle('    async function saveTemplateMainPage()', '    async function deleteTemplateMainPage')
    result = run_node('''
const values = { editTemplateName: 'Story template', editTemplateDescription: 'Stories', editTemplateAuthor: 'User', editTemplateItemType: 'stories', editSpecialRequirements: '', editOutputTitle: '', editOutputContent: '', editWordCountMode: 'none', editContextPrompt: 'Generate concepts', editSpecificPrompt: 'Use {context_pool}', editFormatPrompt: 'Format {input_text}', editCritiquePrompt: 'Critique {idea}', editRefinePrompt: 'Improve {idea} with {critique}', editBreedPrompt: 'Combine {ideas}', editGenotypePrompt: 'Encode {idea_content}' };
const controls = Object.fromEntries(Object.entries(values).map(([id, value]) => [id, { value }]));
controls.templateEditorFeedback = { hidden: false };
controls.templateEditModal = {};
global.document = { getElementById: id => controls[id], querySelectorAll: selector => selector.includes('criterion-input') ? [{ value: 'Originality' }] : selector.includes('optional-template-prompt') ? [{ value: 'Updated oracle constraints', dataset: { promptScope: 'prompts', promptKey: 'oracle_constraints' } }] : [] };
let templateEditorBusy = false, templateEditorDirty = true, templateEditorAllowClose = false, templateEditorFromGenerated = false;
let currentEditingTemplateId = 'story-template', currentEditingTemplateIsSystem = false;
let currentEditingTemplateSource = { version: '1.2.0', metadata: { item_type: 'stories', custom_note: 'Keep' }, prompts: { example_idea_prompts: 'A specific story direction', oracle_constraints: 'Original constraints' }, comparison_prompt: 'Legacy root comparison' };
let request;
function setTemplateEditorBusy(value) { templateEditorBusy = value; }
function showTemplateEditorFeedback(message) { throw new Error(message); }
function showSimpleAlert() {}
async function loadExistingTemplates() {}
function selectTemplate() {}
global.bootstrap = { Modal: { getInstance: () => ({ hide() {} }) } };
global.fetch = async (url, options) => { request = { url, ...options }; return { ok: true, json: async () => ({ status: 'success', template_id: 'story-template' }) }; };
''' + ERROR_HELPER + bundle('    function getTemplateOutputRepairIssues(', '    function updateTemplateOutputFields') + bundle('    function readTemplateOutput()', '    function selectTemplateEditorSection') + save + '''
(async () => {
 await saveTemplateMainPage();
 process.stdout.write(JSON.stringify({ request, dirty: templateEditorDirty, busy: templateEditorBusy }));
})().catch(error => { console.error(error); process.exit(1); });
''')
    data = json.loads(result['request']['body'])
    assert data['template_data']['prompts']['example_idea_prompts'] == 'A specific story direction'
    assert data['template_data']['prompts']['oracle_constraints'] == 'Updated oracle constraints'
    assert data['template_data']['comparison_prompt'] == 'Legacy root comparison'
    assert data['template_data']['metadata']['custom_note'] == 'Keep'
    assert data['template_data']['prompts']['specific_prompt'] == data['specific_prompt'] == 'Use {context_pool}'
    assert data['special_requirements'] == ''
    assert result['dirty'] is False
    assert result['busy'] is False


def test_lost_generation_response_explains_possible_billing_without_discarding_brief():
    result = run_node(generation_setup() + ERROR_HELPER + GENERATION + '''
currentGeneratedTemplate = { name: 'Previous draft' };
fetch = async () => { throw new TypeError('Failed to fetch'); };
(async () => {
 await handleTemplateGeneration();
 process.stdout.write(JSON.stringify({ template: currentGeneratedTemplate, brief: controls.ideaTypeInput.value, disabled: controls.generateTemplateBtn.disabled, message: statuses.at(-1)[0] }));
})().catch(error => { console.error(error); process.exit(1); });
''')
    assert result['template']['name'] == 'Previous draft'
    assert result['brief'] == 'A tiny science-fiction story'
    assert result['disabled'] is False
    assert 'could not be confirmed' in result['message']
    assert 'processing and billing' in result['message']
    assert 'wait before trying again' in result['message']


def test_key_status_failure_before_generation_does_not_claim_possible_billing():
    result = run_node(generation_setup() + ERROR_HELPER + GENERATION + '''
window.IdeaWorkbench.loadProviderSettings = async () => { throw new TypeError('Could not check provider key'); };
(async () => {
 await handleTemplateGeneration();
 process.stdout.write(JSON.stringify({ requests, message: statuses.at(-1)[0], disabled: controls.generateTemplateBtn.disabled }));
})().catch(error => { console.error(error); process.exit(1); });
''')
    assert result == {'requests': [], 'message': 'Could not check provider key', 'disabled': False}
