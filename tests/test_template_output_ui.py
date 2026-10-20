"""Typed output/rule editing preserves legacy source data and guides correction."""
import json

import pytest

from tests.test_template_editor_navigation import run_editor


@pytest.mark.parametrize("word_count,mode", [
    (None, "none"),
    ({"min": 100, "max": 100}, "exactly"),
    ({"min": None, "max": 250}, "at-most"),
    ({"min": 50, "max": None}, "at-least"),
    ({"min": 80, "max": 120}, "between"),
])
def test_output_populate_and_read_round_trips_typed_word_limits(word_count, mode):
    result = run_editor("""
populateTemplateOutput({output:{title:'Short title',content:'A complete story',word_count:COUNT}});
const output=readTemplateOutput();
process.stdout.write(JSON.stringify({output,mode:field('editWordCountMode').value,firstRequired:field('editWordCountValue').required,secondRequired:field('editWordCountMax').required,firstHidden:field('templateWordCountValueGroup').hidden,secondHidden:field('templateWordCountMaxGroup').hidden,dirty:templateEditorDirty}));
""".replace("COUNT", json.dumps(word_count)))
    assert result["output"] == {"title": "Short title", "content": "A complete story", "word_count": word_count}
    assert result["mode"] == mode and result["dirty"] is False
    assert result["firstRequired"] is (mode != "none")
    assert result["secondRequired"] is (mode == "between")
    assert result["firstHidden"] is (mode == "none")
    assert result["secondHidden"] is (mode != "between")


@pytest.mark.parametrize("value", ["", "0", "-1", "1.5", "1e3", "9007199254740992"])
def test_invalid_word_counts_activate_output_and_focus_the_number(value):
    result = run_editor("""
field('editWordCountMode').value='exactly';field('editWordCountValue').value=VALUE;updateTemplateOutputFields();
selectTemplateEditorSection('prompts');const output=readTemplateOutput();
process.stdout.write(JSON.stringify({output,section:templateEditorSection,focus:document.activeElement?.id,invalid:field('editWordCountValue').getAttribute('aria-invalid'),message:field('templateEditorFeedback').textContent}));
""".replace("VALUE", json.dumps(value)))
    assert result == {"output": None, "section": "output", "focus": "editWordCountValue", "invalid": "true", "message": "Enter a positive whole number for the word count."}


def test_reversed_range_focuses_maximum_and_no_limit_ignores_retained_numbers():
    result = run_editor("""
populateTemplateOutput({output:{word_count:{min:200,max:100}}});selectTemplateEditorSection('overview');
const bad=readTemplateOutput();const error={value:bad,focus:document.activeElement?.id,message:field('templateEditorFeedback').textContent};
field('editWordCountMode').value='none';updateTemplateOutputFields();
process.stdout.write(JSON.stringify({error,output:readTemplateOutput(),disabled:field('editWordCountValue').disabled&&field('editWordCountMax').disabled}));
""")
    assert result["error"] == {"value": None, "focus": "editWordCountMax", "message": "Maximum word count must be at least the minimum."}
    assert result["output"]["word_count"] is None and result["disabled"] is True


def test_requirements_are_atomic_and_legacy_guidance_is_never_split():
    result = run_editor(r"""
const legacy='  Existing rules:\n- First legacy rule\n- Second legacy rule\n';
populateTemplateOutput({requirements:['One requirement with\na deliberate newline','Another requirement'],special_requirements:legacy});
selectTemplateEditorSection('output');
const before={values:document.querySelectorAll('.requirement-input').map(x=>x.value),legacy:field('editSpecialRequirements').value,count:field('templateRequirementsCount').textContent,legacyHidden:field('templateAdditionalGuidance').hidden};
addEditRequirement('New requirement',true);
const rows=document.querySelectorAll('#editRequirementsContainer .template-requirement-row');rows[1].querySelector('button').listeners.click();
process.stdout.write(JSON.stringify({before,values:document.querySelectorAll('.requirement-input').map(x=>x.value),labels:document.querySelectorAll('.requirement-input').map(x=>x.getAttribute('aria-label')),count:field('templateRequirementsCount').textContent,focus:document.activeElement?.value,dirty:templateEditorDirty}));
""")
    assert result["before"] == {"values": ["One requirement with\na deliberate newline", "Another requirement"], "legacy": "  Existing rules:\n- First legacy rule\n- Second legacy rule\n", "count": "2", "legacyHidden": False}
    assert result["values"] == ["One requirement with\na deliberate newline", "New requirement"]
    assert result["labels"] == ["Requirement 1", "Requirement 2"]
    assert result["count"] == "2" and result["focus"] == "New requirement" and result["dirty"] is True


def test_unsupported_output_is_preserved_until_explicit_removal_then_valid_save_keeps_other_source():
    result = run_editor(r"""
currentEditingTemplateId='custom';currentEditingTemplateSource={metadata:{item_type:'stories',custom:'keep'},advanced:{arbitrary:['keep']},output:{title:'Before',content:'Before',word_count:{min:80,max:120,unit:'words'},sections:['Keep until reviewed']},prompts:{oracle_constraints:'untouched'},special_requirements:'Legacy rules'};
populateTemplateOutput({requirements:['Existing requirement'],special_requirements:'Legacy rules'});
field('editOutputTitle').value='Evocative title';field('editOutputContent').value='A story\nwith a twist';field('editWordCountMode').value='between';field('editWordCountValue').value='80';field('editWordCountMax').value='120';updateTemplateOutputFields();
field('editSpecialRequirements').value='';addEditRequirement('New requirement');addEditCriterion('Originality');
let saved,requests=0;global.fetch=async(url,options)=>{requests++;saved=JSON.parse(options.body);return{ok:true,json:async()=>({status:'success',template_id:'custom'})};};
global.bootstrap={Modal:{getInstance:()=>({hide(){}})}};global.loadExistingTemplates=async()=>{};global.showSimpleAlert=()=>{};
await saveTemplateMainPage();const blocked={requests,output:JSON.parse(JSON.stringify(currentEditingTemplateSource.output)),section:templateEditorSection,focus:document.activeElement?.id,open:field('templateOutputRepair').open};
removeUnsupportedTemplateOutputFields();const repaired={hidden:field('templateOutputRepair').hidden,dirty:templateEditorDirty,title:field('editOutputTitle').value};
await saveTemplateMainPage();const source=currentEditingTemplateSource;populateTemplateOutput(source);
process.stdout.write(JSON.stringify({blocked,repaired,saved,legacyHidden:field('templateAdditionalGuidance').hidden,legacy:field('editSpecialRequirements').value,error:field('templateEditorFeedback').textContent}));
""")
    assert result["blocked"] == {"requests": 0, "output": {"title": "Before", "content": "Before", "word_count": {"min": 80, "max": 120, "unit": "words"}, "sections": ["Keep until reviewed"]}, "section": "output", "focus": "removeUnsupportedOutputFields", "open": True}
    assert result["repaired"] == {"hidden": True, "dirty": True, "title": "Evocative title"}
    saved = result["saved"]
    assert saved["template_data"]["output"] == {"title": "Evocative title", "content": "A story\nwith a twist", "word_count": {"min": 80, "max": 120}}
    assert saved["template_data"]["requirements"] == ["Existing requirement", "New requirement"]
    assert saved["template_data"]["advanced"] == {"arbitrary": ["keep"]}
    assert saved["template_data"]["metadata"]["custom"] == "keep"
    assert saved["template_data"]["prompts"]["oracle_constraints"] == "untouched"
    assert saved["special_requirements"] == saved["template_data"]["special_requirements"] == ""
    assert result["legacyHidden"] is True and result["legacy"] == ""
    assert result["error"] == "Output contains unsupported fields. Review and remove them before saving."


def test_clearing_editor_resets_structured_fields_and_hides_legacy_guidance():
    result = run_editor("""
populateTemplateOutput({requirements:['Rule'],output:{title:'Before',content:'Before',word_count:{min:5,max:10}},special_requirements:'Legacy'});
clearTemplateEditForm();
process.stdout.write(JSON.stringify({output:readTemplateOutput(),requirements:document.querySelectorAll('.requirement-input').length,count:field('templateRequirementsCount').textContent,legacy:field('editSpecialRequirements').value,hidden:field('templateAdditionalGuidance').hidden}));
""")
    assert result == {"output": {"title": "", "content": "", "word_count": None}, "requirements": 0, "count": "0", "legacy": "", "hidden": True}


@pytest.mark.parametrize("invalid_output", ["A paragraph instead of output fields", ["unexpected"], {"title": "Keep title", "word_count": "100 words"}, {"title": "Keep title", "word_count": [100]}])
def test_malformed_output_containers_require_explicit_repair_without_losing_other_data(invalid_output):
    result = run_editor("""
currentEditingTemplateSource={output:OUTPUT,metadata:{custom:'keep'},prompts:{extra:'keep'}};
const before=JSON.stringify(currentEditingTemplateSource);populateTemplateOutput(currentEditingTemplateSource);
const preserved=JSON.stringify(currentEditingTemplateSource)===before;
selectTemplateEditorSection('output');const blocked=readTemplateOutput();
removeUnsupportedTemplateOutputFields();
process.stdout.write(JSON.stringify({preserved,blocked,source:currentEditingTemplateSource,hidden:field('templateOutputRepair').hidden,dirty:templateEditorDirty,readable:readTemplateOutput()}));
""".replace("OUTPUT", json.dumps(invalid_output)))
    assert result["preserved"] is True and result["blocked"] is None
    expected = {"title": "Keep title", "word_count": None} if isinstance(invalid_output, dict) else {}
    assert result["source"]["output"] == expected
    assert result["source"]["metadata"] == {"custom": "keep"}
    assert result["source"]["prompts"] == {"extra": "keep"}
    assert result["hidden"] is True and result["dirty"] is True
    assert result["readable"]["word_count"] is None


@pytest.mark.parametrize("source", [{}, {"output": None}, {"output": {}}])
def test_optional_output_can_be_absent_null_or_empty_without_repair(source):
    result = run_editor("""
currentEditingTemplateSource=SOURCE;populateTemplateOutput(currentEditingTemplateSource);
process.stdout.write(JSON.stringify({issues:getTemplateOutputRepairIssues(),hidden:field('templateOutputRepair').hidden,output:readTemplateOutput(),dirty:templateEditorDirty}));
""".replace("SOURCE", json.dumps(source)))
    assert result == {"issues": [], "hidden": True, "output": {"title": "", "content": "", "word_count": None}, "dirty": False}


def test_repair_disclosure_renders_unknown_keys_and_values_as_text_without_source_mutation():
    result = run_editor(r"""
const key='<img src=x onerror=alert(1)>';
currentEditingTemplateSource={output:{title:'Title',content:'Content',word_count:{min:10,max:20,extra:'<svg onload=alert(2)>'},[key]:['</pre><script>alert(3)</script>']}};
const before=JSON.stringify(currentEditingTemplateSource);populateTemplateOutput(currentEditingTemplateSource);
const entries=field('templateOutputRepairFields').children.map(item=>({key:item.querySelector('code').textContent,value:item.querySelector('pre').textContent,html:item.innerHTML}));
process.stdout.write(JSON.stringify({entries,preserved:JSON.stringify(currentEditingTemplateSource)===before,hidden:field('templateOutputRepair').hidden,dirty:templateEditorDirty}));
""")
    assert result["entries"] == [
        {"key": "output.<img src=x onerror=alert(1)>", "value": '[\n  "</pre><script>alert(3)</script>"\n]', "html": ""},
        {"key": "output.word_count.extra", "value": '"<svg onload=alert(2)>"', "html": ""},
    ]
    assert result["preserved"] is True and result["dirty"] is False and result["hidden"] is False


def test_inline_add_requirement_handler_is_not_shadowed_by_form_named_properties():
    result = run_editor("""
populateTemplateOutput({requirements:['First requirement','Second requirement']});selectTemplateEditorSection('output');
const named = Object.fromEntries(document.querySelectorAll('#templateEditForm [id]').filter(element=>['BUTTON','INPUT','TEXTAREA','SELECT'].includes(element.tagName)).map(element=>[element.id,element]));
// Browser inline form handlers resolve named controls before global functions.
eval('with(named){' + field('addTemplateRequirement').getAttribute('onclick') + '}');
process.stdout.write(JSON.stringify({values:document.querySelectorAll('.requirement-input').map(input=>input.value),count:field('templateRequirementsCount').textContent,focused:document.activeElement?.getAttribute('aria-label'),dirty:templateEditorDirty}));
""")
    assert result == {"values": ["First requirement", "Second requirement", ""], "count": "3", "focused": "Requirement 3", "dirty": True}
