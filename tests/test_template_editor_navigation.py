"""Behavioral checks for the focused editor using its real markup and JS helpers."""
import json
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

HTML = (Path(__file__).resolve().parents[1] / "idea/static/html/viewer.html").read_text()


def section(start, end):
    offset = HTML.index(start)
    return HTML[offset:HTML.index(end, offset)]


class EditorMarkup(HTMLParser):
    def __init__(self):
        super().__init__()
        self.root = {"tag": "root", "attrs": {}, "children": []}
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        item = {"tag": tag, "attrs": dict(attrs), "children": []}
        self.stack[-1]["children"].append(item)
        if tag not in {"input", "br", "hr", "img", "meta", "link"}:
            self.stack.append(item)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index]["tag"] == tag:
                self.stack = self.stack[:index]
                break


MARKUP = EditorMarkup()
MARKUP.feed(section("  <!-- Focused template editor -->", "  <!-- Include original viewer.js"))
HELPERS = section("    // Template editing functions for main page", "    async function editTemplateMainPage")
CRITERION = section("    function addEditCriterion", "    async function saveTemplateMainPage")
SAVE = section("    async function saveTemplateMainPage", "    async function deleteTemplateMainPage")
POPULATE = section("    function populateTemplateEditForm", "    function addEditCriterion")


def run_editor(body):
    harness = r"""
const camel = name => name.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
class Element {
    constructor(tag, attrs={}) {
        this.tagName=tag.toUpperCase();this.attrs={...attrs};this.children=[];this.parentElement=null;this.dataset={};this.style={};this.listeners={};this.value=attrs.value||'';this.textContent='';this.hidden='hidden' in attrs;this.disabled=false;this.selectionStart=0;this.selectionEnd=0;
        for (const [name,value] of Object.entries(attrs)) if(name.startsWith('data-')) this.dataset[camel(name.slice(5))]=value||'';
        this.id=attrs.id||'';this.className=attrs.class||'';
    }
    set value(value){this._value=String(value??'');}
    get value(){return this._value||'';}
    append(...children){children.forEach(child=>this.appendChild(child));}
    appendChild(child){child.parentElement=this;this.children.push(child);return child;}
    replaceChildren(...children){this.children.forEach(child=>child.parentElement=null);this.children=[];this.append(...children);}
    set innerHTML(value){this.replaceChildren();this.html=value;}
    get innerHTML(){return this.html||'';}
    setAttribute(name,value){this.attrs[name]=String(value);if(name.startsWith('data-'))this.dataset[camel(name.slice(5))]=String(value);}
    getAttribute(name){return name.startsWith('data-')?this.dataset[camel(name.slice(5))]:this.attrs[name];}
    hasAttribute(name){return name.startsWith('data-')?camel(name.slice(5)) in this.dataset:name in this.attrs;}
    removeAttribute(name){delete this.attrs[name];}
    toggleAttribute(name,value){if(value)this.setAttribute(name,'');else this.removeAttribute(name);}
    get offsetParent(){for(let node=this;node;node=node.parentElement)if(node.hidden)return null;return this.parentElement;}
    get scrollHeight(){return this.measuredHeight??44;}
    get nextElementSibling(){return this.parentElement?.children[this.parentElement.children.indexOf(this)+1]||null;}
    get previousElementSibling(){return this.parentElement?.children[this.parentElement.children.indexOf(this)-1]||null;}
    remove(){this.parentElement.children=this.parentElement.children.filter(child=>child!==this);this.parentElement=null;}
    addEventListener(type,listener){this.listeners[type]=listener;}
    focus(){if(this.offsetParent)document.activeElement=this;}
    scrollIntoView(){this.scrolled=true;}
    setRangeText(text,start,end){this.value=this.value.slice(0,start)+text+this.value.slice(end);this.selectionStart=this.selectionEnd=start+text.length;}
    matches(selector){
        if(selector.startsWith('#'))return this.id===selector.slice(1);
        if(selector.startsWith('.'))return this.className.split(/\s+/).includes(selector.slice(1));
        const attr=selector.match(/^\[([^=\]]+)(?:="([^"]*)")?\]$/);
        if(attr)return this.hasAttribute(attr[1])&&(attr[2]===undefined||this.getAttribute(attr[1])===attr[2]);
        return this.tagName===selector.toUpperCase();
    }
    closest(selector){for(let node=this;node;node=node.parentElement)if(node.matches(selector))return node;return null;}
    querySelectorAll(selector){
        const selectors=selector.split(/\s+/);const all=[];
        const visit=node=>{node.children.forEach(child=>{if(child.matches(selectors.at(-1))){let ancestor=child.parentElement;let i=selectors.length-2;while(i>=0&&ancestor){if(ancestor.matches(selectors[i]))i--;ancestor=ancestor.parentElement;}if(i<0)all.push(child);}visit(child);});};visit(this);return all;
    }
    querySelector(selector){return this.querySelectorAll(selector)[0]||null;}
}
function build(data){const element=new Element(data.tag,data.attrs);data.children.forEach(child=>element.appendChild(build(child)));return element;}
const tree=build(MARKUP);
global.document={activeElement:null,getElementById:id=>tree.querySelector('#'+id),querySelector:selector=>tree.querySelector(selector),querySelectorAll:selector=>tree.querySelectorAll(selector),createElement:tag=>new Element(tag)};
const field=id=>document.getElementById(id);
const prompts=()=>document.querySelectorAll('[data-template-prompt]');
const rows=()=>document.querySelectorAll('#editCriteriaContainer .template-criterion-row');
global.window={};
""".replace("MARKUP", json.dumps(MARKUP.root))
    setup = """
buildTemplatePromptNavigation();
populateTemplateOutput({});
['editTemplateName','editTemplateDescription','editTemplateItemType'].forEach(id=>field(id).value='Valid value');
prompts().forEach(input=>input.value='Prompt '+(input.dataset.promptVariables||''));
"""
    script = harness + HELPERS + POPULATE + CRITERION + SAVE + setup + "\n(async()=>{\n" + body + "\n})().catch(error=>{console.error(error);process.exit(1);});"
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_switching_sections_and_prompts_preserves_all_entered_text_without_marking_dirty():
    result = run_editor(r"""
const long='A long prompt\n'+ 'rich source text '.repeat(200);
field('editContextPrompt').value=long;field('editRefinePrompt').value='Refine {idea} using {critique}\nKeep this second line.';
field('editTemplateDescription').value='Overview\nwith a newline';addEditCriterion('First line\nSecond line');
selectTemplateEditorSection('prompts');selectTemplatePrompt('editRefinePrompt');
selectTemplateEditorSection('criteria');selectTemplateEditorSection('overview');selectTemplateEditorSection('prompts');
process.stdout.write(JSON.stringify({context:field('editContextPrompt').value,refine:field('editRefinePrompt').value,description:field('editTemplateDescription').value,criterion:rows()[0].querySelector('textarea').value,visible:prompts().filter(x=>!x.hidden).map(x=>x.id),section:templateEditorSection,dirty:templateEditorDirty,activeNav:document.querySelectorAll('[data-editor-section]').filter(x=>x.hasAttribute('aria-current')).map(x=>x.dataset.editorSection)}));
""")
    assert result["context"] == "A long prompt\n" + "rich source text " * 200
    assert result["refine"] == "Refine {idea} using {critique}\nKeep this second line."
    assert result["description"] == "Overview\nwith a newline"
    assert result["criterion"] == "First line\nSecond line"
    assert result["visible"] == ["editRefinePrompt"]
    assert result["section"] == "prompts"
    assert result["activeNav"] == ["prompts"]
    assert result["dirty"] is False


@pytest.mark.parametrize("invalid_id,expected_section", [("editTemplateDescription", "overview"), ("editSpecificPrompt", "prompts")])
def test_save_validation_reveals_exact_hidden_field_before_focusing(invalid_id, expected_section):
    result = run_editor(f"""
addEditCriterion('Originality');selectTemplateEditorSection('criteria');
field({json.dumps(invalid_id)}).value='';await saveTemplateMainPage();
process.stdout.write(JSON.stringify({{section:templateEditorSection,focus:document.activeElement?.id,hidden:field({json.dumps(invalid_id)}).hidden,invalid:field({json.dumps(invalid_id)}).getAttribute('aria-invalid'),activePrompt:activeTemplatePromptId,error:field('templateEditorFeedback').textContent,feedbackHidden:field('templateEditorFeedback').hidden}}));
""")
    assert result["section"] == expected_section
    assert result["focus"] == invalid_id
    assert result["invalid"] == "true"
    assert result["hidden"] is False and result["feedbackHidden"] is False
    if expected_section == "prompts":
        assert result["activePrompt"] == invalid_id


def test_empty_criteria_validation_activates_criteria_and_creates_focused_entry():
    result = run_editor(r"""
selectTemplateEditorSection('prompts');await saveTemplateMainPage();
process.stdout.write(JSON.stringify({section:templateEditorSection,count:field('templateCriteriaCount').textContent,focus:document.activeElement?.getAttribute('aria-label'),message:field('templateEditorFeedback').textContent}));
""")
    assert result == {"section": "criteria", "count": "1", "focus": "Comparison criterion 1", "message": "Add at least one comparison criterion."}


def test_deleting_criteria_renumbers_and_moves_focus_to_nearest_survivor_or_add_button():
    result = run_editor(r"""
selectTemplateEditorSection('criteria');['One','Two','Three'].forEach(text=>addEditCriterion(text));
rows()[1].querySelector('button').listeners.click();
const middle={values:rows().map(x=>x.querySelector('textarea').value),labels:rows().map(x=>x.querySelector('textarea').getAttribute('aria-label')),removeLabels:rows().map(x=>x.querySelector('button').getAttribute('aria-label')),numbers:rows().map(x=>x.querySelector('.template-criterion-number').textContent),focused:document.activeElement.value,count:field('templateCriteriaCount').textContent};
rows()[1].querySelector('button').listeners.click();const previous=document.activeElement.value;
rows()[0].querySelector('button').listeners.click();
process.stdout.write(JSON.stringify({middle,previous,lastFocus:document.activeElement.id,count:field('templateCriteriaCount').textContent,dirty:templateEditorDirty}));
""")
    assert result["middle"] == {"values": ["One", "Three"], "labels": ["Comparison criterion 1", "Comparison criterion 2"], "removeLabels": ["Remove criterion 1", "Remove criterion 2"], "numbers": ["01", "02"], "focused": "Three", "count": "2"}
    assert result["previous"] == "One"
    assert result["lastFocus"] == "addTemplateCriterion"
    assert result["count"] == "0" and result["dirty"] is True


def test_variable_insertion_replaces_selection_in_active_prompt_and_marks_draft_dirty():
    result = run_editor(r"""
selectTemplateEditorSection('prompts');selectTemplatePrompt('editRefinePrompt');
const input=field('editRefinePrompt');input.value='Improve OLD using critique.';input.selectionStart=8;input.selectionEnd=11;
field('templatePromptVariables').querySelectorAll('button')[0].listeners.click();
selectTemplatePrompt('editContextPrompt');selectTemplatePrompt('editRefinePrompt');
process.stdout.write(JSON.stringify({value:input.value,context:field('editContextPrompt').value,selection:input.selectionStart,focus:document.activeElement.id,dirty:templateEditorDirty,state:field('templateEditorSaveState').textContent}));
""")
    assert result == {"value": "Improve {idea} using critique.", "context": "Prompt ", "selection": 14, "focus": "editRefinePrompt", "dirty": True, "state": "Unsaved changes"}


def test_optional_prompts_extend_navigation_and_preserve_distinct_root_scope():
    result = run_editor(r"""
renderOptionalTemplatePrompts({prompts:{oracle_constraints:'Keep oracle\nconstraints',comparison_prompt:'Nested comparison'},comparison_prompt:'Legacy comparison'});
const optional=document.querySelectorAll('.optional-template-prompt');selectTemplateEditorSection('prompts');selectTemplatePrompt(optional[2].id);
const first={count:field('templatePromptsCount').textContent,options:field('templatePromptSelect').children.length,visible:prompts().filter(x=>!x.hidden).map(x=>x.value),optional:optional.map(x=>({scope:x.dataset.promptScope,key:x.dataset.promptKey,value:x.value})),selected:field('templatePromptSelect').value};
renderOptionalTemplatePrompts({});
process.stdout.write(JSON.stringify({first,remaining:field('templatePromptsCount').textContent,navCount:field('templatePromptNavigation').children.length,selected:field('templatePromptSelect').value}));
""")
    assert result["first"]["count"] == "10" and result["first"]["options"] == 10
    assert result["first"]["visible"] == ["Legacy comparison"]
    assert result["first"]["optional"] == [
        {"scope": "prompts", "key": "oracle_constraints", "value": "Keep oracle\nconstraints"},
        {"scope": "prompts", "key": "comparison_prompt", "value": "Nested comparison"},
        {"scope": "root", "key": "comparison_prompt", "value": "Legacy comparison"},
    ]
    assert result["remaining"] == "7" and result["navCount"] == 7
    assert result["selected"] == "editContextPrompt"


def test_criterion_resize_waits_for_visible_panel_and_caps_height_without_changing_value():
    result = run_editor(r"""
addEditCriterion('Long criterion\n'+ 'text '.repeat(100));const input=rows()[0].querySelector('textarea');input.measuredHeight=300;
resizeTemplateCriterion(input);const hiddenHeight=input.style.height;
selectTemplateEditorSection('criteria');const visibleHeight=input.style.height;
input.measuredHeight=22;input.listeners.input();
process.stdout.write(JSON.stringify({hiddenHeight:hiddenHeight||null,visibleHeight,smallHeight:input.style.height,value:input.value,count:field('templateCriteriaCount').textContent}));
""")
    assert result["hiddenHeight"] is None
    assert result["visibleHeight"] == "152px" and result["smallHeight"] == "44px"
    assert result["value"] == "Long criterion\n" + "text " * 100
    assert result["count"] == "1"
