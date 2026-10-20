"""Exercise run-panel lifecycle with the real viewer functions in a small DOM."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def section(source, start, end):
    offset = source.index(start)
    return source[offset:source.index(end, offset)]


def run_transition(body):
    html = (ROOT / "idea/static/html/viewer.html").read_text()
    viewer = (ROOT / "idea/static/js/viewer.js").read_text()
    new_mode = section(html, "    async function showNewEvolutionMode()", "    async function showEvolutionMode")
    reset_start = "function clearTokenCounts()" if "function clearTokenCounts()" in viewer else "function resetUIState()"
    reset = section(viewer, reset_start, "// Function to create the progress bar")
    display = section(viewer, "function displayTokenCounts(tokenCounts)", "// Function to show the token details modal")
    harness = r"""
class Element {
    constructor(id='') { this.id=id; this.children=[]; this.parentNode=null; this.dataset={}; this.style={}; this.value=''; this.listeners={}; this.textContent=''; }
    appendChild(node) { if(node.parentNode) node.remove(); this.children.push(node); node.parentNode=this; return node; }
    insertBefore(node,reference) { if(node.parentNode) node.remove(); const index=this.children.indexOf(reference); this.children.splice(index<0?this.children.length:index,0,node);node.parentNode=this; }
    remove() { if(this.parentNode) this.parentNode.children=this.parentNode.children.filter(child=>child!==this);this.parentNode=null; }
    get nextSibling() { return this.parentNode?.children[this.parentNode.children.indexOf(this)+1]||null; }
    set innerHTML(html) { this.children.forEach(node=>node.parentNode=null);this.children=[];this.html=html; for(const match of html.matchAll(/id="([^"]+)"/g))this.appendChild(new Element(match[1])); }
    get innerHTML() { return this.html||''; }
    querySelector(selector) { const id=selector.startsWith('#')?selector.slice(1):selector; for(const child of this.children){if(child.id===id)return child;const found=child.querySelector(selector);if(found)return found;}return null; }
    addEventListener(type,listener) { this.listeners[type]=listener; }
}
const main=new Element('mainContent');
global.document={getElementById:id=>main.querySelector('#'+id),querySelector:()=>null,createElement:()=>new Element()};
const add=id=>main.appendChild(new Element(id));
['templateEntry','settingsSection','newEvolutionBtn','evolutionSelect','generations-container','diversity-plot-section','ideaType','downloadButton','contextDisplay','contextContainer'].forEach(add);
document.getElementById('ideaType').value='drabble';
document.getElementById('evolutionSelect').value='saved-run';
document.getElementById('generations-container').innerHTML='<p>Previous population</p>';
let currentMode='evolution',currentEvolutionName='Previous experiment',lastTokenCountsSignature='',latestDiagnosticsTokenCounts=null,persistStateTimeout=null;
let treeResets=0,inspectorCloses=0,progressCreated=0,stoppedTimers=0,modalTokenCounts=null,loadTemplates=async()=>{};
global.window={currentUser:{uid:'test-user'},IdeaWorkbench:{closeInspector(){inspectorCloses++;}}};
global.localStorage={removeItem(){}};
function setDisplayById(id,value){document.getElementById(id).style.display=value;}
function resetEvolutionTreeView(){treeResets++;}
function loadExistingTemplates(){return loadTemplates();}
function setTemplateFlowMode(){}
function stopElapsedTimeUpdater(){stoppedTimers++;}
function createProgressBar(){progressCreated++;add('progress-container');}
function addActivityLogItem(){}
function updateEvolutionStatusIndicator(){}
function safeSignature(value){return JSON.stringify(value);}
function getDiagnosticsSummary(){return {totalCount:0};}
function showTokenDetailsModal(value){modalTokenCounts=value;}
console.log=()=>{};
const oldCounts={total:12000,total_input:9000,total_output:3000,cost:{total_cost:12.68,estimated_total_cost:20}};
const freshCounts={total:23,total_input:15,total_output:8,cost:{total_cost:0.0003}};
add('progress-container');
"""
    script = harness + new_mode + reset + display + "\n(async()=>{\n" + body + "\n})().catch(error=>{console.error(error);process.exit(1);});"
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_new_experiment_removes_previous_cost_and_progress_before_templates_load():
    result = run_transition(r"""
displayTokenCounts(oldCounts);
let releaseTemplates;loadTemplates=()=>new Promise(resolve=>releaseTemplates=resolve);
const transition=showNewEvolutionMode();
const during={costPresent:!!document.getElementById('token-counts-container'),progressPresent:!!document.getElementById('progress-container'),templateDisplay:document.getElementById('templateEntry').style.display,selector:document.getElementById('evolutionSelect').value,population:document.getElementById('generations-container').innerHTML};
releaseTemplates();await transition;
process.stdout.write(JSON.stringify({during,currentMode,selectedTemplate:document.getElementById('ideaType').value,progressCreated,treeResets}));
""")
    assert result["during"] == {
        "costPresent": False, "progressPresent": False, "templateDisplay": "block",
        "selector": "", "population": "",
    }
    assert result["currentMode"] == "new"
    assert result["selectedTemplate"] == "drabble"
    assert result["progressCreated"] == 0
    assert result["treeResets"] == 1


def test_new_run_recreates_cost_markup_and_details_with_only_current_run_values():
    result = run_transition(r"""
displayTokenCounts(oldCounts);
const oldPanel=document.getElementById('token-counts-container');
resetUIState();
const reset={oldDetached:oldPanel.parentNode===null,costPresent:!!document.getElementById('token-counts-container'),diagnostics:latestDiagnosticsTokenCounts};
displayTokenCounts(freshCounts);
const panel=document.getElementById('token-counts-container');
const details=panel?.querySelector('#token-details-btn');details?.listeners.click();
process.stdout.write(JSON.stringify({reset,newPanel:panel!==oldPanel,cost:panel?.querySelector('#token-cost-value')?.textContent,total:panel?.querySelector('#token-total-value')?.textContent,estimateDisplay:panel?.querySelector('#token-estimated-total-wrap')?.style.display,detailsBound:!!details?.listeners.click,modalTokenCounts,progressCreated,downloadDisabled:document.getElementById('downloadButton').disabled}));
""")
    assert result["reset"] == {"oldDetached": True, "costPresent": False, "diagnostics": None}
    assert result["newPanel"] is True
    assert result["cost"] == "$0.0003"
    assert result["total"] == "23 tokens"
    assert result["estimateDisplay"] == "none"
    assert result["detailsBound"] is True
    assert result["modalTokenCounts"]["total"] == 23
    assert result["modalTokenCounts"]["cost"]["total_cost"] == 0.0003
    assert result["progressCreated"] == 1
    assert result["downloadDisabled"] is True


def test_reopening_identical_saved_cost_after_new_experiment_recreates_the_panel():
    result = run_transition(r"""
displayTokenCounts(oldCounts);
const oldPanel=document.getElementById('token-counts-container');
await showNewEvolutionMode();
const clearedDiagnostics=latestDiagnosticsTokenCounts===null;
displayTokenCounts(oldCounts);
const panel=document.getElementById('token-counts-container');
process.stdout.write(JSON.stringify({recreated:!!panel&&panel!==oldPanel,cost:panel?.querySelector('#token-cost-value')?.textContent,clearedDiagnostics}));
""")
    assert result == {"recreated": True, "cost": "$12.6800", "clearedDiagnostics": True}
