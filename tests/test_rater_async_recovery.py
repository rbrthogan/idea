"""Run the real rater in Node with delayed responses to verify selection ownership."""
import json
import subprocess
from pathlib import Path

RATER = Path(__file__).resolve().parents[1] / "idea/static/js/rater.js"


def run_rater(body):
    harness = r"""
const nodes = {};
function node() {
    return { value: '', textContent: '', hidden: false, disabled: false, style: {}, dataset: {}, options: [], listeners: {},
        set innerHTML(value) { this.html = value; if (value.includes('option')) this.options = [{value:''}]; },
        get innerHTML() { return this.html || ''; },
        addEventListener(type, fn) { this.listeners[type] = fn; },
        setAttribute(name, value) { this[name] = value; }, getAttribute(name) { return this[name]; },
        replaceChildren() { this.html = ''; }, appendChild(child) { this.options.push(child); },
        querySelector() { return this.child || (this.child = node()); },
        classList: {add(){},remove(){},toggle(){}}, focus(){} };
}
const getNode = id => nodes[id] || (nodes[id] = node());
const buttons = [node(), node(), node()];
const events = {};
global.window = { currentUser: {uid:'owner-a'}, addEventListener(type, fn) { events[type] = fn; }, IdeaWorkbench: {renderPrimaryArtifact(){}} };
global.document = { getElementById: getNode, querySelector: getNode, querySelectorAll: () => buttons,
    addEventListener(){}, createElement: node, readyState:'complete' };
console.log = () => {}; console.error = () => {}; console.warn = () => {};
const requests = [];
global.fetch = (url, options) => new Promise((resolve, reject) => requests.push({url, options, resolve, reject}));
const respond = (index, payload, ok = true) => requests[index].resolve({ok, status:ok?200:500, json:async()=>payload});
const tick = () => new Promise(resolve => setImmediate(resolve));
const pair = suffix => ({idea_a:{id:'a'+suffix,title:'A'+suffix,content:'alpha'},idea_b:{id:'b'+suffix,title:'B'+suffix,content:'beta'},pair_token:'token-'+suffix});
const history = suffix => ({data:{history:[[pair(suffix).idea_a,pair(suffix).idea_b]]}});
const selectReady = suffix => { beginRatingSelection('run'+suffix); ratingSelectionLoading=false; ideasDb={['a'+suffix]:pair(suffix).idea_a,['b'+suffix]:pair(suffix).idea_b}; };
const readyPair = suffix => { selectReady(suffix); currentPair=Object.values(ideasDb); pairContext={...captureRatingContext(),pairToken:pair(suffix).pair_token}; updateManualRatingControls(); };
"""
    script = harness + RATER.read_text() + "\n(async()=>{\n" + body + "\n})().catch(error=>{process.stderr.write(error.stack);process.exit(1);});"
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, timeout=10)
    return json.loads(result.stdout)


def test_pair_refresh_clears_previous_pair_and_disables_voting_until_loaded():
    result = run_rater("""
readyPair('1');
const pending=refreshPair();
const loading={pair:currentPair, disabled:buttons.every(button=>button.disabled), retry:nodes.retryManualPair.disabled};
await vote('A');
respond(0,pair('fresh'));await pending;
process.stdout.write(JSON.stringify({loading,requests:requests.length,ids:currentPair.map(x=>x.id),enabled:buttons.every(button=>!button.disabled)}));
""")
    assert result == {"loading": {"pair": None, "disabled": True, "retry": True}, "requests": 1,
                      "ids": ["afresh", "bfresh"], "enabled": True}


def test_pair_response_and_failure_from_superseded_request_are_ignored():
    result = run_rater("""
selectReady('1');const old=refreshPair();const fresh=refreshPair();
respond(1,pair('fresh'));await fresh;requests[0].reject(Error('offline'));await old;
process.stdout.write(JSON.stringify({ids:currentPair.map(x=>x.id),feedback:nodes.manualRatingFeedback.hidden,enabled:buttons.every(x=>!x.disabled)}));
""")
    assert result == {"ids": ["afresh", "bfresh"], "feedback": True, "enabled": True}


def test_pair_response_cannot_cross_run_or_account_boundary():
    result = run_rater("""
selectReady('1');const old=refreshPair();selectReady('2');
const next=refreshPair();respond(1,pair('2'));await next;respond(0,pair('old'));await old;
const afterRun=currentPair.map(x=>x.id);
const previousOwner=refreshPair();window.currentUser={uid:'owner-b'};respond(2,pair('secret'));await previousOwner;
await vote('A');
process.stdout.write(JSON.stringify({afterRun,keys:Object.keys(ideasDb),pair:currentPair,requests:requests.length}));
""")
    assert result["afterRun"] == ["a2", "b2"]
    assert "asecret" not in result["keys"]
    assert result["pair"] is None
    assert result["requests"] == 3


def test_vote_keeps_original_pair_and_ignores_old_completion_after_selection_change():
    result = run_rater("""
readyPair('1');const oldVote=vote('A');
readyPair('2');const newVote=vote('B');
respond(0,{updated_elos:{a2:9999}});await oldVote;
const pending={disabled:buttons.every(x=>x.disabled),voteInFlight,elo:ideasDb.a2.ratings};
requests[1].reject(Error('lost response'));await newVote;
process.stdout.write(JSON.stringify({pending,first:JSON.parse(requests[0].options.body),second:JSON.parse(requests[1].options.body),pair:currentPair,feedback:nodes.manualRatingFeedback.hidden,disabled:buttons.every(x=>x.disabled),retry:nodes.retryManualPair.disabled}));
""")
    assert result["pending"] == {"disabled": True, "voteInFlight": True}
    assert result["first"] == {"idea_a_id": "a1", "idea_b_id": "b1", "outcome": "A", "evolution_id": "run1", "pair_token": "token-1"}
    assert result["second"] == {"idea_a_id": "a2", "idea_b_id": "b2", "outcome": "B", "evolution_id": "run2", "pair_token": "token-2"}
    assert result["pair"] is None and result["feedback"] is False
    assert result["disabled"] and not result["retry"]


def test_current_evolution_resolves_progress_then_loads_canonical_record_before_pair():
    result = run_rater("""
const loading=loadCurrentEvolution();respond(0,{evolution_id:'current-run',is_running:false});await tick();
const before=JSON.parse(JSON.stringify(requests.map(x=>x.url)));
respond(1,history('current'));await tick();
const pairBody=JSON.parse(requests[2].options.body);respond(2,pair('current'));await loading;
process.stdout.write(JSON.stringify({before,pairBody,id:currentEvolutionId,ids:currentPair.map(x=>x.id)}));
""")
    assert result == {"before": ["/api/progress", "/api/evolution/current-run"],
                      "pairBody": {"evolution_id": "current-run"}, "id": "current-run", "ids": ["acurrent", "bcurrent"]}


def test_saved_run_load_ignores_older_response_even_after_selecting_same_run_again():
    result = run_rater("""
const old=loadEvolution('A');const middle=loadEvolution('B');const fresh=loadEvolution('A');
respond(2,history('new'));await tick();respond(3,pair('new'));await fresh;
respond(0,history('old'));respond(1,history('other'));await Promise.all([old,middle]);
process.stdout.write(JSON.stringify({id:currentEvolutionId,keys:Object.keys(ideasDb),requests:requests.length,enabled:buttons.every(x=>!x.disabled)}));
""")
    assert result == {"id": "A", "keys": ["anew", "bnew"], "requests": 4, "enabled": True}


def test_saved_list_and_current_run_responses_ignore_previous_account():
    result = run_rater("""
const list=loadEvolutions();const current=loadCurrentEvolution();
window.currentUser={uid:'owner-b'};events['idea-auth-cleared']();
respond(0,[{id:'private-old-run'}]);respond(1,{evolution_id:'private-old-run',is_running:false});await Promise.all([list,current]);
process.stdout.write(JSON.stringify({id:currentEvolutionId,options:nodes.evolutionSelect?.options||[],requests:requests.length,disabled:buttons.every(x=>x.disabled)}));
""")
    assert result == {"id": None, "options": [{"value": ""}], "requests": 2, "disabled": True}


def test_reset_completion_does_not_reselect_old_run():
    result = run_rater("""
global.confirm=()=>true;global.alert=()=>{throw Error('stale notice');};
readyPair('1');nodes.evolutionSelect=getNode('evolutionSelect');nodes.evolutionSelect.value='run1';
const reset=resetRatings('manual');readyPair('2');respond(0,{message:'reset'});await reset;
process.stdout.write(JSON.stringify({id:currentEvolutionId,requests:requests.length,ids:currentPair.map(x=>x.id)}));
""")
    assert result == {"id": "run2", "requests": 1, "ids": ["a2", "b2"]}


def test_auto_rating_progress_and_result_cannot_mutate_new_selection():
    result = run_rater("""
readyPair('1');nodes.evolutionSelect.value='run1';nodes.modelSelect=getNode('modelSelect');nodes.modelSelect.value='cheap-model';
getAutoTournamentEstimate=()=>({tournamentCount:1,targetRounds:1,estimatedTotalMatches:1,fullTournamentRounds:1});
updateProgressBar=()=>{};let tableUpdates=0;updateRankingsTable=()=>{tableUpdates++;};
const running=startAutoRatingBtn.listeners.click();readyPair('2');
respond(0,{is_running:true,version:1,ideas:[{id:'wrong'}]});respond(1,{ideas:[{id:'wrong'}],results:[]});await running;await tick();
process.stdout.write(JSON.stringify({tableUpdates,keys:Object.keys(ideasDb),id:currentEvolutionId,poll:autoRatingProgressPollTimer,autoRatingInProgress}));
""")
    assert result == {"tableUpdates": 0, "keys": ["a2", "b2"], "id": "run2", "poll": None, "autoRatingInProgress": False}
