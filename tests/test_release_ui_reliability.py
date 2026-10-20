"""Behavioral regression coverage for browser recovery and authentication."""
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_js(script):
    result = subprocess.run(['node', '-e', script], text=True, capture_output=True, check=True, timeout=10)
    return json.loads(result.stdout)


def section(file, start, end):
    source = (ROOT / 'idea/static/js' / file).read_text()
    offset = source.index(start)
    return source[offset:source.index(end, offset)]


@pytest.mark.parametrize("outcome", ["missing", "unreadable", "failed", "unknown", "ready"])
def test_provider_readiness_blocks_start_until_a_usable_selected_key_is_confirmed(outcome):
    helper = section('auth_logic.js', 'let providerStatusCheckVersion', '// Export for other modules')
    result = run_js('const outcome=' + json.dumps(outcome) + ';\n' + '''
const nodes={startButton:{disabled:false},startAutoRating:{disabled:false},apiKeyAlert:{style:{}}};
global.document={getElementById:id=>nodes[id]||null};
let resolveStatus,rejectStatus;
global.window={IdeaWorkbench:{selectedProvider:()=> 'google_generative_ai',updateProviderReadiness(){},loadProviderSettings:()=>new Promise((resolve,reject)=>{resolveStatus=resolve;rejectStatus=reject;})}};
''' + helper + '''
(async()=>{
 const loading=checkApiKeyStatus();
 const pending=nodes.startButton.disabled;
 if(outcome==='failed')rejectStatus(new Error('offline'));
 else resolveStatus(outcome==='unknown'?{}:{api_key_missing:false,api_keys:{google_generative_ai:{configured:outcome==='ready',error_code:outcome==='unreadable'?'stored_key_unreadable':null},openai:{configured:true}}});
 await loading;
 process.stdout.write(JSON.stringify({pending,start:nodes.startButton.disabled,auto:nodes.startAutoRating.disabled,title:nodes.startButton.title,hasApiKey:window.hasApiKey,banner:nodes.apiKeyAlert.style.display}));
})().catch(error=>{console.error(error);process.exit(1);});
''')
    assert result['pending'] is True
    assert result['start'] is (outcome != 'ready')
    assert result['auto'] is (outcome != 'ready')
    assert result['hasApiKey'] is (outcome == 'ready')
    if outcome == 'unreadable':
        assert 'replacement in Settings' in result['title']
    if outcome == 'failed':
        assert 'Open Settings to retry' in result['title']


def test_viewer_settings_exposes_replacement_without_hiding_unreadable_warning():
    helper = section('viewer.js', 'let settingsStatusVersion', '/**\n * Save API Key')
    result = run_js('''
const ids=['view-settings','saveSettingsBtn','apiKeyInputContainer','apiKeyConfiguredContainer','settingsStatus','apiKeyProviderSelect','apiKeyInput'];
const nodes=Object.fromEntries(ids.map(id=>[id,{style:{setProperty(k,v){this[k]=v;}},value:id==='apiKeyProviderSelect'?'google_generative_ai':'',setAttribute(){},removeAttribute(){}}]));
global.document={getElementById:id=>nodes[id]||null};
global.window={IdeaWorkbench:{setProviderSettings(){}}};
const loadExecutionSettings=async()=>({api_keys:{google_generative_ai:{configured:false,source:'unreadable',error_code:'stored_key_unreadable',message:'Enter a replacement. The saved key has not been changed.'}}});
function showStatusInSettings(message,type){nodes.settingsStatus.textContent=message;nodes.settingsStatus.hidden=false;nodes.settingsStatus.style.display='block';}
''' + helper + '''
(async()=>{const refreshed=await checkSettingsStatus();process.stdout.write(JSON.stringify({refreshed,input:nodes.apiKeyInputContainer.style.display,configured:nodes.apiKeyConfiguredContainer.style.display,disabled:nodes.saveSettingsBtn.disabled,hidden:nodes.settingsStatus.hidden,message:nodes.settingsStatus.textContent}));})().catch(error=>{console.error(error);process.exit(1);});
''')
    assert result['refreshed']['api_keys']['google_generative_ai']['configured'] is False
    assert result['input'] == 'block' and result['configured'] == 'none'
    assert result['disabled'] is False and result['hidden'] is False
    assert 'replacement' in result['message']


@pytest.mark.parametrize("storage_full", [False, True])
def test_failed_run_never_announces_success_and_retains_saved_checkpoint(storage_full):
    helper = section('viewer.js', 'function markEvolutionComplete', '// Function to show the context modal')
    result = run_js('const storageFull=' + json.dumps(storage_full) + ';\n' + '''
const elements = {};
for (const id of ['evolution-spinner','evolution-status-text','progress-container','progress-bar-activity','forceStopButton']) {
 const classes = new Set(['running','spinning']);
 elements[id] = {textContent:'',style:{},classList:{remove(...xs){xs.forEach(x=>classes.delete(x));},add(...xs){xs.forEach(x=>classes.add(x));}},classes};
}
global.document = {getElementById:id=>elements[id]};
global.window = {currentUser:{uid:'test-user'}};
const records = [], stored = {};
let stopped = false, refreshed = false, evolutionStartTime = 10;
const formatDuration = ()=>'1:00';
const stopElapsedTimeUpdater = ()=>{stopped=true;};
const addActivityLogItem = (text,type)=>records.push({text,type});
const loadHistory = ()=>{refreshed=true;};
const localStorage = {setItem:(k,v)=>{if(storageFull)throw Error('Quota exceeded');stored[k]=v;}};
''' + helper + '''
markEvolutionComplete(false,true,'checkpoint-1','Provider unavailable');
process.stdout.write(JSON.stringify({status:elements['evolution-status-text'].textContent,classes:[...elements['progress-container'].classes],records,stored,stopped,refreshed}));
''')
    assert result['status'] == 'Failed · checkpoint saved'
    assert result['classes'] == ['spinning', 'failed']
    assert result['records'][0]['type'] == 'error'
    assert 'Completed' not in result['records'][0]['text']
    if storage_full:
        assert result['stored'] == {}
    else:
        assert result['stored']['idea.test-user.lastCheckpointId'] == 'checkpoint-1'
    assert result['stopped'] and result['refreshed']


def test_auth_refresh_preserves_request_headers_and_retries_only_once():
    policy = section('auth_logic.js', 'function isInternalRequestUrl', '// 3. Create Login Modal')
    wrapper = section('auth_logic.js', 'const originalFetch = window.fetch;', '// 8. Check API Key Status')
    result = run_js('''
const calls=[], refreshes=[];
let idToken=null, signedOut=false;
const user={getIdToken:async force=>{refreshes.push(!!force);return force?'fresh':'cached';}};
global.window={location:{origin:'https://idea.test'},currentUser:user,fetch:async (url,options)=>{
 calls.push({url:typeof url==='string'?url:url.url,headers:Object.fromEntries(new Headers(options.headers))});
 return {status:calls.length===1?401:200};
}};
function completeSignOut(){signedOut=true;}
''' + policy + wrapper + '''
(async()=>{
 await window.fetch(new Request('https://idea.test/api/history',{headers:{'X-Trace':'keep'}}));
 await window.fetch('https://other.test/api',{});
 process.stdout.write(JSON.stringify({calls,refreshes,signedOut}));
})().catch(e=>{console.error(e);process.exit(1);});
''')
    assert len(result['calls']) == 3
    assert result['calls'][0]['headers'] == {'authorization': 'Bearer cached', 'x-trace': 'keep'}
    assert result['calls'][1]['headers'] == {'authorization': 'Bearer fresh', 'x-trace': 'keep'}
    assert result['calls'][2]['headers'] == {}
    assert result['refreshes'] == [False, True]
    assert not result['signedOut']


def test_auth_does_not_send_cached_credentials_when_token_refresh_fails():
    policy = section('auth_logic.js', 'function isInternalRequestUrl', '// 3. Create Login Modal')
    wrapper = section('auth_logic.js', 'const originalFetch = window.fetch;', '// 8. Check API Key Status')
    result = run_js('''
let idToken='stale',calls=0;
global.window={location:{origin:'https://idea.test'},currentUser:{getIdToken:async()=>{throw Error('offline');}},fetch:async()=>{calls++;}};
function completeSignOut(){}
''' + policy + wrapper + '''
window.fetch('/api/start-evolution').catch(error=>process.stdout.write(JSON.stringify({calls,idToken,message:error.message})));
''')
    assert result['calls'] == 0
    assert result['idToken'] is None
    assert 'session' in result['message']


def test_manual_vote_cannot_submit_same_pair_twice_while_pending():
    from tests.test_rater_async_recovery import run_rater

    result = run_rater("""
readyPair('1');pairContext.pairToken='current-pair-token';const first=vote('A');await vote('B');
const disabled=buttons.every(button=>button.disabled);
respond(0,{updated_match_counts:{}});await tick();respond(1,{...pair('next'),pair_token:'next-pair-token'});await first;
process.stdout.write(JSON.stringify({submissions:requests.filter(x=>x.url==='/api/submit-rating').length,disabled,enabled:buttons.every(x=>!x.disabled),ids:currentPair.map(x=>x.id)}));
""")
    assert result == {'submissions': 1, 'disabled': True, 'enabled': True, 'ids': ['anext', 'bnext']}


def test_validation_errors_remain_readable():
    helper = section('viewer.js', 'function apiErrorMessage', 'function showErrorMessage')
    result = run_js(helper + '''
process.stdout.write(JSON.stringify(apiErrorMessage({detail:[{loc:['body','popSize'],msg:'Must be at least 2'}]},'failed')));
''')
    assert result == 'popSize: Must be at least 2'


def test_auth_rejects_response_from_an_account_that_signed_out_mid_request():
    policy = section('auth_logic.js', 'function isInternalRequestUrl', '// 3. Create Login Modal')
    wrapper = section('auth_logic.js', 'const originalFetch = window.fetch;', '// 8. Check API Key Status')
    result = run_js('''
let idToken=null,release;
const user={uid:'a',getIdToken:async()=> 'a-token'};
global.window={location:{origin:'https://idea.test'},currentUser:user,fetch:async()=>new Promise(resolve=>{release=resolve;})};
function completeSignOut(){}
''' + policy + wrapper + '''
(async()=>{
 const request=window.fetch('/api/history');await new Promise(resolve=>setImmediate(resolve));
 window.currentUser={uid:'b'};release({status:200,privateAccount:'a'});
 try{await request;process.stdout.write(JSON.stringify({rejected:false}));}
 catch(error){process.stdout.write(JSON.stringify({rejected:true,message:error.message}));}
})();
''')
    assert result['rejected']
    assert 'account changed' in result['message']


def test_browser_checkpoint_backup_is_scoped_to_the_user_and_tolerates_full_storage():
    helper = section('viewer.js', 'function schedulePersistEvolutionState', 'function normalizeMarkdownStructure')
    result = run_js('''
let callback,persistStateTimeout=null,currentEvolutionId='run-1',currentEvolutionName='Story';
const geneCatalogue={version:1,revision:2,genes:[{id:'g',label:'Time'}]};
const LOCAL_STORAGE_PERSIST_DELAY_MS=1;
global.window={currentUser:{uid:'alice'}};
const writes=[];const warnings=[];
const localStorage={setItem:(key,value)=>writes.push({key,value:JSON.parse(value)})};
const setTimeout=fn=>{callback=fn;return 1;};const clearTimeout=()=>{};
console.warn=(...args)=>warnings.push(args.map(String));
''' + helper + '''
schedulePersistEvolutionState({history:[[ {id:'private-idea'} ]]});
window.currentUser={uid:'bob'};callback();const switchedWrites=writes.length;
schedulePersistEvolutionState({history:[]});callback();
localStorage.setItem=()=>{throw Error('Quota exceeded');};
schedulePersistEvolutionState({history:[]});callback();
process.stdout.write(JSON.stringify({switchedWrites,writes,warnings}));
''')
    assert result['switchedWrites'] == 0
    assert result['writes'][0]['key'] == 'idea.bob.currentEvolutionData'
    assert result['writes'][0]['value']['evolution_id'] == 'run-1'
    assert result['writes'][0]['value']['gene_catalogue']['genes'][0]['id'] == 'g'
    assert len(result['warnings']) == 1


def test_template_defaults_do_not_change_low_cost_evolution_default():
    # Both defaults are visible in the catalog rather than hidden request fallbacks.
    from idea.config import DEFAULT_MODEL, LLM_MODELS
    assert DEFAULT_MODEL == 'gemini-3.5-flash-lite'
    assert any(model['id'] == DEFAULT_MODEL for model in LLM_MODELS)


def test_cached_history_cannot_restore_archived_execution_actions():
    helper = section('viewer.js', 'function overlayHistoryItemsWithCachedEvolution', 'async function loadHistory')
    result = run_js('''
const cached={evolution_id:'prime-run',history:[[],[]],current_generation:1,status:'complete'};
const normalizeCachedEvolutionState=value=>value;
const getCachedEvolutionState=()=>cached;
const getCachedEvolutionId=value=>value?.evolution_id;
const cachedEvolutionIsNewer=()=>true;
const repairPersistedEvolutionFromClientCache=()=>{throw Error('must not repair archived data');};
''' + helper + '''
process.stdout.write(JSON.stringify(overlayHistoryItemsWithCachedEvolution([{id:'prime-run',execution_archived:true,can_continue:false,can_use_config:false}])));
''')
    assert result[0]['can_continue'] is False
    assert result[0]['can_use_config'] is False


def test_budget_pause_explains_cumulative_cap_without_claiming_completion():
    helper = section('viewer.js', 'function markEvolutionComplete', '// Function to show the context modal')
    result = run_js('''
const status = {}, messages = [];
const document = {getElementById: id => id === 'evolution-status-text' ? status : null};
const stopElapsedTimeUpdater = () => {}, evolutionStartTime = null;
const addActivityLogItem = (text, type) => messages.push({text, type});
''' + helper + '''
markEvolutionComplete(true, true, null, null, 'budget');
process.stdout.write(JSON.stringify({status:status.textContent,messages}));
''')
    assert result['status'] == 'Paused · budget reached'
    assert 'larger budget' in result['messages'][0]['text']
    assert 'Completed' not in result['messages'][0]['text']


def test_manual_pair_without_server_token_cannot_enable_or_submit_a_vote():
    from tests.test_rater_async_recovery import run_rater

    result = run_rater("""
selectReady('1');const loading=refreshPair();
const incomplete=pair('old');delete incomplete.pair_token;
respond(0,incomplete);await loading;await vote('A');
process.stdout.write(JSON.stringify({pair:currentPair,context:pairContext,disabled:buttons.every(button=>button.disabled),requests:requests.length,feedback:nodes.manualRatingFeedback.querySelector('span').textContent,retryDisabled:nodes.retryManualPair.disabled}));
""")
    assert result['pair'] is None and result['context'] is None
    assert result['disabled'] is True and result['requests'] == 1
    assert 'Reload pair' in result['feedback']
    assert result['retryDisabled'] is False


def test_manual_vote_sends_exact_server_pair_token_and_replaces_it_after_success():
    from tests.test_rater_async_recovery import run_rater

    result = run_rater("""
selectReady('1');const loading=refreshPair();
respond(0,{...pair('old'),pair_token:'opaque-token-one'});await loading;
const firstToken=pairContext.pairToken;const saving=vote('A');
const submitted=JSON.parse(requests[1].options.body);
respond(1,{updated_match_counts:{}});await tick();
respond(2,{...pair('next'),pair_token:'opaque-token-two'});await saving;
process.stdout.write(JSON.stringify({firstToken,submitted,nextToken:pairContext.pairToken,ids:currentPair.map(idea=>idea.id),enabled:buttons.every(button=>!button.disabled)}));
""")
    assert result['firstToken'] == 'opaque-token-one'
    assert result['submitted'] == {
        'idea_a_id': 'aold', 'idea_b_id': 'bold', 'outcome': 'A',
        'evolution_id': 'run1', 'pair_token': 'opaque-token-one',
    }
    assert result['nextToken'] == 'opaque-token-two'
    assert result['ids'] == ['anext', 'bnext'] and result['enabled'] is True


def test_conflicted_manual_vote_reloads_explains_no_save_and_allows_retry_with_new_token():
    from tests.test_rater_async_recovery import run_rater

    result = run_rater("""
selectReady('1');const loading=refreshPair();
respond(0,{...pair('old'),pair_token:'outdated-token'});await loading;
const saving=vote('A');
requests[1].resolve({ok:false,status:409,json:async()=>({message:'Pair changed'})});await tick();
const pending={disabled:buttons.every(button=>button.disabled),pair:currentPair,request:requests[2]?.url};
respond(2,{...pair('current'),pair_token:'current-token'});await saving;
const recovered={enabled:buttons.every(button=>!button.disabled),pairToken:pairContext.pairToken,feedback:nodes.manualRatingFeedback.querySelector('span').textContent,feedbackVisible:!nodes.manualRatingFeedback.hidden,oldMatches:ideasDb.aold.manual_match_count||0};
const retry=vote('B');const retryPayload=JSON.parse(requests[3].options.body);
respond(3,{updated_match_counts:{}});await tick();respond(4,{...pair('next'),pair_token:'next-token'});await retry;
process.stdout.write(JSON.stringify({pending,recovered,retryPayload,votes:requests.filter(request=>request.url==='/api/submit-rating').length,feedbackHidden:nodes.manualRatingFeedback.hidden,enabled:buttons.every(button=>!button.disabled)}));
""")
    assert result['pending'] == {'disabled': True, 'pair': None, 'request': '/api/get-efficient-pair'}
    assert result['recovered']['enabled'] is True
    assert result['recovered']['pairToken'] == 'current-token'
    assert result['recovered']['feedbackVisible'] is True
    assert 'not saved' in result['recovered']['feedback']
    assert 'reloaded' in result['recovered']['feedback']
    assert result['recovered']['oldMatches'] == 0
    assert result['retryPayload'] == {
        'idea_a_id': 'acurrent', 'idea_b_id': 'bcurrent', 'outcome': 'B',
        'evolution_id': 'run1', 'pair_token': 'current-token',
    }
    assert result['votes'] == 2 and result['feedbackHidden'] is True and result['enabled'] is True


@pytest.mark.parametrize('switch_account', [False, True])
def test_conflict_recovery_cannot_overwrite_new_run_or_account(switch_account):
    from tests.test_rater_async_recovery import run_rater

    result = run_rater('const switchAccount=' + json.dumps(switch_account) + ';\n' + """
selectReady('1');const loading=refreshPair();
respond(0,{...pair('old'),pair_token:'old-token'});await loading;
const saving=vote('A');
requests[1].resolve({ok:false,status:409,json:async()=>({message:'Pair changed'})});await tick();
if(switchAccount)window.currentUser={uid:'owner-b'};
selectReady('2');const newLoading=refreshPair();
respond(3,{...pair('new'),pair_token:'new-owner-run-token'});await newLoading;
respond(2,{...pair('obsolete'),pair_token:'obsolete-token'});await saving;
process.stdout.write(JSON.stringify({owner:pairContext.owner,evolution:pairContext.evolutionId,token:pairContext.pairToken,ids:currentPair.map(idea=>idea.id),feedbackHidden:nodes.manualRatingFeedback.hidden,requests:requests.length,enabled:buttons.every(button=>!button.disabled),obsoleteAdded:!!ideasDb.aobsolete}));
""")
    assert result == {
        'owner': 'owner-b' if switch_account else 'owner-a',
        'evolution': 'run2', 'token': 'new-owner-run-token', 'ids': ['anew', 'bnew'],
        'feedbackHidden': True, 'requests': 4, 'enabled': True, 'obsoleteAdded': False,
    }


def test_conflict_reload_failure_never_claims_a_new_pair_was_loaded():
    from tests.test_rater_async_recovery import run_rater

    result = run_rater("""
selectReady('1');const loading=refreshPair();
respond(0,{...pair('old'),pair_token:'outdated-token'});await loading;
const saving=vote('A');
requests[1].resolve({ok:false,status:409,json:async()=>({message:'Pair changed'})});await tick();
requests[2].reject(Error('Connection lost'));await saving;
process.stdout.write(JSON.stringify({pair:currentPair,feedback:nodes.manualRatingFeedback.querySelector('span').textContent,disabled:buttons.every(button=>button.disabled),retryDisabled:nodes.retryManualPair.disabled}));
""")
    assert result['pair'] is None and result['disabled'] is True
    assert result['retryDisabled'] is False
    assert 'not saved' in result['feedback']
    assert 'Reload pair' in result['feedback']
    assert 'has been reloaded' not in result['feedback']
