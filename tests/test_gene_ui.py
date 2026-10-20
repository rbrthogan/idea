"""Behavioural checks for the observational catalogue, without provider requests."""
import json
import subprocess
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'idea/static/js/viewer.js').read_text()


def run_gene_js(body):
    start = SOURCE.index('function getGeneObservation(')
    end = SOURCE.index('function getRovingTabIndex(', start)
    result = subprocess.run(
        ['node', '-e', SOURCE[start:end] + '\n' + body],
        capture_output=True, text=True, check=True, timeout=10,
    )
    return json.loads(result.stdout)


def test_current_projection_drives_memberships_but_keeps_raw_and_rejects_stale():
    result = run_gene_js('''
const raw={status:'ok',fingerprint:'fp',genotype:'space western',genes:[{text:'space western',gene_id:null}]};
const projection={version:1,status:'ok',revision:2,raw_fingerprint:'fp',genotype:'space setting; western',
 genes:[{text:'space setting',gene_id:'space'},{text:'western',gene_id:'west'}]};
const a={id:'a',gene_observation:raw,gene_projection:projection};
const cat={genes:[{id:'space',label:'space setting'},{id:'west',label:'western'}]};
const view=buildGeneCatalogueView([[a]],cat);
const current=getGeneObservation(a);
projection.raw_fingerprint='stale';
process.stdout.write(JSON.stringify({current,counts:view.rows.map(r=>r.carriers.length),
 stale:getGeneObservation(a),raw}));
''')
    assert result['counts'] == [1, 1]
    assert result['current']['raw_genotype'] == 'space western'
    assert result['current']['revision'] == 2
    assert result['stale'] == result['raw']


def test_sampling_controls_follow_capabilities_and_preserve_requested_values():
    from idea.config import THINKING_MODEL_CONFIG
    start = SOURCE.index('function supportsSamplingControls(')
    end = SOURCE.index('function updateThinkingBudgetVisibility(', start)
    body = '''
let level='medium';
const elements={modelType:{value:'gemini-4-argon',selectedOptions:[{dataset:{provider:'google_generative_ai'}}]},
 creativeTemp:{value:'1.5'},topP:{value:'0.9'},creativeTempValue:{},topPValue:{},samplingControlsNotice:{},geneAnalysisModelNotice:{}};
const document={getElementById:id=>elements[id]};
const getThinkingSettings=()=>({thinkingLevel:level});
updateSamplingControls();
const argon=JSON.parse(JSON.stringify(elements));
elements.modelType.value='gemini-2.5-flash'; updateSamplingControls();
const legacy=JSON.parse(JSON.stringify(elements));
elements.modelType.value='gpt-5.6-luna'; updateSamplingControls();
const reasoningDisabled=elements.topP.disabled;
level='none'; updateSamplingControls();
process.stdout.write(JSON.stringify({argon,legacy,reasoningDisabled,noneDisabled:elements.topP.disabled}));
'''
    script = 'const window={IDEA_THINKING_MODEL_CONFIG:' + json.dumps(THINKING_MODEL_CONFIG) + '};\n'
    result = json.loads(subprocess.run(['node', '-e', script + SOURCE[start:end] + body],
                                     capture_output=True, text=True, check=True, timeout=10).stdout)
    assert result['argon']['creativeTemp']['disabled'] is True
    assert result['argon']['topPValue']['textContent'] == 'Provider controlled'
    assert result['argon']['samplingControlsNotice']['hidden'] is False
    assert 'Argon' in result['argon']['geneAnalysisModelNotice']['textContent']
    assert result['legacy']['creativeTemp']['disabled'] is False
    assert result['legacy']['creativeTempValue']['textContent'] == '1.5'
    assert result['reasoningDisabled'] is True and result['noneDisabled'] is False


def test_retired_vocabulary_without_carriers_does_not_create_empty_history_lanes():
    result = run_gene_js('''
const history=[[{id:'a',gene_observation:{status:'ok',genes:[{gene_id:'current'}]}}]];
const catalogue={genes:[{id:'retired',label:'Old bundle'},{id:'current',label:'Reusable factor'}]};
process.stdout.write(JSON.stringify({history:buildGeneHistory(history,catalogue).rows.map(r=>r.id),
 catalogue:buildGeneCatalogueView(history,catalogue).rows.map(r=>r.id)}));
''')
    assert result['history'] == ['current']
    assert set(result['catalogue']) == {'current', 'retired'}


def test_occupancy_deduplicates_survivors_and_memberships_without_losing_unknowns():
    result = run_gene_js('''
const catalogue = {genes:[{id:'g1',label:'Momentum'},{id:'g2',label:'Timing'}]};
const observed = (id, genes) => ({id,gene_observation:{status:'ok',genes}});
const a = observed('a', [{gene_id:'g1'},{gene_id:'g1'},{gene_id:null}]);
const b = observed('b', [{gene_id:'g1'},{gene_id:'missing-catalogue-entry'}]);
const c = {id:'c'};
const d = {id:'d',gene_observation:{status:'failed'}};
const e = observed('e', []);
const view=buildGeneCatalogueView([[a,c],[a,b,c,d,e]],catalogue,'history');
process.stdout.write(JSON.stringify({...view,rows:view.rows.map(row=>({id:row.id,carriers:row.carriers.map(c=>c.id)}))}));
''')
    assert result['total'] == 5
    assert result['analysed'] == 3
    assert result['missing'] == 1 and result['failed'] == 1
    assert result['unclassified'] == 2 and result['unclassifiedGenes'] == 2
    assert result['empty'] == 1
    assert result['rows'] == [{'id': 'g1', 'carriers': ['a', 'b']}, {'id': 'g2', 'carriers': []}]


def test_scopes_use_requested_population_and_latest_revision_metadata_for_history():
    result = run_gene_js('''
const catalogue={genes:[{id:'g',label:'Time'}]};
const a={id:'a',title:'A',gene_observation:{status:'ok',genes:[{gene_id:'g'}]}};
const history=[[{id:'a'}, {id:'old'}],[a,{id:'new'}]];
process.stdout.write(JSON.stringify(['current','history','generation:0'].map(scope=>{
const v=buildGeneCatalogueView(history,catalogue,scope);
return {total:v.total,analysed:v.analysed,carriers:v.rows[0].carriers.map(c=>c.id)};
})));
''')
    assert result == [
        {'total': 2, 'analysed': 1, 'carriers': ['a']},
        {'total': 3, 'analysed': 1, 'carriers': ['a']},
        {'total': 2, 'analysed': 0, 'carriers': []},
    ]


def test_failed_or_absent_observations_are_not_zero_gene_measurements():
    result = run_gene_js('''
const history=[[{id:'old'},{id:'failed',gene_observation:{status:'failed',genes:[]}}]];
const view=buildGeneCatalogueView(history,{genes:[]},'history');
process.stdout.write(JSON.stringify({analysed:view.analysed,empty:view.empty,missing:view.missing,failed:view.failed}));
''')
    assert result == {'analysed': 0, 'empty': 0, 'missing': 1, 'failed': 1}


def test_unmapped_terms_remain_visible_without_catalogue_and_deduplicate_carriers():
    result = run_gene_js('''
const a={id:'a',gene_observation:{status:'ok',mapping_status:'failed',genes:[
    {text:'slapstick comedy',gene_id:null}, {text:'third-person narration',gene_id:null},
    {text:'third-person narration',gene_id:null}]}};
const b={id:'b',gene_observation:{status:'ok',mapping_status:'failed',genes:[
    {text:'biopunk',gene_id:null}, {text:'third-person narration',gene_id:null}]}};
const history=[[a],[a,b]];
const before=JSON.stringify(history);
const view=buildGeneCatalogueView(history,null,'history');
process.stdout.write(JSON.stringify({rows:view.rows,raw:view.unclassifiedRows.map(row=>({
    label:row.label,carriers:row.carriers.map(c=>c.id),hasId:Object.hasOwn(row,'id')})),
    mappingFailed:view.mappingFailed,unclassifiedGenes:view.unclassifiedGenes,unchanged:before===JSON.stringify(history)}));
''')
    assert result['rows'] == []
    assert result['raw'] == [
        {'label': 'third-person narration', 'carriers': ['a', 'b'], 'hasId': False},
        {'label': 'biopunk', 'carriers': ['b'], 'hasId': False},
        {'label': 'slapstick comedy', 'carriers': ['a'], 'hasId': False},
    ]
    assert result['mappingFailed'] == 2
    assert result['unclassifiedGenes'] == 5
    assert result['unchanged'] is True


def test_partial_mapping_keeps_raw_terms_separate_from_identical_canonical_labels():
    result = run_gene_js('''
const catalogue={genes:[{id:'g',label:'Momentum'}]};
const a={id:'a',gene_observation:{status:'ok',mapping_status:'failed',genes:[
    {text:'Momentum',gene_id:'g'},{text:'airship',gene_id:null}]}};
const b={id:'b',gene_observation:{status:'ok',genes:[
    {text:'Momentum',gene_id:null},{text:'Mystery',gene_id:'unknown-id'}]}};
const view=buildGeneCatalogueView([[a,b]],catalogue);
process.stdout.write(JSON.stringify({mapped:view.rows.map(row=>({label:row.label,carriers:row.carriers.map(c=>c.id)})),
    raw:view.unclassifiedRows.map(row=>({label:row.label,carriers:row.carriers.map(c=>c.id)}))}));
''')
    assert result['mapped'] == [{'label': 'Momentum', 'carriers': ['a']}]
    assert result['raw'] == [
        {'label': 'airship', 'carriers': ['a']},
        {'label': 'Momentum', 'carriers': ['b']},
        {'label': 'Mystery', 'carriers': ['b']},
    ]


def test_unmapped_rows_follow_scope_and_disappear_when_latest_observation_is_mapped():
    result = run_gene_js('''
const catalogue={genes:[{id:'g',label:'Time'}]};
const earlier={id:'a',gene_observation:{status:'ok',genes:[{text:'Time',gene_id:null}]}};
const latest={id:'a',gene_observation:{status:'ok',genes:[{text:'Time',gene_id:'g'}]}};
process.stdout.write(JSON.stringify(['generation:0','current','history'].map(scope=>{
    const view=buildGeneCatalogueView([[earlier],[latest]],catalogue,scope);
    return {raw:view.unclassifiedRows.length,mapped:view.rows[0].carriers.length};
})));
''')
    assert result == [{'raw': 1, 'mapped': 0}, {'raw': 0, 'mapped': 1}, {'raw': 0, 'mapped': 1}]


def test_main_genes_view_renders_failed_mapping_and_candidate_clicks_safely():
    escape_start = SOURCE.index('function escapeHtml(')
    escape_end = SOURCE.index('function getTournamentEntryForGeneration(', escape_start)
    result = run_gene_js(SOURCE[escape_start:escape_end] + '''
let geneCatalogue={genes:[]},geneScope='current',selectedGeneId=null;
const unsafe='</th><script>alert(1)</script>';
const candidate={id:'story',title:'Story <img src=x onerror=alert(1)>',gene_observation:{
    status:'ok',mapping_status:'failed',genes:[{text:unsafe,gene_id:null}]}};
const generations=[[candidate]];
let opened=null;
function getIdeaData(idea){return idea;}
function showIdeaModal(idea){opened=idea;}
const container={
    innerHTML:'',buttons:[],querySelector(){return null;},
    querySelectorAll(selector){
        if(selector!=='[data-unmapped-carrier]')return [];
        this.buttons=[...this.innerHTML.matchAll(/data-unmapped-gene="(\\d+)" data-unmapped-carrier="(\\d+)"/g)].map(match=>({
            dataset:{unmappedGene:match[1],unmappedCarrier:match[2]},listeners:{},
            addEventListener(event,callback){this.listeners[event]=callback;}
        }));
        return this.buttons;
    }
};
const document={getElementById:id=>id==='evolution-genes-container'?container:null};
getGeneExplorerState().mode='catalogue';
renderGeneCatalogue(generations);
container.buttons[0].listeners.click();
process.stdout.write(JSON.stringify({html:container.innerHTML,openedId:opened.id,
    openedRaw:opened.gene_observation.genes[0],catalogue:geneCatalogue}));
''')
    assert 'Unmapped genes' in result['html']
    assert 'Catalogue matching failed for 1 candidate.' in result['html']
    assert '&lt;/th&gt;&lt;script&gt;alert(1)&lt;/script&gt;' in result['html']
    assert 'Story &lt;img src=x onerror=alert(1)&gt;' in result['html']
    assert '<script>' not in result['html'] and '<img' not in result['html']
    assert 'No catalogued genes yet' not in result['html']
    assert 'data-gene-row=' not in result['html']
    assert result['openedId'] == 'story'
    assert result['openedRaw']['gene_id'] is None
    assert result['catalogue'] == {'genes': []}


def test_legacy_load_clears_catalogue_but_partial_live_payload_preserves_it():
    result = run_gene_js('''
let geneCatalogue={revision:1,genes:[{id:'g',label:'Time'}]},selectedGeneId='g';
const safeSignature=JSON.stringify;
const partialChanged=applyGeneCatalogue({});
const retained=geneCatalogue.genes.length;
const resetChanged=applyGeneCatalogue({},true);
process.stdout.write(JSON.stringify({partialChanged,retained,resetChanged,geneCatalogue,selectedGeneId}));
''')
    assert result == {'partialChanged': False, 'retained': 1, 'resetChanged': True, 'geneCatalogue': None, 'selectedGeneId': None}


def test_observation_only_live_update_refreshes_history_and_selected_candidate():
    start = SOURCE.index('function applyHistoryIfNew(')
    end = SOURCE.index('function schedulePersistEvolutionState(', start)
    result = run_gene_js('''
let geneCatalogue=null,selectedGeneId=null,lastGeneObservationSignature='',lastRenderedHistoryVersion=7;
let currentModalIdea={id:'a'},renders=0,selectedObservation=null;
const safeSignature=JSON.stringify;
const document={getElementById:()=>({hidden:false})};
function renderGenerations(){renders++;}
function renderIdeaModalGenes(idea){selectedObservation=idea.gene_observation;}
''' + SOURCE[start:end] + '''
const data={history_version:7,history:[[{id:'a'}]]};
applyHistoryIfNew(data);
const unchanged=applyHistoryIfNew(data);
data.history[0][0].gene_observation={status:'ok',genes:[{gene_id:'g'}]};
const changed=applyHistoryIfNew(data);
process.stdout.write(JSON.stringify({unchanged,changed,renders,selectedObservation}));
''')
    assert result['unchanged'] is False and result['changed'] is True
    assert result['renders'] == 2
    assert result['selectedObservation']['status'] == 'ok'


def test_saved_export_preserves_catalogue_ids_and_candidate_memberships():
    start = SOURCE.index('function downloadResults(')
    end = SOURCE.index('// Global history state', start)
    script = '''
let currentEvolutionData=null,tournamentHistory=[],specificPrompts=[],breedingPrompts=[];
const geneCatalogue={version:1,revision:5,genes:[{id:'g1',label:'Timing'}]};
const requests=[];
const document={getElementById:()=>null};
const prompt=()=> 'gene-export.json';const alert=()=>{};
console.log=()=>{};console.error=()=>{};
const fetch=async (url,options)=>{requests.push(JSON.parse(options.body));return {ok:true,json:async()=>({})};};
''' + SOURCE[start:end] + '''
downloadResults({history:[[{id:'a',gene_observation:{status:'ok',genes:[{text:'Timing',gene_id:'g1'}]}}]]});
setImmediate(()=>process.stdout.write(JSON.stringify(requests[0].data)));
'''
    result = subprocess.run(['node', '-e', script], capture_output=True, text=True, check=True, timeout=10)
    data = json.loads(result.stdout)
    assert data['gene_catalogue']['revision'] == 5
    assert data['gene_catalogue']['genes'][0]['id'] == data['history'][0][0]['gene_observation']['genes'][0]['gene_id']


def test_gene_history_counts_turnover_without_inventing_absence_or_future_membership():
    result = run_gene_js('''
const catalogue={genes:[{id:'g',label:'Time'},{id:'h',label:'Humour'},{id:'unused',label:'Unused'}]};
const candidate=(id, genes, fitness)=>({id,fitness,gene_observation:{status:'ok',genes:genes.map(gene_id=>({gene_id}))}});
const a=candidate('a',['g','g'],0.2),b=candidate('b',['h'],0.8),c=candidate('c',['g','h'],0.4);
const history=[[a,b],[a,c,c],[{id:'a'},c],[b]];
const before=JSON.stringify(history);
const result=buildGeneHistory(history,catalogue,(idea)=>idea.fitness??null);
process.stdout.write(JSON.stringify({...result,unchanged:before===JSON.stringify(history)}));
''')
    g, h = result['rows']
    assert [point['count'] for point in g['points']] == [1, 2, 1, 0]
    assert [point['analysed'] for point in g['points']] == [2, 2, 1, 1]
    assert [point['complete'] for point in g['points']] == [True, True, False, True]
    assert g['unique'] == 2 and g['peak'] == 2 and g['first'] == 0
    assert h['unique'] == 2
    assert g['points'][1]['mean'] == pytest.approx(0.3)
    assert g['points'][3]['mean'] is None
    assert [point['lost'] for point in result['summaries']] == [None, 0, None, None]
    assert [point['added'] for point in result['summaries']] == [2, 0, 0, 0]
    assert result['unchanged'] is True


def test_gene_history_fitness_uses_only_the_requested_snapshot_and_excludes_missing_values():
    start = SOURCE.index('function getFitnessPlotSelectionGeneration(')
    end = SOURCE.index('function summarizeFitnessPlotValues(', start)
    # Keep the actual shared generation-aware metric implementation.
    helpers = '''
function toFiniteDiversityNumber(v){return typeof v==='number'&&Number.isFinite(v)?v:null;}
function toFiniteDiversityInteger(v){return Number.isInteger(v)?v:null;}
function firstFiniteDiversityValue(...values){return values.find(v=>toFiniteDiversityNumber(v)!==null)??null;}
'''
    result = run_gene_js(helpers + SOURCE[start:end] + '''
const candidate={id:'a',fitness:999,selection_metrics:{evaluation_generation:2,fitness:0.9},
    generation_metrics:{'0':{generation:0,fitness:0.25}},
    gene_observation:{status:'ok',genes:[{gene_id:'g'}]}};
const result=buildGeneHistory([[candidate],[candidate],[candidate]],{genes:[{id:'g',label:'Time'}]});
process.stdout.write(JSON.stringify(result.rows[0].points.map(p=>({mean:p.mean,measured:p.measured}))));
''')
    assert result == [{'mean': 0.25, 'measured': 1}, {'mean': None, 'measured': 0}, {'mean': 0.9, 'measured': 1}]



def test_gene_threads_keep_500_genes_and_100_generations_on_one_axis():
    escape_start = SOURCE.index('function escapeHtml(')
    escape_end = SOURCE.index('function getTournamentEntryForGeneration(', escape_start)
    result = run_gene_js(SOURCE[escape_start:escape_end] + '''
let selectedGeneId=null;
const catalogue={genes:Array.from({length:500},(_,i)=>({id:`g${i}`,label:`<unsafe ${i}>`}))};
const history=Array.from({length:100},(_,gen)=>Array.from({length:20},(_,i)=>({id:`s${gen}-${i}`,
    gene_observation:{status:'ok',genes:catalogue.genes.filter((_,j)=>j%20===i).map(g=>({gene_id:g.id}))}})));
const model=buildGeneHistory(history,catalogue);
const threads=buildGeneThreads(model,750);
const html=renderGeneHistory(model,{search:''});
process.stdout.write(JSON.stringify({tracks:threads.tracks.length,points:threads.tracks[0].row.points.length,
    groups:(html.match(/data-gene-track=/g)||[]).length,ticks:threads.generations,
    bounds:threads.tracks.every(t=>t.y>=threads.top&&t.y<=threads.bottom),
    finite:!html.includes('NaN')&&!html.includes('Infinity'),escaped:html.includes('&lt;unsafe'),
    noTiles:!html.includes('gene-map-tile'),noFilters:!html.includes('gene-history-filter')}));
''')
    assert result == {'tracks': 500, 'points': 100, 'groups': 500, 'ticks': [0,20,40,59,79,99],
                      'bounds': True, 'finite': True, 'escaped': True, 'noTiles': True, 'noFilters': True}


def test_gene_threads_have_stable_order_and_do_not_bridge_unobserved_runs_with_solid_paths():
    result = run_gene_js('''
const catalogue={genes:[{id:'late',label:'Late'},{id:'small',label:'Small'},{id:'large',label:'Large'},{id:'unused',label:'Unused'}]};
const c=(id,genes)=>({id,gene_observation:{status:'ok',genes:genes.map(gene_id=>({gene_id}))}});
const history=[[c('a',['large','small']),c('b',['large'])],[c('c',['late'])],[c('d',['large','late'])]];
const model=buildGeneThreads(buildGeneHistory(history,catalogue));
process.stdout.write(JSON.stringify({order:model.tracks.map(t=>t.row.id),
    largeRuns:(model.tracks[0].path.match(/Z/g)||[]).length,
    lateFirst:model.tracks.find(t=>t.row.id==='late').row.first,
    hasUnused:model.tracks.some(t=>t.row.id==='unused'),ticks:model.generations}));
''')
    assert result == {'order': ['large','small','late'], 'largeRuns': 2,
                      'lateFirst': 1, 'hasUnused': False, 'ticks': [0,1,2]}


def test_fitness_trend_breaks_at_missing_data_and_preserves_true_zero():
    result = run_gene_js('''
const points=[{v:.5},{v:null},{v:0},{v:.8},{v:NaN},{v:.4}];
process.stdout.write(JSON.stringify({path:geneTrendPath(points,p=>p.v),
    one:geneTrendPath([{v:0}],p=>p.v)}));
''')
    assert result['path'].count('M') == 3
    assert result['path'].count('L') == 1
    assert 'NaN' not in result['path']
    assert result['one'] == 'M113.00,74.00'


def test_gene_paths_break_on_unrelated_reappearance_but_connect_survivors_and_children():
    result = run_gene_js(r'''
const c=(id,parents=[])=>({id,parent_ids:parents,
 gene_observation:{status:'ok',genes:[{gene_id:'g'}]}});
const catalogue={genes:[{id:'g',label:'Shared'}]};
const history=[[c('a')],[c('b')],[c('b')],[c('c',['b']),c('d',['b'])],
 [c('e',['c','d'])],[],[c('f',['e'])]];
const model=buildGeneThreads(buildGeneHistory(history,catalogue),726);
const track=model.tracks[0];
const spans=track.path.split('Z').filter(Boolean).map(path=>{
 const xs=[...path.matchAll(/(-?\d+\.\d+),-?\d+\.\d+/g)].map(m=>Number(m[1]));
 return [Math.min(...xs),Math.max(...xs)];
});
process.stdout.write(JSON.stringify({links:track.row.points.map(p=>[p.survived,p.parentLinked]),spans}));
''')
    assert result['links'] == [[0, 0], [0, 0], [1, 0], [0, 2], [0, 1], [0, 0], [0, 0]]
    assert result['spans'] == [[12, 109], [115, 512], [612, 712]]


def test_gene_paths_fill_equal_generation_bands_including_endpoints():
    result = run_gene_js(r'''
const c=ids=>({id:'a',gene_observation:{status:'ok',genes:ids.map(gene_id=>({gene_id}))}});
const catalogue={genes:['first','last','two','all'].map(id=>({id,label:id}))};
const model=buildGeneThreads(buildGeneHistory([
    [c(['first','two','all'])],[c(['two','all'])],[c(['all'])],[c(['last','all'])]
],catalogue),426);
const spans=Object.fromEntries(model.tracks.map(t=>{
    const xs=[...t.path.matchAll(/(-?\d+\.\d+),-?\d+\.\d+/g)].map(m=>Number(m[1]));
    return [t.row.id,[Math.min(...xs),Math.max(...xs)]];
}));
const single=buildGeneThreads(buildGeneHistory([[c(['all'])]],catalogue),426);
process.stdout.write(JSON.stringify({step:model.step,centres:[0,1,2,3].map(model.x),spans,
    single:[single.tracks[0].start,single.tracks[0].end,single.x(0)]}));
''')
    assert result == {'step': 100, 'centres': [62,162,262,362],
                      'spans': {'first':[12,112], 'last':[312,412],
                                'two':[12,212], 'all':[12,412]},
                      'single':[12,412,212]}


def test_search_keeps_all_threads_and_escapes_the_selected_name():
    escape_start = SOURCE.index('function escapeHtml(')
    escape_end = SOURCE.index('function getTournamentEntryForGeneration(', escape_start)
    result = run_gene_js(SOURCE[escape_start:escape_end] + '''
let selectedGeneId=null;
const model=buildGeneHistory([[{id:'a',gene_observation:{status:'ok',genes:[{gene_id:'a'},{gene_id:'b'}]}}]],
    {genes:[{id:'a',label:'<script>'},{id:'b',label:'Time',aliases:['Clock']}]});
const html=renderGeneHistory(model,{search:'CLOCK'});
process.stdout.write(JSON.stringify({groups:(html.match(/data-gene-track=/g)||[]).length,
    dimmed:(html.match(/gene-thread is-search-dimmed/g)||[]).length,
    matched:html.includes('1 matching · all 2 visible'),escaped:html.includes('&lt;script&gt;'),raw:html.includes('<script>')}));
''')
    assert result == {'groups':2,'dimmed':1,'matched':True,'escaped':True,'raw':False}


def test_pinned_gene_and_story_selection_block_hover_until_cleared():
    result = run_gene_js('''
const state={};
changeGeneExplorerFocus(state,'a',0);
changeGeneExplorerFocus(state,'b',1,'pin');
const blocked=!changeGeneExplorerFocus(state,'c',2)&&state.focusId==='b'&&state.focusGeneration===1;
changeGeneExplorerFocus(state,'c',2,'keyboard');
const keyboard=state.pinnedGeneId==='c';
clearGeneExplorerSelection(state);
const cleared=changeGeneExplorerFocus(state,'a',0);
state.candidateId='story';
const dnaBlocked=!changeGeneExplorerFocus(state,'b',1);
changeGeneExplorerFocus(state,'b',1,'pin');
process.stdout.write(JSON.stringify({blocked,keyboard,cleared,dnaBlocked,pinClearsStory:state.candidateId===null}));
''')
    assert all(result.values())


def test_linked_idea_dna_preserves_historical_carriers_and_projection():
    escape_start = SOURCE.index('function escapeHtml(')
    escape_end = SOURCE.index('function getTournamentEntryForGeneration(', escape_start)
    result = run_gene_js(SOURCE[escape_start:escape_end] + '''
let selectedGeneId=null;
const getIdeaData=x=>x;
const a={id:'a',title:'<A>',gene_observation:{status:'ok',fingerprint:'fp',genes:[{gene_id:'old'}]},
 gene_projection:{version:1,status:'ok',genotype:'space; west',raw_fingerprint:'fp',genes:[{gene_id:'space'},{gene_id:'west'}]}};
const b={id:'b',title:'B',gene_observation:{status:'ok',genes:[{gene_id:'west'}]}};
const c={id:'c',title:'C'};
const gens=[[a,b,c],[b]];
const history=buildGeneHistory(gens,{genes:['space','west'].map(id=>({id,label:id}))});
const state={search:'',candidateId:'a',focusGeneration:0,focusId:'west'};
const html=renderGeneHistory(history,state,gens);
const retained=state.candidateId==='a';
const unavailable=geneExplorerHeading(history.rows[0],[{id:'c',idea:c}],{candidateId:'c'}).summary;
state.focusGeneration=1; renderGeneHistory(history,state,gens);
process.stdout.write(JSON.stringify({dna:[...geneStoryIds(geneHistoryCarriers(gens,'west')[0])],retained,
 historyRetained:state.candidateId==='a',unavailable,
 dnaLines:(html.match(/is-dna"/g)||[]).length,storyButtons:(html.match(/data-gene-story=/g)||[]).length,
 escaped:html.includes('&lt;A&gt;'),noDisclosures:!html.includes('<details'),
 panels:['gene-thread-stories','gene-thread-canvas','gene-thread-detail'].every(id=>html.includes(`id="${id}"`))}));
''')
    assert result == {'dna':['space','west'], 'retained':True, 'historyRetained':True,
                      'unavailable':'Gene analysis unavailable', 'dnaLines':2, 'storyButtons':2,
                      'escaped':True, 'noDisclosures':True, 'panels':True}


def test_occupancy_counts_separate_selected_latest_and_unique_history():
    escape_start = SOURCE.index('function escapeHtml(')
    escape_end = SOURCE.index('function getTournamentEntryForGeneration(', escape_start)
    result = run_gene_js(SOURCE[escape_start:escape_end] + '''
const c=(id,ids)=>({id,gene_observation:{status:'ok',genes:ids.map(gene_id=>({gene_id}))}});
const model=buildGeneHistory([[c('a',['g']),c('b',[])],[c('b',[])],[c('a',['g']),c('c',['g'])]],{genes:[{id:'g',label:'Gene'}]});
const row=model.rows[0];
const old=renderGeneTrend(row,'occupancy',1),now=renderGeneTrend(row,'occupancy',2);
row.points[2].complete=false;
const incomplete=renderGeneTrend(row,'occupancy',1);
process.stdout.write(JSON.stringify({unique:row.unique,
 selected:old.includes('<small>Gen 1</small><strong>0 / 1</strong>'),
 current:old.includes('<small>Now</small><strong>2 / 2</strong>'),
 ever:old.includes('<small>Ever</small><strong>2</strong>'),
 noDuplicate:!now.includes('<small>Gen 2</small>'),
 incomplete:incomplete.includes('<small>Now</small><strong>2 known</strong>')}));
''')
    assert result == {'unique':2,'selected':True,'current':True,'ever':True,'noDuplicate':True,'incomplete':True}


def test_history_carriers_deduplicate_survivors_and_order_first_membership():
    result = run_gene_js('''
const c=(id,genes)=>({id,gene_observation:{status:'ok',genes:genes.map(gene_id=>({gene_id}))}});
const first=c('early',['g','a']), later=c('later',['g']), absent=c('early',[]);
const gens=[[first,c('other',[])],[later,first],[absent,later,c('new',['g'])]];
const carriers=geneHistoryCarriers(gens,'g');
process.stdout.write(JSON.stringify(carriers.map(c=>({id:c.id,first:c.firstGeneration,last:c.generation,genes:[...geneStoryIds(c)]}))));
''')
    assert result == [
        {'id':'early','first':0,'last':1,'genes':['g','a']},
        {'id':'later','first':1,'last':2,'genes':['g']},
        {'id':'new','first':2,'last':2,'genes':['g']},
    ]
