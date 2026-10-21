"""Manual rankings follow human evidence, independently of evolution fitness."""
from tests.test_rater_async_recovery import run_rater

SETUP = """
const originalQueryAll = document.querySelectorAll;
document.querySelectorAll = selector => selector === '.sortable-header' ? [] : originalQueryAll(selector);
createEloChart = () => {};
ideasDb = {
    favourite: {id:'favourite',title:'Human favourite',ratings:{auto:1400,manual:1548},manual_match_count:3,auto_match_count:1,selection_metrics:{fitness:.1}},
    automatic: {id:'automatic',title:'Automated favourite',ratings:{auto:1700,manual:1484},manual_match_count:3,auto_match_count:10,selection_metrics:{fitness:.9}}
};
"""


def test_manual_ranking_is_default_and_does_not_sort_by_auto_fitness_or_matches():
    result = run_rater(SETUP + """
await showRanking();
process.stdout.write(JSON.stringify({type:currentRatingType,sort:currentSortColumn,ids:window.rankedIdeas.map(i=>i.id),stats:nodes.ratingStats.innerHTML}));
""")
    assert result['type'] == 'manual'
    assert result['sort'] == 'rating'
    assert result['ids'] == ['favourite', 'automatic']
    assert 'Provisional manual ranking' in result['stats']
    assert '3 saved comparisons' in result['stats']
    assert '3 comparisons per idea' in result['stats']


def test_switching_rating_type_resets_sort_to_selected_rating():
    result = run_rater(SETUP + """
await showRanking();
sortTable('fitness','desc');
const fitnessOrder=window.rankedIdeas.map(i=>i.id);
toggleRatingType('auto');const autoOrder=window.rankedIdeas.map(i=>i.id);
currentSortColumn='matches';toggleRatingType('manual');
process.stdout.write(JSON.stringify({fitnessOrder,autoOrder,manualOrder:window.rankedIdeas.map(i=>i.id),sort:currentSortColumn}));
""")
    assert result == {'fitnessOrder': ['automatic', 'favourite'], 'autoOrder': ['automatic', 'favourite'],
                      'manualOrder': ['favourite', 'automatic'], 'sort': 'rating'}


def test_unplayed_candidates_are_unrated_not_a_neutral_score_above_losing_candidates():
    result = run_rater(SETUP + """
delete ideasDb.favourite;
ideasDb.unplayed={id:'unplayed',title:'Not compared',ratings:{auto:1800,manual:1500},manual_match_count:0};
await showRanking();const descending=window.rankedIdeas.map(i=>i.id);
sortTable('rating','asc');
process.stdout.write(JSON.stringify({descending,ascending:window.rankedIdeas.map(i=>i.id),cell:getIdeaRatingCellHtml(ideasDb.unplayed),value:getRatingValueByType(ideasDb.unplayed,'manual'),stats:nodes.ratingStats.innerHTML}));
""")
    assert result['descending'] == result['ascending'] == ['automatic', 'unplayed']
    assert 'Unrated' in result['cell']
    assert result['value'] is None
    assert '1 of 2 ideas compared' in result['stats']
    assert '0–3 comparisons per idea' in result['stats']


def test_rating_mode_selects_matching_results_and_swiss_progress_includes_current_pair():
    result = run_rater(SETUP + """
toggleRatingMode('auto');const automatic=currentRatingType;
toggleRatingMode('manual');
setManualSwissStatus({round_number:4,remaining_pairs_in_round:0});
process.stdout.write(JSON.stringify({automatic,manual:currentRatingType,status:nodes.manualSwissStatus.textContent}));
""")
    assert result == {'automatic': 'auto', 'manual': 'manual',
                      'status': 'Round 4 · 1 comparison left in this round (including this pair)'}


def test_equal_scores_share_rank_and_ranks_survive_sorting_other_columns():
    result = run_rater(SETUP + """
ideasDb.tied={...ideasDb.favourite,id:'tied',title:'Another favourite'};
ideasDb.unplayed={id:'unplayed',title:'Not compared',manual_match_count:0};
await showRanking();
const ranks=Object.fromEntries(window.rankedIdeas.map(i=>[i.id,i.table_rating_rank]));
sortTable('title','asc');sortTable('rank','asc');
process.stdout.write(JSON.stringify({ranks,ids:window.rankedIdeas.map(i=>i.id),row:buildRankingRowHtml(window.rankedIdeas[0],0)}));
""")
    assert result['ranks'] == {'favourite': 1, 'tied': 1, 'automatic': 3, 'unplayed': None}
    assert set(result['ids'][:2]) == {'favourite', 'tied'}
    assert result['ids'][2:] == ['automatic', 'unplayed']
    assert '<td>1</td>' in result['row']
