import copy
import pytest
from scripts.analyze_gene_factors import analyze


def sample():
    genes = [{'id': 'a', 'label': 'space western'}, {'id': 'b', 'label': 'dinosaur western'},
             {'id': 'c', 'label': 'deadpan humor', 'aliases': ['deadpan tone']}]
    def candidate(cid, gene, text, parents):
        return {'id': cid, 'idea': {'title': cid}, 'parent_ids': parents,
                'gene_observation': {'genotype': text, 'genes': [{'gene_id': gene, 'text': text}]}}
    a = candidate('parent', 'a', 'space western', [])
    b = candidate('child', 'b', 'dinosaur western', ['parent'])
    c = candidate('other', 'c', 'deadpan tone', [])
    return {'gene_catalogue': {'genes': genes}, 'history': [[a, c], [a, b]]}


def test_factor_analysis_retains_evidence_and_deduplicates_survivors():
    source = sample()
    original = copy.deepcopy(source)
    result = analyze(source, {'rules': {'space western': ['space', 'western'],
                                      'dinosaur western': ['dinosaurs', 'western']},
                              'raw_overrides': {'deadpan tone': ['deadpan tone']}})
    assert source == original
    assert result['candidate_count'] == 3
    assert result['after']['occupancy']['western'] == 2
    assert result['before']['child_parent_overlap_fraction'] == 0
    assert result['after']['child_parent_overlap_fraction'] == .5
    assert result['candidates']['parent']['original_genotype'] == 'space western'
    assert result['candidates']['child']['evidence'][0]['source_label'] == 'dinosaur western'
    assert result['candidates']['other']['ingredients'] == ['deadpan tone']
    assert result['after_generations'][1]['occupancy']['western'] == 2


def test_review_rejects_unknown_labels_and_empty_replacements():
    with pytest.raises(ValueError, match='unknown'):
        analyze(sample(), {'rules': {'unseen': ['something']}})
    with pytest.raises(ValueError, match='nonempty'):
        analyze(sample(), {'rules': {'space western': []}})
