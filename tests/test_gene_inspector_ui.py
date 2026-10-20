"""Candidate gene inspector behavior without browser services or model calls."""
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def render(idea, catalogue=None, history=None):
    source = (ROOT / "idea/static/js/viewer.js").read_text()
    start = source.index("function getGeneObservation(")
    functions = source[start:source.index("function getRovingTabIndex(", start)]
    start = source.index("function escapeHtml(")
    escaping = source[start:source.index("\nfunction ", start + 1)]
    payload = json.dumps({"idea": idea, "catalogue": catalogue or {"genes": []}, "history": history or []})
    script = functions + escaping + "\nconst input=" + payload + r""";
const geneCatalogue = input.catalogue, generations = input.history, opened = [];
const buttons = [];
const panel = {
 innerHTML: '',
 querySelectorAll() {
  return [...this.innerHTML.matchAll(/data-candidate-gene="(\d+)"/g)].map(match => {
   const button = {dataset:{candidateGene:match[1]}, addEventListener(event,handler){this.click=handler;}};
   buttons.push(button); return button;
  });
 }
};
const document = {getElementById:()=>panel};
openGeneCatalogue = id => opened.push(id);
renderIdeaModalGenes(input.idea);
buttons.forEach(button => button.click());
process.stdout.write(JSON.stringify({html:panel.innerHTML,opened,colors:Object.fromEntries(input.catalogue.genes.map(g=>[g.id,geneThreadColor(g.id)]))}));
"""
    result = subprocess.run(["node", "-e", script], text=True, capture_output=True, check=True, timeout=10)
    return json.loads(result.stdout)


def observed(identity, genes, **metadata):
    return {"id": identity, "gene_observation": {"status": "ok", "genes": genes, **metadata}}


def test_identity_genes_have_compact_rows_and_no_empty_or_redundant_details():
    idea = observed("a", [
        {"gene_id": "west", "text": "Western", "decisions": []},
        {"gene_id": "space", "text": "space setting", "decisions": [
            {"source": "space setting", "reason": "Carried forward unchanged", "unresolved": ""}
        ]},
    ], genotype="western; Space setting.")
    catalogue = {"genes": [{"id": "west", "label": "western"}, {"id": "space", "label": "space setting"}]}
    result = render(idea, catalogue, [[idea], [idea]])
    assert "2 genes" in result["html"]
    assert result["html"].count('class="gene-inspector-count"') == 2
    assert result["html"].count(">1 idea</span>") == 2
    assert "<details" not in result["html"]
    assert "gene-inspector-source" not in result["html"]
    assert "Source &amp; reasoning" not in result["html"]
    for color in result["colors"].values():
        assert f"--gene-color:{color}" in result["html"]
    assert result["opened"] == ["west", "space"]


def test_duplicate_mappings_share_one_row_and_count_unique_historical_carriers():
    a = observed("a", [{"gene_id": "g", "text": "deadpan tone"},
                       {"gene_id": "g", "text": "deadpan humor"}])
    b = observed("b", [{"gene_id": "g", "text": "deadpan humor"}])
    result = render(a, {"genes": [{"id": "g", "label": "deadpan humor"}]}, [[a, b], [a]])
    assert result["html"].count("<li ") == 1
    assert ">2 ideas</span>" in result["html"]
    assert "deadpan tone" in result["html"]
    assert result["opened"] == ["g"]


def test_factoring_shows_original_terms_inline_and_collects_meaningful_rationale_once():
    raw = {"status": "ok", "fingerprint": "fp", "genotype": "space western", "genes": []}
    projection = {"version": 1, "status": "ok", "revision": 8, "raw_fingerprint": "fp",
                  "genotype": "western; space setting", "genes": [
                      {"gene_id": "west", "text": "western", "source_terms": ["space western"],
                       "decisions": [{"source": "space western", "input": "space western",
                                      "reason": "Split genre and setting.", "unresolved": ""}]},
                      {"gene_id": "space", "text": "space setting", "source_terms": ["space western"],
                       "decisions": [{"source": "space western", "reason": "Carried forward unchanged"}]},
                  ]}
    idea = {"id": "a", "gene_observation": raw, "gene_projection": projection}
    result = render(idea, {"genes": [{"id": "west", "label": "western"}, {"id": "space", "label": "space setting"}]})
    assert result["html"].count("<details") == 1
    assert "<summary>Extraction details</summary>" in result["html"]
    assert result["html"].count('<span>From</span> space western') == 2
    assert "Split genre and setting." in result["html"]
    assert "Carried forward unchanged" not in result["html"]
    assert "gene-inspector-original" not in result["html"]


def test_unknown_mapping_retains_source_and_review_without_inventing_a_link_or_count():
    idea = observed("a", [{"gene_id": "missing", "text": "urban western", "decisions": [
        {"source": "urban western", "reason": "", "unresolved": "Check whether the genre is supported."}
    ]}], mapping_status="failed")
    result = render(idea)
    assert "Unmapped" in result["html"] and "Mapping incomplete" in result["html"]
    assert "Review: Check whether the genre is supported." in result["html"]
    assert "data-candidate-gene" not in result["html"]
    assert "<details" not in result["html"]
    assert result["opened"] == []


def test_nonredundant_raw_extraction_remains_available():
    result = render(observed("a", [{"text": "western"}], genotype="western; retained original nuance"))
    assert result["html"].count("<details") == 1
    assert "western; retained original nuance" in result["html"]


@pytest.mark.parametrize("case", ["mapped", "unmapped", "failed"])
def test_all_untrusted_labels_sources_reasons_and_errors_are_escaped(case):
    attack = '<img src=x onerror="alert(1)">'
    gene = {"gene_id": "g", "text": attack, "source_terms": [attack + " source"],
            "decisions": [{"source": attack + " source", "reason": attack + " reason", "unresolved": attack + " review"}]}
    idea = observed("a", [gene], genotype=attack + " raw")
    if case == "failed":
        idea["gene_observation"].update(status="failed", error=attack)
    catalogue = {"genes": [{"id": "g", "label": attack}]} if case == "mapped" else None
    result = render(idea, catalogue)
    assert "<img" not in result["html"]
    assert "&lt;img" in result["html"]
    assert 'onerror="alert(1)"' not in result["html"]
