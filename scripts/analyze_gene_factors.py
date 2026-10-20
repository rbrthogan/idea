#!/usr/bin/env python3
"""Apply reviewed one-to-many gene projections offline, preserving source evidence.

This is an analysis tool, not a breeding or catalogue-maintenance policy.
Writes analysis.json and review.csv in a separate output directory. Never edits
source checkpoints or invokes providers. Every source label gets a review row.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

FORM_GENES = {'flash fiction', 'microfiction', 'drabble'}


def analyze(source, specification):
    catalogue = {g['id']: g for g in source['gene_catalogue']['genes']}
    rules = specification.get('rules', {})
    overrides = specification.get('raw_overrides', {})
    unknown = set(rules) - {g['label'] for g in catalogue.values()}
    if unknown:
        raise ValueError(f'Rules refer to unknown catalogue labels: {sorted(unknown)}')
    for targets in [*rules.values(), *overrides.values()]:
        if not isinstance(targets, list) or not targets or any(not isinstance(t, str) or not t.strip() for t in targets):
            raise ValueError('Each rule needs at least one nonempty ingredient')
    candidates = {}
    for generation in source['history']:
        for candidate in generation:
            candidates[str(candidate['id'])] = candidate
    original, projected, evidence = {}, {}, {}
    for cid, candidate in candidates.items():
        before, after, rows = set(), set(), []
        for assignment in candidate.get('gene_observation', {}).get('genes', []):
            gene = catalogue[assignment['gene_id']]
            label, raw = gene['label'], assignment['text']
            targets = overrides.get(raw, rules.get(label, [label]))
            before.add(label)
            after.update(targets)
            rows.append({'source_gene_id': gene['id'], 'source_label': label,
                         'raw_text': raw, 'ingredients': targets})
        original[cid], projected[cid], evidence[cid] = before, after, rows

    def metrics(mapping, exclude_forms=False):
        sets = {cid: genes - FORM_GENES if exclude_forms else genes for cid, genes in mapping.items()}
        occupancy = Counter(g for genes in sets.values() for g in genes)
        total = sum(occupancy.values())
        inherited, child_total, per_child = 0, 0, []
        for cid, genes in sets.items():
            parents = [sets[str(pid)] for pid in candidates[cid].get('parent_ids', []) if str(pid) in sets]
            if not parents or not genes:
                continue
            shared = genes & set().union(*parents)
            inherited += len(shared)
            child_total += len(genes)
            per_child.append({'candidate_id': cid, 'shared': len(shared), 'total': len(genes)})
        return {'unique_ingredients': len(occupancy), 'singletons': sum(n == 1 for n in occupancy.values()),
                'shared_ingredients': sum(n >= 2 for n in occupancy.values()), 'assignments': total,
                'shared_assignment_fraction': sum(n for n in occupancy.values() if n >= 2) / total if total else None,
                'mean_ingredients_per_story': total / len(sets) if sets else None,
                'child_parent_overlap_fraction': inherited / child_total if child_total else None,
                'children_measured': len(per_child), 'per_child': per_child,
                'occupancy': dict(occupancy.most_common())}
    def trajectory(mapping):
        seen = set()
        result = []
        for generation, population in enumerate(source['history']):
            counts = Counter(g for c in population for g in mapping[str(c['id'])])
            current = set(counts)
            result.append({'generation': generation, 'active': len(current), 'new': len(current-seen),
                           'singletons': sum(n == 1 for n in counts.values()), 'occupancy': dict(counts)})
            seen |= current
        return result
    review = []
    for gene in catalogue.values():
        label = gene['label']
        targets = rules.get(label, [label])
        action = 'keep' if targets == [label] else 'add broader facet' if label in targets else 'factor' if len(targets) > 1 else 'normalize'
        raw_rules = {term: targets for term, targets in overrides.items() if term in [label, *gene.get('aliases', [])]}
        review.append({'source_gene_id': gene['id'], 'source_label': label, 'ingredients': targets,
                       'action': 'review raw alias' if raw_rules else action, 'raw_overrides': raw_rules,
                       'note': specification.get('review_notes', {}).get(label, '')})
    return {'analysis_only': True, 'candidate_count': len(candidates), 'review': review,
            'before': metrics(original), 'after': metrics(projected),
            'before_excluding_format': metrics(original, True), 'after_excluding_format': metrics(projected, True),
            'before_generations': trajectory(original), 'after_generations': trajectory(projected),
            'candidates': {cid: {'title': c['idea'].get('title') if isinstance(c['idea'], dict) else '',
                                 'original_genotype': c.get('gene_observation', {}).get('genotype'),
                                 'original_observation': c.get('gene_observation'),
                                 'ingredients': sorted(projected[cid]), 'evidence': evidence[cid]}
                           for cid, c in candidates.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('rules', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source_bytes, rule_bytes = args.source.read_bytes(), args.rules.read_bytes()
    result = analyze(json.loads(source_bytes), json.loads(rule_bytes))
    result['source_sha256'] = hashlib.sha256(source_bytes).hexdigest()
    result['rules_sha256'] = hashlib.sha256(rule_bytes).hexdigest()
    result['source_path'] = str(args.source.resolve())
    if args.source.resolve() in {(args.output/name).resolve() for name in ('analysis.json', 'review.csv')}:
        raise ValueError('Output would overwrite the source checkpoint')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'analysis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    with (args.output/'review.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source_gene_id', 'source_label', 'action', 'ingredients', 'raw_overrides', 'note'])
        writer.writeheader()
        writer.writerows({**row, 'ingredients': '; '.join(row['ingredients']),
                          'raw_overrides': json.dumps(row['raw_overrides'])} for row in result['review'])
    assert args.source.read_bytes() == source_bytes, 'Source checkpoint changed'
    print(json.dumps({k: {a:b for a,b in result[k].items() if a not in {'occupancy', 'per_child'}}
                      for k in ('before', 'after', 'before_excluding_format', 'after_excluding_format')}, indent=2))


if __name__ == '__main__':
    main()
