"""Offline tests of the exact prompts reaching each runtime role."""

from unittest.mock import Mock, patch

import pytest

from idea.llm import Breeder, Critic, Formatter, Ideator, Oracle
from idea.models import Idea
from idea.prompts.loader import get_builtin_template_snapshot, get_prompts_from_dict
from idea.template_runtime import extract_title_content_envelope, render_prompt
from idea.gene_prompts import GENE_INGREDIENT_GUIDANCE


def test_envelope_extraction_preserves_inline_indentation_and_line_endings():
    body = "    if ready:\r\n        run({\"x\": 1})\r\n"
    assert extract_title_content_envelope("Title: Example\r\nContent: " + body) == {
        "title": "Example", "content": body,
    }


def template_data():
    return {
        'name': 'Contract test', 'description': 'Offline prompt contract',
        'version': '1.0.0', 'author': 'Tests', 'created_date': '2026-10-18',
        'metadata': {'item_type': 'stories'},
        'requirements': ['REQ: preserve the literal {idea} and {"code":{"x":1}}.'],
        'output': {'title': 'TITLE: a short title containing {title}.',
                   'content': 'BODY: a narrative, preserving {critique} as text.',
                   'word_count': {'min': 100, 'max': 100}},
        'comparison_criteria': ['Originality'],
        'prompts': {
            'context': 'CONTEXT: make concepts with notation {x}.',
            'specific_prompt': 'DIRECTION: {context_pool}',
            'format': 'FORMAT_MARKER: {input_text}',
            'critique': 'CRITIQUE: {idea}',
            'refine': 'REFINE: {idea}\nFeedback: {critique}',
            'breed': 'BREED_EDITED: {ideas}',
            'genotype_encode': 'GENOTYPE: {idea_content}',
            'oracle_constraints': 'ORACLE_GOAL: preserve {x}.',
            'example_idea_prompts': 'ORACLE_EXAMPLE: explore {x}.',
        },
    }


def agent(kind, data=None):
    with patch('idea.llm.LLMWrapper._setup_provider'):
        result = kind()
    result.register_custom_template('custom', data or template_data())
    return result


def assert_contract(prompt):
    assert prompt.count('REQ: preserve the literal {idea} and {"code":{"x":1}}.') == 1
    assert prompt.count('TITLE: a short title containing {title}.') == 1
    assert prompt.count('BODY: a narrative, preserving {critique} as text.') == 1
    assert prompt.count('Body length: exactly 100 words, excluding the title.') == 1


def test_context_direction_and_generation_keep_distinct_roles_without_extra_calls():
    ideator = agent(Ideator)
    ideator.generate_text = Mock(side_effect=['CONCEPTS: alpha, beta, gamma', 'DIRECTION_LITERAL {idea}', 'child'])
    ideator.generate_context('custom')
    idea, direction = ideator.generate_idea_from_context('POOL_LITERAL {critique}', 'custom')
    assert idea == 'child'
    assert direction == 'DIRECTION_LITERAL {idea}'
    assert ideator.generate_text.call_count == 3
    calls = [call.args[0] for call in ideator.generate_text.call_args_list]
    assert calls[0] == template_data()['prompts']['context']
    assert 'REQ:' not in calls[0]
    assert calls[1].count('REQ: preserve the literal {idea} and {"code":{"x":1}}.') == 1
    assert 'Background requirements for the future candidate (not this response)' in calls[1]
    for prompt in calls[:2]:
        assert 'TITLE: a short title' not in prompt
        assert 'BODY: a narrative' not in prompt
        assert 'Body length:' not in prompt
        assert 'Return only the candidate' not in prompt
    assert calls[1].endswith(
        'Return only one concise generation direction for a later model to follow. '
        'Do not write the candidate, a candidate title, commentary, or Markdown headings.')
    assert_contract(calls[2])
    assert 'notation {x}' in calls[0]
    assert 'POOL_LITERAL {critique}' in calls[1]
    assert 'DIRECTION_LITERAL {idea}' in calls[2]
    assert 'Return only the candidate' in calls[2]


def test_drabble_context_and_direction_do_not_request_a_finished_story():
    data = get_builtin_template_snapshot('drabble')
    prompts = get_prompts_from_dict(data)
    context = str(prompts.CONTEXT_PROMPT)
    assert context == data['prompts']['context']
    assert 'Return as: CONCEPTS:' in context
    assert 'Tell a complete fictional narrative' not in context
    direction = render_prompt(prompts.SPECIFIC_PROMPT, context_pool='salt, archive, {memory}')
    assert 'salt, archive, {memory}' in direction
    assert 'Prioritize originality' in direction
    assert direction.endswith(
        'Return only one concise generation direction for a later model to follow. '
        'Do not write the candidate, a candidate title, commentary, or Markdown headings.')
    for prompt in (context, direction):
        assert 'Title guidance:' not in prompt
        assert 'Body guidance:' not in prompt
        assert 'Body length:' not in prompt
    generated = prompts.build_generation_prompt('Write a drabble about a salt archive.')
    assert 'Body length: exactly 100 words, excluding the title.' in generated
    assert 'Title guidance: A short, evocative title.' in generated


def test_explicit_legacy_guidance_placeholders_in_intermediate_stages_remain_verbatim():
    data = template_data()
    data['requirements'] = []
    data['special_requirements'] = 'Author-owned text: keep {idea} and {"x":1}; Title: an explicit example.'
    data['prompts']['context'] = 'Author context:\n{requirements}\nReturn CONCEPTS only.'
    data['prompts']['specific_prompt'] = 'Plan from {context_pool}.\n{requirements_digest}'
    data['prompts']['genotype_encode'] = 'Extract from {idea_content}.\n{requirements}\n{requirements_digest}'
    prompts = get_prompts_from_dict(data)
    assert str(prompts.CONTEXT_PROMPT) == (
        'Author context:\n' + data['special_requirements'] + '\nReturn CONCEPTS only.')
    direction = render_prompt(prompts.SPECIFIC_PROMPT, context_pool='literal {title}')
    assert direction.startswith('Plan from literal {title}.\n' + data['special_requirements'])
    assert direction.count(data['special_requirements']) == 1
    assert 'Body length:' not in direction
    genotype = render_prompt(prompts.GENOTYPE_ENCODE_PROMPT, idea_content='literal {title} and {"state":1}')
    assert genotype == ('Extract from literal {title} and {"state":1}.\n'
                        + data['special_requirements'] + '\n')
    assert 'Title guidance:' not in genotype
    assert 'Body guidance:' not in genotype
    assert 'Body length:' not in genotype


def test_critique_and_refine_receive_contract_without_extra_calls():
    critic = agent(Critic)
    critic.generate_text = Mock(side_effect=['feedback {idea}', 'refined'])
    critic.refine('candidate {critique}', 'custom')
    assert critic.generate_text.call_count == 2
    for call in critic.generate_text.call_args_list:
        assert_contract(call.args[0])
        assert 'candidate {critique}' in call.args[0]
    assert 'feedback {idea}' in critic.generate_text.call_args_list[1].args[0]


def test_genotype_extracts_literal_candidate_without_automatic_candidate_contract():
    breeder = agent(Breeder)
    breeder.generate_text = Mock(return_value='genotype')
    breeder.encode_to_genotype({'idea': Idea(title='Title', content='body {idea} and {"state":1}')}, 'custom')
    breeder.generate_text.assert_called_once()
    prompt = breeder.generate_text.call_args.args[0]
    assert 'body {idea} and {"state":1}' in prompt
    assert 'REQ:' not in prompt
    assert 'Title guidance:' not in prompt
    assert 'Body guidance:' not in prompt
    assert 'Body length:' not in prompt
    assert 'Return only the candidate' not in prompt


@pytest.mark.parametrize('domain', ['game_design', 'arcade_game_design', 'drabble', 'airesearch', 'custom'])
def test_extraction_and_mutation_share_ingredient_rules_for_every_template(domain):
    data = template_data() if domain == 'custom' else get_builtin_template_snapshot(domain)
    breeder = agent(Breeder, data)
    ideator = agent(Ideator, data)
    breeder.generate_text = Mock(return_value='first-person; tactile crafting')
    ideator.generate_text = Mock(return_value='Genotype: turn-based; risk, reward tradeoff; TURN-BASED')
    breeder.encode_to_genotype('Full phenotype with {literal} braces', 'custom')
    genes = ideator.generate_mutation_genes('custom', ['first-person; tactile crafting'])
    assert genes == ['turn-based', 'risk, reward tradeoff']
    breeder.generate_text.assert_called_once()
    ideator.generate_text.assert_called_once()
    extraction = breeder.generate_text.call_args.args[0]
    mutation = ideator.generate_text.call_args.args[0]
    for prompt in (extraction, mutation):
        assert prompt.count(GENE_INGREDIENT_GUIDANCE) == 1
        assert 'Body length:' not in prompt
        assert 'Title guidance:' not in prompt
        assert 'Body guidance:' not in prompt
        assert 'Return only the candidate' not in prompt
    assert 'Full phenotype with {literal} braces' in extraction
    assert 'Full phenotype' not in mutation
    assert 'first-person; tactile crafting' in mutation
    assert 'CONCEPTS:' not in mutation
    assert data['prompts']['context'] not in mutation
    if domain == 'custom':
        assert 'GENOTYPE: Full phenotype' in extraction
        assert 'REQ: preserve the literal {idea} and {"code":{"x":1}}.' in mutation


def test_shared_gene_rule_changes_invalidate_saved_and_cached_extraction():
    breeder = agent(Breeder)
    breeder.generate_text = Mock(side_effect=['first-person; crafting', 'crafting; turn-based'])
    candidate = {'id': 'saved', 'idea': 'Full phenotype'}
    original = breeder.observe_genes(candidate, 'custom')
    assert breeder.encode_to_genotype(candidate, 'custom') == original['genotype']
    with patch('idea.gene_prompts.GENE_INGREDIENT_GUIDANCE', GENE_INGREDIENT_GUIDANCE + ' Revised rule.'):
        revised = breeder.observe_genes(candidate, 'custom')
    assert revised['genotype'] == 'crafting; turn-based'
    assert revised['fingerprint'] != original['fingerprint']
    assert breeder.generate_text.call_count == 2


def test_parsed_mutation_genes_keep_phrase_boundaries_and_replace_without_growing_context():
    ideator = agent(Ideator)
    ideator.generate_text = Mock(return_value='risk, reward tradeoff; shared clock')
    genes = ideator.generate_mutation_genes('custom', ['solo; puzzle; exploration'])
    ideator._random_sample = lambda items, count: list(items)[:count]
    ideator._random_shuffle = lambda items: None
    ideator.generate_context_from_parents(['solo; puzzle; exploration'], 0.5, genes)
    details = ideator.last_mutation_details
    assert details['context_size'] == 3
    assert details['replaced_count'] == 2
    assert details['introduced'] == ['risk, reward tradeoff', 'shared clock']
    assert len(details['removed']) == 2


@pytest.mark.parametrize('oracle', [False, True])
def test_formatter_uses_saved_prompt_for_all_candidates_and_preserves_body(oracle):
    formatter = agent(Formatter)
    formatter.generate_text = Mock(return_value='{"title":"Original","content":"original body"}')
    raw = {'idea': 'An unstructured original body', 'oracle_generated': oracle}
    formatter.format_idea(raw, 'custom')
    formatter.generate_text.assert_called_once()
    prompt = formatter.generate_text.call_args.args[0]
    assert 'FORMAT_MARKER' in prompt
    assert 'Preserve the body' in prompt
    assert 'Do not shorten, expand, or rewrite' in prompt
    assert 'Body length: exactly 100' not in prompt
    assert formatter.generate_text.call_args.kwargs['response_schema'] is Idea


@pytest.mark.parametrize('oracle', [False, True])
def test_canonical_formatter_preserves_100_word_body_and_metadata_without_a_call(oracle):
    formatter = agent(Formatter)
    formatter.generate_text = Mock(side_effect=AssertionError('Canonical envelope needs no provider'))
    body = ' '.join(['word'] * 94) + '\n"Keep {braces}, punctuation and quoted text."'
    assert len(body.split()) == 100
    raw = {'id': 'candidate-id', 'parent_ids': ['parent-id'],
           'idea': 'Title: Original title\n\nContent:\n' + body,
           'oracle_generated': oracle, 'oracle_analysis': 'Saved analysis'}
    result = formatter.format_idea(raw, 'custom')
    formatter.generate_text.assert_not_called()
    assert result['idea'].title == 'Original title'
    assert result['idea'].content == body
    assert result['id'] == 'candidate-id'
    assert result['parent_ids'] == ['parent-id']
    if oracle:
        assert result['oracle_generated'] is True
        assert result['oracle_analysis'] == 'Saved analysis'
    assert formatter.get_diagnostics()['formatter_envelope_extractions'] == 1


@pytest.mark.parametrize('source', [
    'Preface\nTitle: Example\nContent: body',
    'Title: \nContent: body',
    'Title: One\nContent: first\nTitle: Two\nContent: second',
    'Title: One\nContent:\n',
    'Title: One\nContent: first\nContent: second',
    'Title: One\nContent: first\nTitle: an alternative',
    'Title: One\nContent: first\n  Title: Two\n  Content: second',
    'Title: One\nContent: first\n   content: second',
    'Title: One\nContent: body\n```',
    'Title: One\nContent: body\n```python\nunfinished()',
    'Title: One\nContent:\n```text\nwrapped body\n```',
    'Title: One\nContent:\n~~~text\nwrapped body\n~~~',
    'Title: One\nContent: {"title":"Nested","content":"body"}',
    'Title: One\nContent: [{"title":"Nested","content":"body"}]',
    'Title: One\nContent: {}',
    'Title: One\nContent: []',
    'Title: One\nContent: { }',
    'Title: One\nContent: [exactly 100-word drabble]',
    'Title: One\nContent: <insert story here>',
    'Title: One\nContent: No response.',
    'Title: ** **\nContent: body',
    'Title: One\nContent: story\n\nAnalysis: explanation',
    'Title: One\nContent: story\n\nWord count: 100',
    'Title: One\nContent: story\n\n  Explanation: rationale',
    'Title: One\nContent: story\n\n**Notes:** commentary',
    '```text\nTitle: One\nContent: story\n```',
    '{"title":"One","content":"story"}',
])
def test_ambiguous_or_incomplete_envelopes_use_existing_formatter(source):
    formatter = agent(Formatter)
    formatter.generate_text = Mock(return_value='{"title":"Parsed","content":"interpreted body"}')
    result = formatter.format_idea(source, 'custom')
    formatter.generate_text.assert_called_once()
    assert source in formatter.generate_text.call_args.args[0]
    assert result.content == 'interpreted body'


def test_canonical_rich_body_with_embedded_balanced_code_skips_formatter():
    formatter = agent(Formatter)
    formatter.generate_text = Mock(side_effect=AssertionError('Unambiguous content needs no provider'))
    body = ('The design uses a dictionary, {"x":{"y":1}}, without modifying it.\r\n'
            '\r\n```python\r\n    render({"title": "literal", "content": "example"})\r\n```\r\n'
            '\r\nThe player then chooses “keep” or “release”.\r\n')
    result = formatter.format_idea('Title: The Dictionary\r\nContent:\r\n' + body, 'custom')
    formatter.generate_text.assert_not_called()
    assert result.title == 'The Dictionary'
    assert result.content == body


def test_envelope_parser_does_not_claim_to_validate_unlabeled_prose():
    body = 'A story.\n\nI hope you enjoy it.'
    assert extract_title_content_envelope('Title: Example\nContent: ' + body)['content'] == body


@pytest.mark.parametrize('custom_comparison', [None, 'JUDGE: {idea_a} vs {idea_b}', 'JUDGE: use this rubric.'])
def test_comparison_honors_custom_source_and_always_includes_complete_candidates(custom_comparison):
    data = template_data()
    if custom_comparison:
        data['comparison_prompt'] = 'ROOT_SHOULD_NOT_WIN'
        data['prompts']['comparison_prompt'] = custom_comparison
    critic = agent(Critic, data)
    critic.generate_text = Mock(return_value='Result: A')
    assert critic.compare_ideas({'title':'A title', 'content':'A full body {idea_b}'},
                               {'title':'B title', 'content':'B full body {idea_a}'}, 'custom') == 'A'
    prompt = critic.generate_text.call_args.args[0]
    assert_contract(prompt)
    for text in ('A title', 'B title', 'A full body {idea_b}', 'B full body {idea_a}'):
        assert text in prompt
    assert 'Result: A, Result: B, or Result: tie' in prompt
    assert 'ROOT_SHOULD_NOT_WIN' not in prompt
    if custom_comparison:
        assert 'JUDGE:' in prompt


def test_root_comparison_fallback_and_metadata_braces_remain_literal():
    data = template_data()
    data['metadata']['item_type'] = 'puzzles {layout}'
    data['comparison_criteria'] = ['Preserve literal {idea_a_content} and {"a":{"b":1}}']
    prompts = get_prompts_from_dict(data)
    prompt = render_prompt(prompts.COMPARISON_PROMPT, idea_a_title='A', idea_a_content='ALPHA',
                           idea_b_title='B', idea_b_content='BETA')
    assert 'puzzles {layout}' in prompt
    assert 'Preserve literal {idea_a_content} and {"a":{"b":1}}' in prompt
    assert 'ALPHA' in prompt and 'BETA' in prompt
    data['comparison_prompt'] = 'ROOT_JUDGE: {idea_a} vs {idea_b}'
    assert get_prompts_from_dict(data).COMPARISON_PROMPT.startswith('ROOT_JUDGE')


def test_breed_prompt_changes_actual_final_call_without_bypassing_concept_sampling():
    breeder = agent(Breeder)
    breeder.mutation_rate = 0
    breeder.encode_to_genotype = Mock(side_effect=['retained; EXCLUDED_PARENT', 'other'])
    with patch('idea.llm.LLMWrapper._setup_provider'):
        helper = breeder._get_thread_ideator()
    helper.generate_context_from_parents = Mock(return_value='retained, sampled_mutation')
    helper.generate_text = Mock(side_effect=['child direction', 'child body', 'new direction', 'new body'])
    parents = [{'id':'p1','idea':'one'}, {'id':'p2','idea':'two'}]
    result = breeder.breed(parents, 'custom')
    assert helper.generate_text.call_count == 2
    prompt = helper.generate_text.call_args.args[0]
    assert 'BREED_EDITED' in prompt
    assert 'Sampled parent/mutation concepts' in prompt
    assert 'retained, sampled_mutation' in prompt
    assert 'EXCLUDED_PARENT' not in prompt
    assert_contract(prompt)
    assert result['parent_genotypes'][0]['genotype'] == 'retained; EXCLUDED_PARENT'
    changed = template_data()
    changed['prompts']['breed'] = 'SECOND_BREED_EDIT: {ideas}'
    breeder.register_custom_template('custom', changed)
    breeder.encode_to_genotype = Mock(return_value='retained; EXCLUDED_PARENT')
    breeder.breed(parents, 'custom')
    assert helper.generate_text.call_count == 4
    assert 'SECOND_BREED_EDIT' in helper.generate_text.call_args.args[0]


def test_oracle_analysis_and_replacement_share_contract_without_changing_protocol():
    oracle = agent(Oracle)
    prompt = oracle._build_analysis_prompt([], 'custom')
    assert_contract(prompt)
    assert 'ORACLE_GOAL: preserve {x}' in prompt
    assert 'ORACLE_EXAMPLE: explore {x}' in prompt
    assert '=== ORACLE ANALYSIS ===' in prompt
    assert '=== IDEA PROMPT ===' in prompt
    prompts = oracle._resolve_prompts('custom')
    assert_contract(prompts.build_generation_prompt('replacement direction {idea}'))


def test_legacy_empty_placeholders_and_json_guidance_render_once_without_truncation():
    data = template_data()
    data.pop('requirements')
    data.pop('output')
    data['special_requirements'] = ''
    data['prompts']['refine'] = '{idea}\n{critique}\n{requirements}\n{requirements_digest}\n{{literal}}\n{"nested":{"x":1}}'
    prompts = get_prompts_from_dict(data)
    rendered = render_prompt(prompts.REFINE_PROMPT, idea='body {critique}', critique='feedback')
    assert '{requirements}' not in rendered and '{requirements_digest}' not in rendered
    assert '{literal}' in rendered
    assert '{"nested":{"x":1}}' in rendered
    assert 'body {critique}' in rendered
    legacy = '\n'.join(f'Legacy rule {i}: keep {{idea}} and {{"x":{i}}}' for i in range(9))
    data['special_requirements'] = legacy
    prompts = get_prompts_from_dict(data)
    assert render_prompt(prompts.REFINE_PROMPT, idea='body', critique='feedback').count(legacy) == 1
    assert render_prompt(prompts.GENOTYPE_ENCODE_PROMPT, idea_content='body') == 'GENOTYPE: body'
    assert prompts.template.special_requirements == legacy


def test_escaped_json_prompt_braces_keep_their_original_meaning():
    assert render_prompt('Example: {{"x":{{"y":1}}}}\n{idea}', idea='{raw}') == 'Example: {"x":{"y":1}}\n{raw}'
