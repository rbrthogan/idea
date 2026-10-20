"""Request validation generated from the canonical run settings.

Legacy persistence may normalize old values. New HTTP requests must reject
invalid values instead of silently running a different experiment.
"""

from copy import deepcopy
from typing import Any, Mapping

from pydantic import ConfigDict, create_model

from idea.config import DEFAULT_MODEL
from idea.evolution_types import EvolutionRunConfig


def _camel_case(name: str) -> str:
    head, *tail = name.split('_')
    return head + ''.join(word.title() for word in tail)


RequestRunConfig = create_model(
    'RequestRunConfig',
    __config__=ConfigDict(extra='forbid', allow_inf_nan=False),
    **{
        name: (field.annotation, deepcopy(field))
        for name, field in EvolutionRunConfig.model_fields.items()
    },
)
ALIASES = {_camel_case(name): name for name in EvolutionRunConfig.model_fields}
ALIASES['evaluation_config'] = 'evaluation'
REQUEST_METADATA = {'evolutionName', 'evolution_name'}


def parse_run_request(data: Mapping[str, Any]) -> EvolutionRunConfig:
    if not isinstance(data, dict):
        raise ValueError('Request body must be a JSON object')
    values = {'pop_size': 3, 'generations': 2, 'model_type': DEFAULT_MODEL}
    seen = set()
    for key, value in data.items():
        if key in REQUEST_METADATA:
            if value is not None and (not isinstance(value, str) or len(value) > 200):
                raise ValueError('Evolution name must be text of at most 200 characters')
            continue
        name = ALIASES.get(key, key)
        if name in seen:
            raise ValueError(f'Duplicate setting: {name}')
        seen.add(name)
        if name == 'random_seed' and isinstance(value, (int, float)) and value > 2**53 - 1:
            raise ValueError('Random seed numbers must not exceed 9007199254740991; send a larger legacy seed as an exact decimal string')
        values[name] = value
    validated = RequestRunConfig.model_validate(values)
    canonical = EvolutionRunConfig.model_validate(validated.model_dump())
    if 'tournament_rounds' not in seen:
        count = canonical.tournament_count if 'tournament_count' in seen else 1.0
        count = max(0.25, round(count * 4) / 4)
        canonical.tournament_count = count
        canonical.tournament_rounds = max(1, int(count * canonical.full_tournament_rounds + 0.5))
    if canonical.tournament_rounds > 1000:
        raise ValueError('Tournament rounds must not exceed 1000')
    if canonical.model_type != DEFAULT_MODEL:
        from idea.config import LLM_MODELS
        if canonical.model_type not in {model['id'] for model in LLM_MODELS}:
            raise ValueError('Choose a supported model')
    return canonical


def parse_additional_generations(data: Any) -> int:
    if not isinstance(data, dict):
        raise ValueError('Request body must be a JSON object')
    value = data.get('additional_generations', data.get('additionalGenerations', 3))
    # The generated request model checks integer coercion and the same bounds.
    return RequestRunConfig.model_validate({'generations': value}).generations
