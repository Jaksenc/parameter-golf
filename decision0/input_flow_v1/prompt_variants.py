"""Four preregisterable input arrangements. No model calls; no reference labels."""
from __future__ import annotations
import json
from typing import Any

SYSTEM = ('Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. '
          'Respond with only its uppercase letter, with no explanation or reasoning.')
VARIANTS = ('baseline', 'schema_first', 'schema_bracket', 'repeat_full')


def messages(row: dict[str, Any], variant: str) -> list[dict[str, str]]:
    if variant not in VARIANTS:
        raise ValueError('Unknown variant')
    state, criterion, options = row['state'], row['question'], row['options']
    if not isinstance(criterion, str) or not criterion.strip():
        raise ValueError('Missing criterion')
    if not isinstance(options, list) or not 2 <= len(options) <= 16:
        raise ValueError('Expected 2-16 outcomes')
    if len({o['id'] for o in options}) != len(options):
        raise ValueError('Duplicate outcome identifiers')
    choices = [{'letter': chr(65+i), 'description': o['description']} for i, o in enumerate(options)]
    if variant == 'baseline' or variant == 'repeat_full':
        payload = {'evidence': state, 'criterion': criterion, 'options': choices}
    elif variant == 'schema_first':
        payload = {'criterion': criterion, 'options': choices, 'evidence': state}
    else:
        payload = {'criterion': criterion, 'options': choices, 'evidence': state,
                   'decision_reminder': {'criterion': criterion, 'options': choices}}
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    if variant == 'repeat_full':
        text = text + '\n\n' + text
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': text}]


def encode_checked(tokenizer: Any, row: dict[str, Any], variant: str, max_tokens: int = 8192) -> dict:
    if max_tokens <= 0:
        raise ValueError('Invalid token limit')
    text = tokenizer.apply_chat_template(messages(row, variant), tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids or len(ids) > max_tokens:
        raise ValueError(f'{len(ids)} tokens; limit {max_tokens}; no truncation permitted')
    slots = []
    for i in range(len(row['options'])):
        letter = chr(65+i)
        token = tokenizer.encode(letter, add_special_tokens=False)
        if len(token) != 1 or tokenizer.encode(text+letter, add_special_tokens=False) != ids+token:
            raise ValueError('Answer-token boundary not preserved')
        slots.append(token[0])
    if len(set(slots)) != len(slots):
        raise ValueError('Answer-token collision')
    return {'input_ids': ids, 'answer_token_ids': slots, 'input_tokens': len(ids), 'variant': variant}
