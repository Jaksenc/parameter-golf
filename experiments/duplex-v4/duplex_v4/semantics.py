"""Bounded epistemic Boolean semantics, independently checked by a bitset evaluator.

Unknown observations permit both truth values. Repeated occurrences of an atom
refer to ONE shared fact. This is possible-world (supervaluational) semantics,
not strong Kleene evaluation. Conflicting reports are unresolved by explicit policy.
"""
from __future__ import annotations
import hashlib
import itertools
import json
from typing import Any

ATOMS = ('p', 'q', 'r')
STATES = (False, True, None)

def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()

def validate(tree: Any, depth: int = 0) -> None:
    if depth > 6:
        raise ValueError('Expression is too deep')
    if isinstance(tree, str):
        if tree not in ATOMS:
            raise ValueError('Unknown atom')
        return
    if not isinstance(tree, (list, tuple)) or not tree:
        raise ValueError('Invalid expression')
    op = tree[0]
    if op not in ('not', 'and', 'or', 'xor') or len(tree) != (2 if op == 'not' else 3):
        raise ValueError('Invalid operation or arity')
    for child in tree[1:]:
        validate(child, depth + 1)

def validate_facts(facts: dict) -> None:
    if set(facts) != set(ATOMS) or any(v is not None and type(v) is not bool for v in facts.values()):
        raise ValueError('Exactly p/q/r with Boolean or None values required')

def world_value(tree: Any, world: dict[str, bool]) -> bool:
    if isinstance(tree, str):
        return world[tree]
    op = tree[0]
    a = world_value(tree[1], world)
    if op == 'not':
        return not a
    b = world_value(tree[2], world)
    if op == 'and': return a and b
    if op == 'or': return a or b
    if op == 'xor': return a != b
    raise ValueError(op)

def possible_worlds(facts: dict) -> list[dict[str, bool]]:
    validate_facts(facts)
    choices = [(False, True) if facts[a] is None else (facts[a],) for a in ATOMS]
    return [dict(zip(ATOMS, values)) for values in itertools.product(*choices)]

def oracle(tree: Any, facts: dict) -> int:
    """Semantic actions: 0 perform, 1 leave unchanged, 2 request clarification."""
    validate(tree)
    outcomes = {world_value(tree, w) for w in possible_worlds(facts)}
    return 0 if outcomes == {True} else 1 if outcomes == {False} else 2

def bitset_reference(tree: Any, facts: dict) -> int:
    """Independent postfix bitwise evaluation, no world_value/oracle call."""
    validate(tree); validate_facts(facts)
    assignments = list(itertools.product((False, True), repeat=3))
    masks = {a: sum(1 << i for i, w in enumerate(assignments) if w[k])
             for k, a in enumerate(ATOMS)}
    postfix: list[str] = []
    def compile_(node):
        if isinstance(node, str): postfix.append(node); return
        for child in node[1:]: compile_(child)
        postfix.append(node[0])
    compile_(tree)
    stack: list[int] = []
    for token in postfix:
        if token in masks: stack.append(masks[token])
        elif token == 'not': stack.append(255 ^ stack.pop())
        else:
            b, a = stack.pop(), stack.pop()
            stack.append(a & b if token == 'and' else a | b if token == 'or' else a ^ b)
    allowed = 255
    for atom, value in facts.items():
        if value is not None:
            allowed &= masks[atom] if value else (255 ^ masks[atom])
    positive = stack[0] & allowed
    return 1 if positive == 0 else 0 if positive == allowed else 2

def function_signature(tree: Any) -> str:
    validate(tree)
    return ''.join(str(int(world_value(tree, dict(zip(ATOMS, vals)))))
                   for vals in itertools.product((False, True), repeat=3))

def minimal_witness(tree: Any, facts: dict) -> list[str]:
    """Smallest observed fact subset sufficient for the same definite decision.
    For an ambiguous condition there is no definite witness; return empty list.
    """
    answer = oracle(tree, facts)
    if answer == 2: return []
    known = [a for a in ATOMS if facts[a] is not None]
    for size in range(len(known) + 1):
        for subset in itertools.combinations(known, size):
            reduced = {a: facts[a] if a in subset else None for a in ATOMS}
            if oracle(tree, reduced) == answer: return list(subset)
    raise AssertionError('Full observed facts must suffice')

def checked_proof(tree: Any, facts: dict) -> dict:
    a, b = oracle(tree, facts), bitset_reference(tree, facts)
    if a != b: raise AssertionError('Independent oracles disagree')
    worlds = possible_worlds(facts)
    return {'semantics': 'possible-world-v1', 'semantic_target': a,
            'feasible_worlds': len(worlds),
            'true_worlds': sum(world_value(tree, w) for w in worlds),
            'minimal_witness_atoms': minimal_witness(tree, facts),
            'binding': digest({'ast': tree, 'facts': facts})}
