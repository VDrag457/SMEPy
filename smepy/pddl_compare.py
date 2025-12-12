#!/usr/bin/env python3
"""PDDL Domain Similarity Calculator using Structure Mapping Engine
Usage: from pddl_compare import calculate_similarity
       score = calculate_similarity(pddl_string1, pddl_string2)
"""
import re
import copy

# S-Expression Parser

term_regex = r'''(?mx)
    \s*(?:
        (?P<brackl>\()|
        (?P<brackr>\))|
        (?P<num>\-?\d+\.\d+|\-?\d+)|
        (?P<sq>"[^"]*")|
        (?P<s>[^(^)\s]+)
       )'''

def parse_sexp(sexp):
    stack = []
    out = []
    for termtypes in re.finditer(term_regex, sexp):
        term, value = [(t,v) for t,v in termtypes.groupdict().items() if v][0]
        if   term == 'brackl':
            stack.append(out)
            out = []
        elif term == 'brackr':
            assert stack, "Trouble with nesting of brackets"
            tmpout, out = out, stack.pop(-1)
            out.append(tmpout)
        elif term == 'num':
            v = float(value)
            if v.is_integer(): v = int(v)
            out.append(v)
        elif term == 'sq':
            out.append(value[1:-1])
        elif term == 's':
            out.append(value)
        else:
            raise NotImplementedError("Error: %r" % (term, value))
    assert not stack, "Trouble with nesting of brackets"
    return out[0]

def read_meld_string(meld_str):
    s_exp_list = parse_sexp('(' + meld_str + ')')
    if not (s_exp_list[0][0] == 'in-microtheory'):
        raise IOError('Not a microtheory format!')
    mt_name = s_exp_list[0][1]
    mt_facts = s_exp_list[1:]
    return (mt_name, mt_facts)

# Structure Case Classes
def get_hash_name(item):
    return '(' + ' '.join(map(get_hash_name, item)) + ')' if isinstance(item, list) else item

class Vocabulary:
    def __init__(self):
        self.p_dict = {}
    def add(self, pred_name, arity):
        new_pred = Predicate(pred_name, arity)
        self.p_dict[pred_name] = new_pred
        return new_pred
    def __getitem__(self, pred_name):
        if pred_name not in self.p_dict:
            raise KeyError(f'Unknown predicate {pred_name}')
        return self.p_dict[pred_name]
    def __contains__(self, pred_name):
        return pred_name in self.p_dict
    def check_arity(self, pred_name, arity):
        if pred_name not in self.p_dict:
            raise KeyError(f'Unknown predicate {pred_name}')
        return self.p_dict[pred_name].arity == arity

current_vocab = Vocabulary()

class Predicate:
    def __init__(self, name, arity, predicate_type='relation'):
        self.name = name
        self.arity = arity
        self.predicate_type = 'function' if name[-2:] == 'Fn' else predicate_type
    @property
    def list_form(self):
        return self.name
    def __repr__(self):
        return '<' + self.name + '>'

class Entity:
    def __init__(self, name):
        self.name = name
    @property
    def list_form(self):
        return self.name
    def __repr__(self):
        return '<' + self.name + '>'

class Expression:
    def __init__(self, case, s_exp, weight=1.0, create_new_pred=True, evidences=None):
        pred_name, arg_list, num_of_args = s_exp[0], s_exp[1:], len(s_exp[1:])
        self.args = []
        if pred_name in current_vocab:
            self.predicate = current_vocab[pred_name]
        elif create_new_pred:
            self.predicate = current_vocab.add(pred_name, num_of_args)
        else:
            raise KeyError(f'Unknown predicate {pred_name}')
        if not current_vocab.check_arity(pred_name, num_of_args):
            raise ValueError(f'Wrong arity for predicate {pred_name}')
        for arg in arg_list:
            self.args.append(case.add(arg))
        self.weight, self.evidences, self.case = weight, evidences, case
    @property
    def name(self):
        s_exp_list = [arg.name for arg in self.args]
        s_exp_list.insert(0, self.predicate.name)
        return '(' + ' '.join(s_exp_list) + ')'
    @property
    def list_form(self):
        return [self.predicate.list_form] + [arg.list_form for arg in self.args]
    def __repr__(self):
        return '<' + self.name + ', ' + repr(self.weight) + '>'
    def __deepcopy__(self, memo):
        new_copy = copy.copy(self)
        new_copy.evidences = copy.copy(self.evidences)
        return new_copy

class StructCase:
    def __init__(self, exp_info_list, name=None):
        self.items = {}
        for exp_info in exp_info_list:
            self.add(exp_info)
        self.vocab, self.name = current_vocab, name
    @property
    def expression_list(self):
        return [self.items[key] for key in list(self.items) if isinstance(self.items[key], Expression)]
    @property
    def entity_list(self):
        return [self.items[key] for key in list(self.items) if isinstance(self.items[key], Entity)]
    @property
    def item_list(self):
        return [self.items[key] for key in list(self.items)]
    def __getitem__(self, index):
        new_index = get_hash_name(index) if isinstance(index, list) else index
        return self.items.get(new_index, None)
    def add(self, item):
        if not item in self:
            if isinstance(item, list):
                return self.add_s_exp_w((item, 1.0))
            elif isinstance(item, tuple):
                return self.add_s_exp_w(item)
            elif isinstance(item, str):
                return self.add_entity(Entity(item))
            elif isinstance(item, (Expression, Entity)):
                return self.add_expression(item) if isinstance(item, Expression) else self.add_entity(item)
        elif isinstance(item, (Expression, Entity)):
            return item
        else:
            return self[item]
    def add_s_exp_w(self, s_exp_w):
        s_exp, w = s_exp_w
        new_expression = Expression(self, s_exp, w)
        self.items[new_expression.name] = new_expression
        return new_expression
    def add_entity(self, entity):
        self.items[entity.name] = entity
        return entity
    def add_expression(self, expression):
        self.items[expression.name] = expression
        for arg in expression.args:
            self.add(arg)
        return expression
    def __contains__(self, item):
        if isinstance(item, (Expression, Entity)):
            return item.name in self.items
        elif isinstance(item, list):
            return get_hash_name(item) in self.items
        elif isinstance(item, tuple):
            return get_hash_name(item[0]) in self.items
        return item in self.items
    def copy(self):
        return StructCase([(expression.list_form, expression.weight) for expression in self.expression_list])

# Structure Mapping Engine

# Structure Mapping Engine
class Match:
    def __init__(self, base, target, score=0.0):
        self.base, self.target, self.score = base, target, score
        self.children, self.parents = [], []
        self.mapping, self.is_incomplete, self.is_inconsistent = None, False, False
    def add_parent(self, parent):
        self.parents.append(parent)
    def add_child(self, child):
        self.children.append(child)
    def local_evaluation(self):
        self.score = predicate_match_score(self.base.predicate, self.target.predicate) if isinstance(self.base, Expression) else 0.0
        return self.score
    def __repr__(self):
        return '('+repr(self.base)+' -- '+repr(self.target)+')'
    def __eq__(self, other):
        return isinstance(other, Match) and self.base == other.base and self.target == other.target
    def __hash__(self):
        return hash(repr(self))

class Mapping:
    def __init__(self, matches=None):
        self.base_to_target, self.target_to_base, self.matches, self.score = {}, {}, set(), 0.0
        if matches:
            self.add_all(matches)
    def get_mapped_base(self, base):
        return self.base_to_target.get(base, None)
    def get_mapped_target(self, target):
        return self.target_to_base.get(target, None)
    def is_consistent_with(self, match):
        base, target = match.base, match.target
        corresponding_target = self.get_mapped_base(base)
        corresponding_base = self.get_mapped_target(target)
        is_base_consistent = (not corresponding_target) or (corresponding_target == target)
        is_target_consistent = (not corresponding_base) or (corresponding_base == base)
        return is_base_consistent and is_target_consistent
    def mutual_consistent(self, mapping):
        return all(self.is_consistent_with(match) for match in mapping.matches)
    def add(self, match, check_consistency=True):
        if match in self.matches:
            return
        if (not check_consistency) or self.is_consistent_with(match):
            self.base_to_target[match.base] = match.target
            self.target_to_base[match.target] = match.base
            self.matches.add(match)
        else:
            raise ValueError(f'Mapping is not consistent with {match}')
    def add_all(self, matches, check_consistency=True):
        for match in matches:
            self.add(match, check_consistency)
    def merge(self, mapping):
        self.add_all(mapping.matches, check_consistency=False)
    def evaluate(self):
        self.score = sum(match.score for match in self.matches)
        return self.score
    def copy(self):
        new_mapping = Mapping(self.matches)
        new_mapping.score = self.score
        return new_mapping
    def __str__(self):
        entity_matches, expression_matches = [], []
        for match in self.matches:
            (expression_matches if isinstance(match.base, Expression) else entity_matches).append(match)
        return 'expression mappings:\n' + ',\n'.join(map(repr, expression_matches)) + '\n' + 'entity mappings:\n' + ', '.join(map(repr, entity_matches))

class SME:
    def __init__(self, base, target):
        self.base, self.target = base, target
    def match(self):
        matches = create_all_possible_matches(self.base, self.target)
        connect_matches(matches)
        valid_matches = consistency_propagation(matches)
        structural_evaluation(valid_matches)
        kernel_mappings = find_kernel_mappings(valid_matches)
        return greedy_merge(kernel_mappings)

def predicate_match_score(pred_1, pred_2):
    if pred_1.predicate_type == 'relation' and pred_1.name == pred_2.name:
        return 0.005
    return 0.002 if pred_1.predicate_type == 'function' else 0.0

def are_predicates_matchable(pred_1, pred_2):
    if pred_1.predicate_type != pred_2.predicate_type:
        return False
    return pred_1.name == pred_2.name if pred_1.predicate_type == 'relation' else True

def are_matchable(item_1, item_2):
    is_exp_1, is_exp_2 = isinstance(item_1, Expression), isinstance(item_2, Expression)
    if is_exp_1 and is_exp_2:
        return are_predicates_matchable(item_1.predicate, item_2.predicate)
    return (not is_exp_1) and (not is_exp_2)

def create_all_possible_matches(case_1, case_2):
    matches = set()
    for exp_1 in case_1.expression_list:
        for exp_2 in case_2.expression_list:
            matches = set.union(matches, match_expression(exp_1, exp_2))
    return list(matches)

def match_expression(exp_1, exp_2):
    pair_list = [(exp_1, exp_2)] + list(zip(exp_1.args, exp_2.args))
    if all([are_matchable(pair[0], pair[1]) for pair in pair_list]):
        return set([Match(pair[0], pair[1]) for pair in pair_list])
    return set()

def connect_matches(matches):
    match_dict = {(match.base, match.target): match for match in matches}
    for match in matches:
        if isinstance(match.base, Expression):
            for arg_pair in list(zip(match.base.args, match.target.args)):
                if arg_pair in match_dict:
                    child_match = match_dict[arg_pair]
                    child_match.add_parent(match)
                    match.add_child(child_match)
                else:
                    match.is_incomplete = True

def consistency_propagation(matches):
    match_graph = {match: match.children for match in matches}
    ordered_from_leaves_matches = topological_sort(match_graph)
    for match in ordered_from_leaves_matches:
        match.mapping = Mapping([match])
        for child in match.children:
            if match.mapping.mutual_consistent(child.mapping):
                match.mapping.merge(child.mapping)
            else:
                match.is_inconsistent = True
                break
    return [match for match in matches if not (match.is_incomplete or match.is_inconsistent)]

def structural_evaluation(matches, trickle_down_factor=16):
    for match in matches:
        match.local_evaluation()
    for match in matches[::-1]:
        for child in match.children:
            child.score = min(child.score + match.score * trickle_down_factor, 1.0)
    for match in matches[::-1]:
        match.mapping.evaluate()
    return matches

def find_kernel_mappings(valid_matches):
    root_matches = [match for match in valid_matches if all([parent not in valid_matches for parent in match.parents])]
    return [match.mapping.copy() for match in root_matches]

def greedy_merge(kernel_mappings):
    sorted_k_mapping_list = sorted(kernel_mappings, key=lambda m: m.score, reverse=True)
    global_mappings, max_score = [], 0.0
    while len(global_mappings) < 3 and sorted_k_mapping_list:
        global_mapping = sorted_k_mapping_list.pop(0)
        for kernel_mapping in sorted_k_mapping_list[:]:
            if global_mapping.mutual_consistent(kernel_mapping):
                global_mapping.merge(kernel_mapping)
                sorted_k_mapping_list.remove(kernel_mapping)
        score = global_mapping.evaluate()
        if score <= 0.8 * max_score:
            break
        max_score = max(score, max_score)
        global_mappings.append(global_mapping)
    return sorted(global_mappings, key=lambda m: m.score, reverse=True)

def topological_sort(graph_dict):
    sorted_list, sorted_set, new_graph_dict = [], set(), graph_dict.copy()
    while new_graph_dict:
        for node in new_graph_dict:
            if all([next_node in sorted_set for next_node in new_graph_dict[node]]):
                sorted_list.append(node)
                sorted_set.add(node)
                del new_graph_dict[node]
                break
        else:
            raise ValueError('Cyclic graph!')
    return sorted_list

# PDDL Parser

# PDDL Parser
class PDDLParser:
    def __init__(self, pddl_string):
        self.pddl_string, self.domain_name, self.predicates, self.actions = pddl_string, None, [], []
        self.parse()
    def parse(self):
        content = re.sub(r';.*', '', self.pddl_string)
        domain_match = re.search(r'\(domain\s+([^\)]+)\)', content, re.IGNORECASE)
        if domain_match:
            self.domain_name = domain_match.group(1).strip()
        predicates_match = re.search(r'\(:predicates(.*?)\)\s*\)', content, re.DOTALL | re.IGNORECASE)
        if predicates_match:
            self.predicates = self._extract_predicates(predicates_match.group(1))
        for match in re.finditer(r'\(:action\s+([^\s]+)(.*?)\)\s*(?=\(:action|\Z)', content, re.DOTALL | re.IGNORECASE):
            self.actions.append(self._parse_action(match.group(1).strip(), match.group(2)))
    def _extract_predicates(self, text):
        return [match.group(1).strip() for match in re.finditer(r'\(([^\)]+)\)', text) if match.group(1).strip()]
    def _parse_action(self, name, body):
        action = {'name': name, 'parameters': [], 'preconditions': [], 'effects': []}
        param_match = re.search(r':parameters\s*\((.*?)\)', body, re.DOTALL | re.IGNORECASE)
        if param_match:
            action['parameters'] = self._parse_parameters(param_match.group(1))
        precond_match = re.search(r':precondition\s*\((.*?)\)\s*:effect', body, re.DOTALL | re.IGNORECASE)
        if precond_match:
            action['preconditions'] = self._parse_conditions(precond_match.group(1))
        effect_match = re.search(r':effect\s*\((.*?)\)\s*\)', body, re.DOTALL | re.IGNORECASE)
        if effect_match:
            action['effects'] = self._parse_effects(effect_match.group(1))
        return action
    def _parse_parameters(self, text):
        params, tokens = [], text.split()
        for i, token in enumerate(tokens):
            if token.startswith('?'):
                param_type = tokens[i+2] if i+2 < len(tokens) and tokens[i+1] == '-' else 'object'
                params.append({'name': token, 'type': param_type})
        return params
    def _parse_conditions(self, text):
        text = re.sub(r'^\s*and\s*', '', text, flags=re.IGNORECASE)
        return [match.group(1).strip() for match in re.finditer(r'\(([^\(\)]+)\)', text) if match.group(1).strip() and not match.group(1).lower().startswith('and')]
    def _parse_effects(self, text):
        effects, text = [], re.sub(r'^\s*and\s*', '', text, flags=re.IGNORECASE)
        for match in re.finditer(r'\((not\s+)?\(([^\)]+)\)\)|\(([^\(\)]+)\)', text):
            if match.group(2):
                effects.append({'type': 'delete', 'predicate': match.group(2).strip()})
            elif match.group(3):
                pred = match.group(3).strip()
                if pred and not pred.lower().startswith('and'):
                    effects.append({'type': 'add', 'predicate': pred})
        return effects

# PDDL to MELD Converter

# PDDL to MELD Converter
class PDDLToMeld:
    def __init__(self, pddl_parser):
        self.parser, self.meld_facts = pddl_parser, []
    def convert(self):
        self.meld_facts = []
        for action in self.parser.actions:
            self._convert_action(action)
        return f"{self.parser.domain_name}Mt", self.meld_facts
    def _convert_action(self, action):
        action_name, action_entity = action['name'], f"action-{action['name']}"
        for precond in action['preconditions']:
            self.meld_facts.append(f"({self._sanitize_predicate(precond)} {action_entity})")
        for effect in action['effects']:
            self.meld_facts.append(f"({self._sanitize_predicate(effect['predicate'])} {action_entity})")
        if action['preconditions'] and action['effects']:
            precond_compound = f"preconditions-{action_name}"
            for precond in action['preconditions']:
                self.meld_facts.append(f"(has-precond {precond_compound} {self._sanitize_predicate(precond)})")
            effect_compound = f"effects-{action_name}"
            for effect in action['effects']:
                self.meld_facts.append(f"(has-effect {effect_compound} {self._sanitize_predicate(effect['predicate'])})")
            self.meld_facts.append(f"(action-type {action_entity} planning-action)")
            self.meld_facts.append(f"(cause {precond_compound} {effect_compound})")
        for param in action['parameters']:
            self.meld_facts.append(f"(operates-on {action_entity} {param['type']})")
    def _sanitize_predicate(self, predicate):
        pred = predicate.split()[0] if ' ' in predicate else predicate
        return pred.replace('?', '').replace('-', '_')
    def to_meld_string(self):
        mt_name, facts = self.convert()
        return f"(in-microtheory {mt_name})\n" + '\n'.join(facts) + '\n'

# Main API

# Main API
def calculate_similarity(pddl_string1, pddl_string2):
    """Calculate similarity score between two PDDL domain strings.
    Args: pddl_string1, pddl_string2 - PDDL domain strings
    Returns: float - Best similarity score (0.0 if no mappings found)
    """
    global current_vocab
    current_vocab = Vocabulary()
    try:
        parser1, parser2 = PDDLParser(pddl_string1), PDDLParser(pddl_string2)
        meld_str1, meld_str2 = PDDLToMeld(parser1).to_meld_string(), PDDLToMeld(parser2).to_meld_string()
        name1, facts1 = read_meld_string(meld_str1)
        name2, facts2 = read_meld_string(meld_str2)
        case1, case2 = StructCase(facts1, name1), StructCase(facts2, name2)
        global_mappings = SME(case1, case2).match()
        return max(m.score for m in global_mappings) if global_mappings else 0.0
    except Exception as e:
        raise ValueError(f"Error calculating similarity: {e}")

if __name__ == "__main__":
    blocksworld = "(define (domain blocksworld) (:requirements :strips) (:predicates (on ?x ?y) (ontable ?x) (clear ?x) (handempty) (holding ?x)) (:action pick-up :parameters (?x) :precondition (and (clear ?x) (ontable ?x) (handempty)) :effect (and (not (ontable ?x)) (not (clear ?x)) (not (handempty)) (holding ?x))) (:action put-down :parameters (?x) :precondition (holding ?x) :effect (and (not (holding ?x)) (clear ?x) (handempty) (ontable ?x))))"
    logistics = "(define (domain logistics) (:requirements :strips) (:predicates (at ?obj ?loc) (in ?pkg ?veh) (vehicle ?v) (package ?p)) (:action load-package :parameters (?pkg ?veh ?loc) :precondition (and (at ?pkg ?loc) (at ?veh ?loc)) :effect (and (not (at ?pkg ?loc)) (in ?pkg ?veh))) (:action unload-package :parameters (?pkg ?veh ?loc) :precondition (and (in ?pkg ?veh) (at ?veh ?loc)) :effect (and (not (in ?pkg ?veh)) (at ?pkg ?loc))))"
    print(f"Similarity Score: {calculate_similarity(blocksworld, logistics):.4f}")
