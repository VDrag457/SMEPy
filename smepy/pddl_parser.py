import re

class PDDLParser:
    """Simple PDDL parser for domain files or strings."""
    
    def __init__(self, filepath=None, pddl_string=None):
        self.filepath = filepath
        self.pddl_string = pddl_string
        self.domain_name = None
        self.predicates = []
        self.actions = []
        self.parse()
    
    def parse(self):
        if self.pddl_string is not None:
            content = self.pddl_string
        elif self.filepath is not None:
            with open(self.filepath, 'r') as f:
                content = f.read()
        else:
            raise ValueError("Either filepath or pddl_string must be provided")
        
        # Remove comments
        content = re.sub(r';.*', '', content)
        
        # Extract domain name
        domain_match = re.search(r'\(domain\s+([^\)]+)\)', content, re.IGNORECASE)
        if domain_match:
            self.domain_name = domain_match.group(1).strip()
        
        # Extract predicates
        predicates_match = re.search(r'\(:predicates(.*?)\)\s*\)', content, re.DOTALL | re.IGNORECASE)
        if predicates_match:
            pred_text = predicates_match.group(1)
            self.predicates = self._extract_predicates(pred_text)
        
        # Extract actions
        action_pattern = r'\(:action\s+([^\s]+)(.*?)\)\s*(?=\(:action|\Z)'
        for match in re.finditer(action_pattern, content, re.DOTALL | re.IGNORECASE):
            action_name = match.group(1).strip()
            action_body = match.group(2)
            action = self._parse_action(action_name, action_body)
            self.actions.append(action)
    
    def _extract_predicates(self, text):
        predicates = []
        pred_pattern = r'\(([^\)]+)\)'
        for match in re.finditer(pred_pattern, text):
            pred = match.group(1).strip()
            if pred:
                predicates.append(pred)
        return predicates
    
    def _parse_action(self, name, body):
        action = {
            'name': name,
            'parameters': [],
            'preconditions': [],
            'effects': []
        }
        
        # Extract parameters
        param_match = re.search(r':parameters\s*\((.*?)\)', body, re.DOTALL | re.IGNORECASE)
        if param_match:
            action['parameters'] = self._parse_parameters(param_match.group(1))
        
        # Extract preconditions
        precond_match = re.search(r':precondition\s*\((.*?)\)\s*:effect', body, re.DOTALL | re.IGNORECASE)
        if precond_match:
            action['preconditions'] = self._parse_conditions(precond_match.group(1))
        
        # Extract effects
        effect_match = re.search(r':effect\s*\((.*?)\)\s*\)', body, re.DOTALL | re.IGNORECASE)
        if effect_match:
            action['effects'] = self._parse_effects(effect_match.group(1))
        
        return action
    
    def _parse_parameters(self, text):
        # Simple parameter extraction
        params = []
        tokens = text.split()
        for i, token in enumerate(tokens):
            if token.startswith('?'):
                param_name = token
                param_type = tokens[i+2] if i+2 < len(tokens) and tokens[i+1] == '-' else 'object'
                params.append({'name': param_name, 'type': param_type})
        return params
    
    def _parse_conditions(self, text):
        conditions = []
        # Handle 'and' wrapper
        text = re.sub(r'^\s*and\s*', '', text, flags=re.IGNORECASE)
        
        # Extract individual predicates
        pred_pattern = r'\(([^\(\)]+)\)'
        for match in re.finditer(pred_pattern, text):
            pred = match.group(1).strip()
            if pred and not pred.lower().startswith('and'):
                conditions.append(pred)
        
        return conditions
    
    def _parse_effects(self, text):
        effects = []
        # Handle 'and' wrapper
        text = re.sub(r'^\s*and\s*', '', text, flags=re.IGNORECASE)
        
        # Extract individual effects (including 'not' effects)
        pred_pattern = r'\((not\s+)?\(([^\)]+)\)\)|\(([^\(\)]+)\)'
        for match in re.finditer(pred_pattern, text):
            if match.group(2):  # 'not' effect
                effects.append({'type': 'delete', 'predicate': match.group(2).strip()})
            elif match.group(3):  # positive effect
                pred = match.group(3).strip()
                if pred and not pred.lower().startswith('and'):
                    effects.append({'type': 'add', 'predicate': pred})
        
        return effects
    
    def __str__(self):
        output = f"Domain: {self.domain_name}\n"
        output += f"\nPredicates ({len(self.predicates)}):\n"
        for pred in self.predicates:
            output += f"  - {pred}\n"
        output += f"\nActions ({len(self.actions)}):\n"
        for action in self.actions:
            output += f"  - {action['name']}\n"
        return output
