from pddl_parser import PDDLParser

class PDDLToMeld:
    """Converts PDDL domain to SME .meld format."""
    
    def __init__(self, pddl_parser):
        self.parser = pddl_parser
        self.meld_facts = []
    
    def convert(self):
        """Convert PDDL domain to .meld facts."""
        self.meld_facts = []
        
        # Create microtheory declaration
        mt_name = f"{self.parser.domain_name}Mt"
        
        # Convert each action to causal relationships
        for action in self.parser.actions:
            self._convert_action(action)
        
        return mt_name, self.meld_facts
    
    def _convert_action(self, action):
        """Convert a PDDL action to SME causal relationships."""
        action_name = action['name']
        
        # Create action entity
        action_entity = f"action-{action_name}"
        
        # Add preconditions as relations
        for i, precond in enumerate(action['preconditions']):
            precond_name = self._sanitize_predicate(precond)
            precond_relation = f"precond-{action_name}-{i}"
            
            # Create precondition fact
            self.meld_facts.append(f"({precond_name} {action_entity})")
        
        # Add effects as relations
        for i, effect in enumerate(action['effects']):
            effect_pred = effect['predicate']
            effect_name = self._sanitize_predicate(effect_pred)
            effect_type = effect['type']
            effect_relation = f"effect-{effect_type}-{action_name}-{i}"
            
            # Create effect fact
            self.meld_facts.append(f"({effect_name} {action_entity})")
        
        # Create causal relationships between preconditions and effects
        if action['preconditions'] and action['effects']:
            # Create compound precondition
            precond_compound = f"preconditions-{action_name}"
            for i, precond in enumerate(action['preconditions']):
                precond_name = self._sanitize_predicate(precond)
                self.meld_facts.append(f"(has-precond {precond_compound} {precond_name})")
            
            # Create compound effect
            effect_compound = f"effects-{action_name}"
            for i, effect in enumerate(action['effects']):
                effect_name = self._sanitize_predicate(effect['predicate'])
                self.meld_facts.append(f"(has-effect {effect_compound} {effect_name})")
            
            # Create causal link
            self.meld_facts.append(f"(action-type {action_entity} planning-action)")
            self.meld_facts.append(f"(cause {precond_compound} {effect_compound})")
        
        # Add action capabilities
        for param in action['parameters']:
            param_type = param['type']
            self.meld_facts.append(f"(operates-on {action_entity} {param_type})")
    
    def _sanitize_predicate(self, predicate):
        """Sanitize predicate name for SME format."""
        # Remove parameters and special characters
        pred = predicate.split()[0] if ' ' in predicate else predicate
        pred = pred.replace('?', '').replace('-', '_')
        return pred
    
    def write_meld_file(self, output_path):
        """Write the converted domain to a .meld file."""
        mt_name, facts = self.convert()
        
        with open(output_path, 'w') as f:
            f.write(f"(in-microtheory {mt_name})\n")
            for fact in facts:
                f.write(f"{fact}\n")
        
        return output_path
