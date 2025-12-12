"""
Example of using pddl_similarity with PDDL strings
"""

from pddl_similarity import calculate_similarity_from_strings

# Define two PDDL domains as strings
blocksworld_pddl = """
(define (domain blocksworld)
  (:requirements :strips)
  
  (:predicates
    (on ?x ?y)
    (ontable ?x)
    (clear ?x)
    (handempty)
    (holding ?x)
  )
  
  (:action pick-up
    :parameters (?x)
    :precondition (and (clear ?x) (ontable ?x) (handempty))
    :effect (and (not (ontable ?x)) (not (clear ?x)) (not (handempty)) (holding ?x))
  )
  
  (:action put-down
    :parameters (?x)
    :precondition (holding ?x)
    :effect (and (not (holding ?x)) (clear ?x) (handempty) (ontable ?x))
  )
)
"""

logistics_pddl = """
(define (domain logistics)
  (:requirements :strips)
  
  (:predicates
    (at ?obj ?loc)
    (in ?pkg ?veh)
    (vehicle ?v)
    (package ?p)
  )
  
  (:action load-package
    :parameters (?pkg ?veh ?loc)
    :precondition (and (at ?pkg ?loc) (at ?veh ?loc))
    :effect (and (not (at ?pkg ?loc)) (in ?pkg ?veh))
  )
  
  (:action unload-package
    :parameters (?pkg ?veh ?loc)
    :precondition (and (in ?pkg ?veh) (at ?veh ?loc))
    :effect (and (not (in ?pkg ?veh)) (at ?pkg ?loc))
  )
)
"""

# Calculate similarity
print("Comparing two PDDL domains from strings...")
mappings = calculate_similarity_from_strings(blocksworld_pddl, logistics_pddl)

if mappings:
    best_score = max(m.score for m in mappings)
    print(f"\nFound {len(mappings)} mapping(s)")
    print(f"Best Similarity Score: {best_score:.4f}")
else:
    print("\nNo similarities found")
    print("Best Similarity Score: 0.0000")
