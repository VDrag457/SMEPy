(define (domain logistics)
  (:requirements :strips)
  
  (:predicates
    (at ?obj ?loc)
    (in ?pkg ?veh)
    (vehicle ?v)
    (package ?p)
    (location ?l)
  )
  
  (:action load-package
    :parameters (?pkg ?veh ?loc)
    :precondition (and (at ?pkg ?loc) (at ?veh ?loc) (package ?pkg) (vehicle ?veh))
    :effect (and (not (at ?pkg ?loc)) (in ?pkg ?veh))
  )
  
  (:action unload-package
    :parameters (?pkg ?veh ?loc)
    :precondition (and (in ?pkg ?veh) (at ?veh ?loc))
    :effect (and (not (in ?pkg ?veh)) (at ?pkg ?loc))
  )
  
  (:action drive-vehicle
    :parameters (?veh ?from ?to)
    :precondition (and (at ?veh ?from) (vehicle ?veh) (location ?from) (location ?to))
    :effect (and (not (at ?veh ?from)) (at ?veh ?to))
  )
  
  (:action pickup-item
    :parameters (?item ?loc)
    :precondition (and (at ?item ?loc) (location ?loc))
    :effect (and (not (at ?item ?loc)) (in ?item carrier))
  )
)
