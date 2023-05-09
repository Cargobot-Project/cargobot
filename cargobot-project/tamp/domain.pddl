(define (domain box-placement)
(:requirements :typing :strips)
(:types
	; Boxes, which are the only moveable objects
	location locatable - object
		bot box - locatable
	robot - bot
)	

(:predicates
	; Is object on given location?
	(on ?obj - locatable ?loc - location)

	; Is robot holding box?
	(holding ?arm - locatable ?box - box)

	; Is the arm free?
	(arm-empty)

	; Is the location occupied?
	(occupied ?loc - location)

	; Box type checking
	(is_highprio ?box - box)
	(is_lowprio ?box - box)
	(is_light ?box - box)
	(is_heavy ?box - box)

	; (Truck) location checking
	(is_ground ?location - location)
	(has_heavy_below ?location - location)
	(has_highprio_behind ?location - location)
)

(:action pick-up
	:parameters
	(?arm - bot
	 ?box - locatable
	 ?loc - location)
	:precondition
	(and
		(on ?arm ?loc)
		(on ?box ?loc)
		(arm-empty)
	)
	:effect
	(and
		(not (on ?box ?loc))
		(holding ?arm ?box)
		(not (arm-empty))
	)
)

(:action drop
	:parameters
	(?arm - bot
	 ?box - box
	 ?loc - location
	)
	:precondition
	(and
		(on ?arm ?loc)
		(holding ?arm ?box)
		(not (occupied ?loc))
		
		; lowprio boxes should not have highprio behind
		(not
			(and
				(is_highprio ?box)
				(not (has_highprio_behind ?loc))
			)
		)
		; heavy boxes can either be put on the ground or only have heavy boxes below
		(not
			(and
				(is_heavy ?box)
				(or
					(not (has_heavy_below ?loc))
					(is_ground ?loc)
				)
				
			)
		)
	)
	:effect
	(and
		(on ?box ?loc)
		(arm-empty)
		(not (holding ?arm ?box))
	)
)

(:action move
    :parameters
    (
    ?r - bot
    ?loc1 - location
    ?loc2 - location
    )

    :precondition 
    (and 
        (on ?r ?loc1)
    )

    :effect
    (and 
        (on ?r ?loc2)
        (not (on ?r ?loc1))
    )
)
)