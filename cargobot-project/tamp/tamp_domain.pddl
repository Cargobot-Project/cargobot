(define (domain box-placement)
(:requirements :typing :strips)
(:types
	; Boxes, which are the only moveable objects
	location locatable - object
		bot box - locatable
	highprio lowprio heavy light - box
	robot - bot
    path - object
    pose - object
)	

(:predicates
	; Is object on given location?
	(on ?obj - locatable ?loc - location)
	; Is robot holding box?
	(holding ?arm - locatable ?box - box)
	; Is the arm free?
	(arm-empty)
	; Is there a valid path between two locs?
	(path ?location1 - location ?location2 - location)
    ; Does the object have given pose?
    (has-pose ?obj - locatable ?pose - pose)

	; Box type checking
	(is_highprio ?box - box)
	(is_lowprio ?box - box)
	(is_light ?box - box)
	(is_heavy ?box - box)


    (path-length ?path - path)
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
    ?pose1 - pose
    ?pose2 - pose
    ?path - path
    )

    :precondition 
    (and 
        (on ?r ?loc1)
        (has-pose ?r ?pose1)
        (NavPose ?loc2 ?pose2)
        (Motion ?pose1 ?pose2 ?path)
    )

    :effect
    (and 
        (on ?r ?loc2)
        (not (on ?r ?loc1))
        (has-pose ?r ?pose2)
        (not (has-pose ?r ?pose1))
        (increase (total-cost) (path-length ?path))
    )
)
)