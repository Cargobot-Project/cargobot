from enum import Enum

class PlannerState(Enum):
    WAIT_FOR_OBJECTS_TO_SETTLE = 1
    PICKING_BOX = 2
    EMPTY_TRUCK = 3
    GO_HOME = 4

class BoxColorEnum(Enum):
    RED = 0
    BLUE = 1
    GREEN = 2
    MAGENTA = 3
    YELLOW = 4
    CYAN = 5

class LabelEnum(Enum):
    LOW_PRIORTY = 0
    HIGH_PRIORTY = 1
    LIGHT = 2
    HEAVY = 3
    MID_PRIORTY = 4