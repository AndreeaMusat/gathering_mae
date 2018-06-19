# Element vector description index
E_ASCII_K = 0
E_RGB_K = 1
E_NAME_K = 2

# Static + Special elements cannot overlap (unique id)
# Cannot Move into them.
DEFAULT_ELEMENT_ID = 0  # ID for empty space for all maps (! MUST BE 0 COLORED)
WALL_ID = 1
STATIC_ELEMENTS = [
    [".", (0, 0, 0), "empty"],
    ["w", (211, 211, 211), "wall"],
]

REWARD_ID = 0
ALL_REWARD_IDS = [0, 1]
LASER_ID = 2
SPECIAL_ELEMENTS = [
    ["o", (124, 252, 0), "reward"],
    ["x", (255, 128, 0), "neg_reward"],
    ["l", (231, 172, 65), "laser"],
]

DIRECTION_EL = ["s", (100, 100, 100), "direction"]

AGENTS_COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0)]

ACTIONS = {
    0: "up",    # 0, 1, 2, 3 (Represent direction as well)
    1: "right",
    2: "down",
    3: "left",
    4: "null",
    5: "turn_c",
    6: "turn_cc",
    7: "action"
}
ACTION_M_MOVE = 4
ACTION_NULL = 5
ACTION_TURN_C = 5
ACTION_TURN_CC = 6
ACTION_ACTION = 7

ACTIONS_MOVE = [(-1, 0), (0, 1), (1, 0), (0, -1)]
TURN_C = [1, 2, 3, 0]
TURN_CC = [3, 0, 1, 2]
AGENT_MAX_STATES = 4

FULL_VISUALIZE_SIZE = (250, 250)
PARTIAL_VISUALIZE_SIZE = (100, 100)
