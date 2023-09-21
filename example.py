import json
from game import Game
import random

import rust_perf

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

with open("map.json") as f:
    global_map = json.load(f)["map"]

global_coin_set = set()
ban_idx = set()
for cell in global_map:
    x = cell["x"]
    y = cell["y"]
    cell_type = cell["type"]
    if cell_type == "COIN":
        global_coin_set.add((x, y))
eatten_set = set()


def get_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def handle_agent(agent) -> str:
    current_pos = (agent["self_agent"]["x"], agent["self_agent"]["y"])
    print(agent["self_agent"]["id"], current_pos)

    if current_pos in global_coin_set:
        eatten_set.add(current_pos)

    other_agent_list = agent["other_agents"]
    attacker_location = []
    for other_agent in other_agent_list:
        if other_agent["role"] == "ATTACKER":
            attacker_location.append((other_agent["x"], other_agent["y"]))

    if attacker_location is None:
        attacker_location = (22, 0)

    # print(current_pos, attacker_location)
    path, expected_score = rust_perf.collect_coins_with_enemy(
        current_pos, attacker_location, eatten_set
    )
    print(path, expected_score)
    if len(path) == 0:
        print(current_pos, attacker_location, eatten_set)
        raise Exception("e")
        return random.choice(ACTIONS)
    else:
        return get_direction(current_pos, path[0])
    # return rust_perf.get_direction(current_pos, move_to, block_list)


def get_direction(curr, next):
    if curr[1] == next[1]:
        if next[0] > curr[0]:
            return "RIGHT"
        else:
            return "LEFT"
    elif curr[0] == next[0]:
        if next[1] > curr[1]:
            return "UP"
        else:
            return "DOWN"
    else:
        return "STAY"


# load map
with open("map.json") as f:
    map = json.load(f)
game = Game(map)


# init game
# seed = random.randint(0, 10000)
seed = 8888
game.reset_game(attacker="attacker", defender="defender", seed=seed)

# game loop
while not game.is_over():
    # get game state for player:
    attacker_state = game.get_agent_states_by_player("attacker")
    defender_state = game.get_agent_states_by_player("defender")

    # apply actions for agents:
    attacker_actions = {_id: random.choice(ACTIONS) for _id in attacker_state.keys()}
    # defender_actions =  { _id: random.choice(ACTIONS) for _id in defender_state.keys() }
    # for k, v in defender_state.items():
    #     print(v)
    #     raise Exception("e")
    defender_actions = {
        _id: handle_agent(defender_state[_id]) for _id in defender_state.keys()
    }
    print("step: ", game.steps)
    # print(defender_actions)
    # print(defender_state)
    total_score = 0
    for _, v in defender_state.items():
        total_score += v["self_agent"]["score"]
    print("Score: ", total_score)
    game.apply_actions(
        attacker_actions=attacker_actions, defender_actions=defender_actions
    )


# get game result
print("seed: ", seed)
print("game result:\r\n", game.get_result())
