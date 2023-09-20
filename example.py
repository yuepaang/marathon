import json
from game import Game
import random

import rust_perf


with open("map.json") as f:
    global_map = json.load(f)["map"]

global_coin_list = list()
global_wall_list = list()
ban_idx = set()
for cell in global_map:
    x = cell["x"]
    y = cell["y"]
    cell_type = cell["type"]
    if cell_type == "COIN":
        global_coin_list.append((x, y))
    if cell_type == "WALL":
        global_wall_list.append((x, y))


def get_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def handle_agent(agent) -> str:
    current_pos = (agent["self_agent"]["x"], agent["self_agent"]["y"])

    for i, coin in enumerate(global_coin_list):
        if coin[0] == current_pos[0] and coin[1] == current_pos[1]:
            ban_idx.add(i)
            break

    other_agent_list = agent["other_agents"]

    block_list = [wall for wall in global_wall_list]
    attacker_location = set()
    for other_agent in other_agent_list:
        if other_agent["role"] == "ATTACKER":
            block_list.append((other_agent["x"], other_agent["y"]))
            attacker_location.add((other_agent["x"], other_agent["y"]))

    coins_distance_list = [
        get_distance(current_pos, (coin[0], coin[1]))
        if i not in ban_idx and (coin not in attacker_location)
        else 999
        for i, coin in enumerate(global_coin_list)
    ]
    if min(coins_distance_list) == 999:
        raise Exception("no road")
    nearest_coin_idx = coins_distance_list.index(min(coins_distance_list))
    move_to = (
        global_coin_list[nearest_coin_idx][0],
        global_coin_list[nearest_coin_idx][1],
    )
    return rust_perf.get_direction(current_pos, move_to, block_list)


# load map
with open("map.json") as f:
    map = json.load(f)
game = Game(map)


# init game
seed = random.randint(0, 10000)
game.reset_game(attacker="attacker", defender="defender", seed=seed)

# game loop
while not game.is_over():
    # get game state for player:
    attacker_state = game.get_agent_states_by_player("attacker")
    defender_state = game.get_agent_states_by_player("defender")

    # apply actions for agents:
    ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
    attacker_actions = {_id: random.choice(ACTIONS) for _id in attacker_state.keys()}
    # defender_actions =  { _id: random.choice(ACTIONS) for _id in defender_state.keys() }
    # for k, v in defender_state.items():
    #     print(v)
    #     raise Exception("e")
    defender_actions = {
        _id: handle_agent(defender_state[_id]) for _id in defender_state.keys()
    }
    # print(game.steps)
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
