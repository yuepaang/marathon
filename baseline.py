import json
import time
from game import Game
import random

import rust_perf

ACTIONS = ["STAY", "LEFT", "RIGHT", "DOWN", "UP"]

with open("map.json") as f:
    global_map = json.load(f)["map"]

global_coin_set = set()
global_powerup_set = set()
global_portal_map = {}
ban_idx = set()
for cell in global_map:
    x = cell["x"]
    y = cell["y"]
    cell_type = cell["type"]
    if cell_type == "COIN":
        global_coin_set.add((x, y))
    if cell_type == "POWERUP":
        global_powerup_set.add((x, y))
    if cell_type == "PORTAL":
        global_portal_map[(x, y)] = (cell["pair"]["x"], cell["pair"]["y"])


def get_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def use_defender(agent, eaten_set, step) -> str:
    # safe phrase

    agent_id = agent["self_agent"]["id"]

    # record powerups
    if "passwall" in agent["self_agent"]["powerups"]:
        passwall = agent["self_agent"]["powerups"]["passwall"]
    else:
        passwall = 0
    current_pos = (agent["self_agent"]["x"], agent["self_agent"]["y"])

    # record locations have been arrived
    if current_pos in global_coin_set:
        eaten_set.add(current_pos)
    if current_pos in global_powerup_set:
        eaten_set.add(current_pos)

    # attack powerup
    for owned_powerup in agent["powerups"]:
        if owned_powerup.get("powerup") == 4:
            eaten_set.add((owned_powerup["x"], owned_powerup["y"]))

    # enemies in out vision
    other_agent_list = agent["other_agents"]
    attacker_location = []
    allies_location = []
    for other_agent in other_agent_list:
        if other_agent["role"] == "ATTACKER":
            attacker_location.append((other_agent["x"], other_agent["y"]))
        else:
            if step > 3:
                allies_location.append((other_agent["x"], other_agent["y"]))

    # strategy one
    if len(attacker_location) == 2:
        next_move = rust_perf.check_two_enemies_move(current_pos, attacker_location)
        if next_move != "NO":
            # raise Exception("early")
            return next_move

    # path, _ = rust_perf.collect_coins(current_pos, eatten_set)
    path, _ = rust_perf.collect_coins_using_powerup(
        step,
        agent_id,
        current_pos,
        eaten_set,
        allies_location,
        attacker_location,
        passwall,
    )
    if len(path) == 0:
        print(
            agent_id,
            agent["self_agent"]["score"],
            current_pos,
            passwall,
            agent["self_agent"]["powerups"],
            attacker_location,
            len(eaten_set),
            eaten_set,
        )
        print(agent["other_agents"])
        raise Exception("no path")
        return random.choice(ACTIONS)
    else:
        next_move = get_direction(current_pos, path[0])
        # print(
        #     agent_id,
        #     agent["self_agent"]["score"],
        #     current_pos,
        #     next_move,
        #     attacker_location,
        #     allies_location,
        #     # "path: ",
        #     # ["COIN" if p in global_coin_set else p for p in path],
        #     # potential_score,
        # )
        # print(agent["self_agent"]["id"], path, current_pos, next_move, score)

        # if next_move == "ERROR":
        #     print(
        #         agent_id,
        #         current_pos,
        #         "path: ",
        #         ["COIN" if p in global_coin_set else p for p in path],
        #         potential_score,
        #     )
        #     next_move = "STAY"
        return next_move
    # # return rust_perf.get_direction(current_pos, move_to, block_list)


def get_direction(curr, next):
    true_next = next

    if curr[1] == true_next[1]:
        if true_next[0] == curr[0]:
            return "STAY"
        elif true_next[0] == curr[0] + 1:
            return "RIGHT"
        elif true_next[0] == curr[0] - 1:
            return "LEFT"
        else:
            return "ERROR"
    elif curr[0] == true_next[0]:
        if true_next[1] == curr[1] + 1:
            return "DOWN"
        elif true_next[1] == curr[1] - 1:
            return "UP"
        else:
            return "ERROR"
    else:
        print(curr, true_next, next)
        raise Exception("e")
        return "ERROR"


# load map
with open("map.json") as f:
    map = json.load(f)
game = Game(map)


# init game
# seed = random.randint(0, 10000)
# seed = 8878
win_count = 0
attacker_score = 0
defender_score = 0
seeds = [random.randint(0, 1000000) for _ in range(5)]
# seeds = [7437]
for seed in seeds:
    game.reset(attacker="attacker", defender="defender", seed=seed)

    eatten_set = set()
    step = 0
    start_game_time = time.time()
    # game loop
    while not game.is_over():
        # get game state for player:
        attacker_state = game.get_agent_states_by_player("attacker")
        defender_state = game.get_agent_states_by_player("defender")

        # apply actions for agents:
        attacker_actions = {
            _id: random.choice(ACTIONS) for _id in attacker_state.keys()
        }
        defender_actions = {
            _id: use_defender(defender_state[_id], eatten_set, step)
            for _id in defender_state.keys()
        }

        game.apply_actions(
            attacker_actions=attacker_actions, defender_actions=defender_actions
        )
        step += 1
        # print(f"{step}/1152")

    # get game result
    print(f"seed: {seed} --- game result:\r\n", game.get_result())
    print("elasped time: ", time.time() - start_game_time, "s")
    if (
        game.get_result()["players"][0]["score"]
        < game.get_result()["players"][1]["score"]
    ):
        win_count += 1
    attacker_score += game.get_result()["players"][0]["score"]
    defender_score += game.get_result()["players"][1]["score"]

print("Win rate is ", win_count / len(seeds))
print(f"Attacker score: {attacker_score} vs Defender score: {defender_score}")