# -*- coding: utf-8 -*-
"""BASELINE.

__author__ = Yue Peng
__copyright__ = Copyright 2022
__version__ = 1.0.0
__maintainer__ = Yue Peng
__email__ = yuepaang@gmail.com
__status__ = Dev
__filename__ = baseline.py
__uuid__ = 267eb52b-88ff-487f-9816-8ebc77769c52
__date__ = 2023-10-08 23:38:55
__modified__ =
"""

from collections import defaultdict
import json
import time
from game import Game, Powerup
import random

import rust_perf

ACTIONS = ["STAY", "LEFT", "RIGHT", "DOWN", "UP"]

with open("map.json") as f:
    global_map = json.load(f)["map"]

global_coin_set = set()
global_powerup_set = set()
global_walls_list = []
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
    if cell_type == "WALL":
        global_walls_list.append((x, y))


# pre calculate real distance
# with open("dist.csv", "w", encoding="utf-8") as f:
#     for i in range(24):
#         for j in range(24):
#             start = (i, j)
#             if start in global_walls_list:
#                 continue
#             for m in range(i, 24):
#                 for n in range(j, 24):
#                     end = (m, n)
#                     if end in global_walls_list:
#                         continue
#                     print(f"{start}${end}\n")
#                     dist = len(rust_perf.get_direction_path(start, end, [])[1])
#                     print(f"{start}${end}${dist}\n")
#                     f.write(f"{start}${end}${dist}\n")
# raise Exception("e")

# load shortest path using astar with portals
dist_map = defaultdict(dict)
with open("dist.csv", "r", encoding="utf-8") as f:
    for line in f.readlines():
        fields = line.split("$")
        dist_map[eval(fields[0])][eval(fields[1])] = int(fields[2])


def get_distance(pos1, pos2):
    if pos1 in dist_map:
        if pos2 in dist_map[pos1]:
            return dist_map[pos1][pos2]
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def use_attacker(agent, enemies, powerup_clock) -> str:
    # record powerups
    if "passwall" in agent["self_agent"]["powerups"]:
        passwall = agent["self_agent"]["powerups"]["passwall"]
    else:
        passwall = 0
    current_pos = (agent["self_agent"]["x"], agent["self_agent"]["y"])

    # record locations have been arrived
    if current_pos in global_powerup_set:
        powerup_clock[current_pos] = 1

    cancel_key = []
    for powerup, clock in powerup_clock.items():
        if clock == 12:
            print(powerup_clock)
            raise Exception("e")
            cancel_key.append(powerup)
    for k in cancel_key:
        del powerup_clock[k]

    path = rust_perf.catch_enemies_using_powerup(
        current_pos,
        passwall,
        enemies,
    )
    if len(path) == 0:
        print(
            agent["self_agent"]["id"],
            current_pos,
            agent["self_agent"]["score"],
            passwall,
            enemies,
        )
        return random.choice(ACTIONS)

    # print(
    #     path,
    #     agent["self_agent"]["id"],
    #     current_pos,
    #     agent["self_agent"]["score"],
    #     passwall,
    #     enemies,
    # )
    next_move = get_direction(current_pos, path[0])
    for powerup, _ in powerup_clock.items():
        powerup_clock[powerup] += 1
    return next_move


def use_defender(
    agent, eaten_set, step, powerup_clock, attacker_location, defender_scatter
) -> str:
    # safe phrase
    agent_id = agent["self_agent"]["id"]

    for p in agent["powerups"]:
        if p["powerup"] == str(Powerup.SWORD):
            eaten_set.add((p["x"], p["y"]))
        elif (p["x"], p["y"]) in eaten_set:
            eaten_set.remove((p["x"], p["y"]))

    # record powerups
    if "passwall" in agent["self_agent"]["powerups"]:
        passwall = agent["self_agent"]["powerups"]["passwall"]
    else:
        passwall = 0

    if "shield" in agent["self_agent"]["powerups"]:
        shield = agent["self_agent"]["powerups"]["shield"]
    else:
        shield = 0

    current_pos = (agent["self_agent"]["x"], agent["self_agent"]["y"])

    # scatter first
    if step < 7:
        if agent_id not in defender_scatter:
            return rust_perf.get_direction(
                current_pos,
                random.choice([(0, 12), (18, 17), (11, 11)]),
                list(eaten_set),
            )
        return rust_perf.get_direction(
            current_pos,
            defender_scatter[agent_id],
            list(eaten_set),
        )

    # record locations have been arrived
    if current_pos in global_coin_set:
        eaten_set.add(current_pos)
    if current_pos in global_powerup_set:
        eaten_set.add(current_pos)
        powerup_clock[current_pos] = 1

    cancel_key = []
    for powerup, clock in powerup_clock.items():
        if clock == 12:
            if powerup in eaten_set:
                eaten_set.remove(powerup)
            cancel_key.append(powerup)
    for k in cancel_key:
        del powerup_clock[k]

    # enemies in out vision
    other_agent_list = agent["other_agents"]
    allies_location = []
    has_sword = False
    for other_agent in other_agent_list:
        if other_agent["role"] == "DEFENDER":
            allies_location.append((other_agent["x"], other_agent["y"]))
        else:
            if "sword" in other_agent["powerups"]:
                has_sword = True

    # strategy one (corner)
    if (len(attacker_location) >= 1 and shield < 3) or has_sword:
        next_move = rust_perf.check_stay_or_not(
            current_pos, attacker_location, passwall
        )
        # print(agent_id, current_pos, attacker_location, next_move, passwall)
        return next_move

    if agent_id == 4:
        # print(
        #     current_pos, agent["self_agent"]["score"], len(eaten_set), passwall, shield
        # )
        path = rust_perf.collect_coins_using_hull(current_pos, eaten_set)
        if len(path) > 0:
            return get_direction(current_pos, path[0])
        else:
            path, _ = rust_perf.collect_coins_using_powerup(
                current_pos,
                eaten_set,
                allies_location,
                attacker_location,
                passwall,
            )
            return get_direction(current_pos, path[0])
    else:
        path, _ = rust_perf.collect_coins_using_powerup(
            current_pos,
            eaten_set,
            allies_location,
            attacker_location,
            passwall,
        )
        if len(path) == 0:
            return random.choice(ACTIONS)
        else:
            next_move = get_direction(current_pos, path[0])
            for powerup, _ in powerup_clock.items():
                powerup_clock[powerup] += 1
            return next_move


def get_direction(curr, next):
    true_next = next

    if curr[1] == true_next[1]:
        if true_next[0] == curr[0]:
            return "STAY"
        elif true_next[0] == curr[0] + 1:
            return "RIGHT"
        elif true_next[0] == curr[0] - 1:
            return "LEFT"
    elif curr[0] == true_next[0]:
        if true_next[1] == curr[1] + 1:
            return "DOWN"
        elif true_next[1] == curr[1] - 1:
            return "UP"


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
# seeds = [990986]
for seed in seeds:
    game.reset(attacker="attacker", defender="defender", seed=seed)

    eatten_set = set()
    step = 0
    powerup_clock = {}
    defender_scatter = {4: (0, 12), 5: (18, 17), 6: (11, 11), 7: (20, 9)}
    start_game_time = time.time()
    # game loop
    while not game.is_over():
        # get game state for player:
        attacker_state = game.get_agent_states_by_player("attacker")
        defender_state = game.get_agent_states_by_player("defender")

        attacker_locations = set()
        defender_locations = set()
        for k, v in defender_state.items():
            other_agent_list = v["other_agents"]
            for other_agent in other_agent_list:
                if other_agent["role"] == "ATTACKER":
                    attacker_locations.add((other_agent["x"], other_agent["y"]))
                elif other_agent["invulnerability_duration"] == 0:
                    defender_locations.add((other_agent["x"], other_agent["y"]))

        attacker_actions = {
            _id: random.choice(ACTIONS) for _id in attacker_state.keys()
        }
        # attacker_actions = {
        #     _id: use_attacker(attacker_state[_id], list(defender_locations), {})
        #     for _id in attacker_state.keys()
        # }

        defender_actions = {
            _id: random.choice(ACTIONS) for _id in defender_state.keys()
        }
        defender_actions = {
            _id: use_defender(
                defender_state[_id],
                eatten_set,
                step,
                powerup_clock,
                list(attacker_locations),
                defender_scatter,
            )
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

    defender_state = game.get_agent_states_by_player("defender")
    for k, v in defender_state.items():
        print(k, v["self_agent"]["score"])

print("Win rate is ", win_count / len(seeds))
print(f"Attacker score: {attacker_score} vs Defender score: {defender_score}")
