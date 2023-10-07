from collections import defaultdict
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


def use_attacker(agent, step, powerup_clock) -> str:
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
            cancel_key.append(powerup)
    for k in cancel_key:
        del powerup_clock[k]

    # attack powerup
    # for owned_powerup in agent["powerups"]:
    #     if owned_powerup.get("powerup") == 4:
    #         eaten_set.add((owned_powerup["x"], owned_powerup["y"]))

    # enemies in out vision
    other_agent_list = agent["other_agents"]
    attacker_location = []
    for other_agent in other_agent_list:
        if other_agent["role"] == "DEFENDER":
            attacker_location.append((other_agent["x"], other_agent["y"]))

    path = rust_perf.catch_enemies_using_powerup(
        current_pos,
        passwall,
        step,
    )
    next_move = get_direction(current_pos, path[0])
    for powerup, _ in powerup_clock.items():
        powerup_clock[powerup] += 1
    return next_move


def use_defender(agent, eaten_set, step, powerup_clock, attacker_location) -> str:
    # safe phrase
    agent_id = agent["self_agent"]["id"]

    for p in agent["powerups"]:
        if p["powerup"] == "4" or p["powerup"] == 4:
            eaten_set.add((p["x"], p["y"]))

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
    if step < 5:
        return rust_perf.get_direction(
            current_pos,
            random.choice([(0, 12), (18, 17), (11, 11)]),
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
            eaten_set.remove(powerup)
            cancel_key.append(powerup)
    for k in cancel_key:
        del powerup_clock[k]

    # enemies in out vision
    other_agent_list = agent["other_agents"]
    allies_location = []
    for other_agent in other_agent_list:
        if other_agent["role"] == "DEFENDER":
            allies_location.append((other_agent["x"], other_agent["y"]))

    # strategy one (corner)
    if len(attacker_location) >= 1 and shield <= 3:
        next_move = rust_perf.check_stay_or_not(
            current_pos, attacker_location, passwall
        )
        # print(agent_id, current_pos, attacker_location, next_move, passwall)
        if next_move != "NO":
            return next_move

    # path, _ = rust_perf.collect_coins(current_pos, eatten_set)
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
seeds = [327705]
for seed in seeds:
    game.reset(attacker="attacker", defender="defender", seed=seed)

    eatten_set = set()
    step = 0
    powerup_clock = {}
    score_cache = defaultdict(int)
    start_game_time = time.time()
    # game loop
    while not game.is_over():
        # get game state for player:
        attacker_state = game.get_agent_states_by_player("attacker")
        defender_state = game.get_agent_states_by_player("defender")

        attacker_location = []
        for k, v in defender_state.items():
            other_agent_list = v["other_agents"]
            for other_agent in other_agent_list:
                if other_agent["role"] == "ATTACKER":
                    attacker_location.append((other_agent["x"], other_agent["y"]))

            # if v["self_agent"]["score"] < score_cache[k]:
            #     print(k, v["self_agent"]["score"], score_cache[k])
            #     print("seed", seed)
            #     raise Exception("e")
            # score_cache[k] = v["self_agent"]["score"]
        # apply actions for agents:
        attacker_actions = {
            _id: random.choice(ACTIONS) for _id in attacker_state.keys()
        }
        # attacker_actions = {
        #     _id: use_attacker(attacker_state[_id], step, {})
        #     for _id in attacker_state.keys()
        # }
        defender_actions = {
            _id: use_defender(
                defender_state[_id], eatten_set, step, powerup_clock, attacker_location
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
