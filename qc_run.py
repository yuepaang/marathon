import json
import time
import random
import numpy as np
from game import Game
from model.wolf_attacker.attacker import Attacker
from model.utils import *

# import rust_perf
# x = rust_perf.get_direction_path((0, 0), (22, 23), [(1, 2), (1, 10)])
# print(x)
# assert False

ACTIONS = ["STAY", "LEFT", "RIGHT", "DOWN", "UP"]

# load map
with open("map.json") as f:
    map = json.load(f)
game = Game(map)

map_size = (map["map_conf"]["height"], map["map_conf"]["width"])

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
    attacker_obs = game.get_agent_states_by_player("attacker")
    defender_obs = game.get_agent_states_by_player("defender")

    attacker_ids = sorted(list(attacker_obs.keys()))
    defender_ids = sorted(list(defender_obs.keys()))

    attacker = Attacker(map_size, attacker_ids, defender_ids, get_walls())

    eatten_set = set()
    step = 0
    start_game_time = time.time()

    # game loop
    while not game.is_over():
        step_start = time.time()
        # get game state for player:
        attacker_obs = game.get_agent_states_by_player("attacker")
        defender_obs = game.get_agent_states_by_player("defender")
        print_map_state(game.get_map_states(), map_size)
        attacker.update(attacker_obs)

        # apply actions for agents:
        attacker_actions = attacker.step()
        defender_actions = {
            _id: random.choice(ACTIONS)
            for _id in defender_obs.keys()
        }
        # defender_actions = {_id: "STAY" for _id in defender_obs.keys()}
        # print(round(time.time() - step_start, 3))
        # print()

        game.apply_actions(attacker_actions=attacker_actions,
                           defender_actions=defender_actions)
        step += 1
        time.sleep(0.5)
        # print(f"{step}/1152")

    # get game result
    print(f"seed: {seed} --- game result:\r\n", game.get_result())
    print("elasped time: ", time.time() - start_game_time, "s")
    if (game.get_result()["players"][0]["score"] <
            game.get_result()["players"][1]["score"]):
        win_count += 1
    attacker_score += game.get_result()["players"][0]["score"]
    defender_score += game.get_result()["players"][1]["score"]

print("Win rate is ", win_count / len(seeds))
print(f"Attacker score: {attacker_score} vs Defender score: {defender_score}")
