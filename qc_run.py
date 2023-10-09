import json
import time
import random
import numpy as np
from game import Game
from map.map import Map
from model.defender import Defender
from model.utils import print_map_state

ACTIONS = ["STAY", "LEFT", "RIGHT", "DOWN", "UP"]

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
    attacker_obs = game.get_agent_states_by_player("attacker")
    defender_obs = game.get_agent_states_by_player("defender")

    attacker_ids = sorted(list(attacker_obs.keys()))
    defender_ids = sorted(list(defender_obs.keys()))

    defender_map = Map()
    defender_map.load_map(map["map"])
    eatten_set = set()
    step = 0
    start_game_time = time.time()

    defender = Defender(
        defender_ids,
        attacker_ids,
        defender_map,
    )
    # game loop
    while not game.is_over():
        step_start = time.time()
        # get game state for player:
        attacker_obs = game.get_agent_states_by_player("attacker")
        defender_obs = game.get_agent_states_by_player("defender")
        defender.map.update_map(defender_obs)
        defender.update(defender_obs)

        opp_pos = dict()
        for id, view in attacker_obs.items():
            opp_pos[id] = defender_map.obs_to_map_coor(
                (view["self_agent"]["x"], view["self_agent"]["y"]))
        print_map_state(defender, opp_pos)

        if defender.round > 100:
            assert False

        # apply actions for agents:
        attacker_actions = {_id: "STAY" for _id in attacker_obs.keys()}
        defender_actions = defender.step()
        print(defender_actions, round(time.time() - step_start, 3))
        print()

        game.apply_actions(attacker_actions=attacker_actions,
                           defender_actions=defender_actions)
        step += 1
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
