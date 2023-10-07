import json
import time
from game import Game
import random
from map.map import Map
from model.defender import Defender

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
    attacker_state = game.get_agent_states_by_player("attacker")
    defender_state = game.get_agent_states_by_player("defender")
    attacker_ids = sorted(list(attacker_state.keys()))
    defender_ids = sorted(list(defender_state.keys()))

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
        # get game state for player:
        attacker_state = game.get_agent_states_by_player("attacker")
        defender_state = game.get_agent_states_by_player("defender")
        defender.map.update_map(defender_state)

        for k, v in attacker_state.items():
            print("===>", k)
            for k2, v2 in v.items():
                print(k2, v2)
        print()

        # apply actions for agents:
        attacker_actions = {
            _id: random.choice(ACTIONS)
            for _id in attacker_state.keys()
        }
        defender_actions = defender.step()

        game.apply_actions(attacker_actions=attacker_actions,
                           defender_actions=defender_actions)
        step += 1
        # print(f"{step}/1152")

        assert step < 2

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
