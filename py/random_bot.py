import json
import argparse
import random
import marathon
import rust_perf
from typing import List

ROLES = ["DEFENDER", "ATTACKER"]
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

with open("map.json") as f:
    global_map = json.load(f)["map"]

global_coin_set = set()
global_powerup_set = set()
for cell in global_map:
    x = cell["x"]
    y = cell["y"]
    cell_type = cell["type"]
    if cell_type == "COIN":
        global_coin_set.add((x, y))
    if cell_type == "POWERUP":
        global_powerup_set.add((x, y))
eaten_set = {}


def get_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_direction(curr, next):
    if curr[1] == next[1]:
        if next[0] == curr[0]:
            return "STAY"
        elif next[0] == curr[0] + 1:
            return "RIGHT"
        elif next[0] == curr[0] - 1:
            return "LEFT"
        else:
            return "ERROR"
    elif curr[0] == next[0]:
        if next[1] == curr[1] + 1:
            return "DOWN"
        elif next[1] == curr[1] - 1:
            return "UP"
        else:
            return "ERROR"
    else:
        return "ERROR"


def handle_agent(agent: marathon.Agent, eatern_set, previous_loc) -> str:
    current_pos = (agent.get_pos["x"], agent.get_pos["y"])

    agent_id = agent.id
    # print(agent)
    # raise Exception("e")
    if "passwall" in agent.self_agent["powerups"]:
        passwall = agent["self_agent"]["powerups"]["passwall"]
    else:
        passwall = 0
    if current_pos in global_coin_set:
        eaten_set.add(current_pos)
    if current_pos in global_powerup_set:
        eaten_set.add(current_pos)

    other_agent_list: List[marathon.AgentDef] = agent.get_other_agents()
    attacker_location = []
    for other_agent in other_agent_list:
        if other_agent["role"] == "ATTACKER":
            attacker_location.append((other_agent["x"], other_agent["y"]))

    if len(previous_loc) == 0:
        previous_loc = {agent_id: (-1, -1)}

    if len(attacker_location) == 2:
        next_move = rust_perf.check_two_enemies_move(current_pos, attacker_location)
        if next_move != "NO":
            return next_move

    path, _ = rust_perf.collect_coins_using_powerup(
        previous_loc[agent_id],
        current_pos,
        eaten_set,
        attacker_location,
        passwall,
    )
    next_move = get_direction(current_pos, path[0])
    return next_move


class RealGame(marathon.Game):
    def __init__(self, match_id):
        super().__init__(match_id=match_id)
        self.previous_loc = {}

    def on_game_start(self, data):
        pass

    def on_game_state(self, data: marathon.MessageGameState):
        action = {}
        for k, v in data.get_states().items():
            action[k] = random.choice(ACTIONS)
        return action

    def on_game_end(self, data):
        pass
        # print(data)

    def on_game_over(self, data):
        pass
        # print(data)


if __name__ == "__main__":
    ps = argparse.ArgumentParser()
    ps.add_argument("-room", type=str, required=True)
    args = ps.parse_args()
    room = args.room
    g = RealGame(match_id=room)
    # 给一个同步的run
    g.sync_run()
