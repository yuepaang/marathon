import json
import argparse
import random
import marathon
import rust_perf

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


def handle_agent(
    agent: marathon.Agent, eaten_set, step, powerup_clock, attacker_location
) -> str:
    current_pos = (agent.get_pos()["x"], agent.get_pos()["y"])

    for p in agent.get_powerups():
        if p["powerup"] == "4" or p["powerup"] == 4:
            eaten_set.add((p["x"], p["y"]))

    if "passwall" in agent.get_self_agent().powerups:
        passwall = agent.get_self_agent().powerups["passwall"]
    else:
        passwall = 0

    if "shield" in agent.get_self_agent().powerups:
        shield = agent.get_self_agent().powerups["shield"]
    else:
        shield = 0

    # scatter first
    if step < 5:
        return rust_perf.get_direction(
            current_pos,
            random.choice([(0, 12), (18, 17), (11, 11)]),
            list(eaten_set),
        )

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

    other_agent_list = agent.get_other_agents()
    allies_location = []
    for other_agent in other_agent_list:
        if other_agent.get_role() == "DEFENDER":
            allies_location.append((other_agent.x, other_agent.y))

    # strategy one (corner)
    if len(attacker_location) >= 1 and shield <= 3:
        next_move = rust_perf.check_stay_or_not(
            current_pos, attacker_location, passwall
        )
        # print(current_pos, attacker_location, next_move, passwall)
        if next_move != "NO":
            return next_move

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


class RealGame(marathon.Game):
    def __init__(self, match_id):
        super().__init__(match_id=match_id)
        self.step = 0
        self.eaten_set = set()
        self.powerup_clock = {}

    def on_game_start(self, data):
        self.step = 0
        self.eaten_set = set()
        self.powerup_clock = {}

    def on_game_state(self, data: marathon.MessageGameState):
        self.step += 1
        action = {}
        attacker_location = []
        for _, v in data.get_states().items():
            other_agent_list = v.get_other_agents()
            for other_agent in other_agent_list:
                if other_agent.get_role() == "ATTACKER":
                    attacker_location.append(
                        (other_agent.get_pos()["x"], other_agent.get_pos()["y"])
                    )

        for k, v in data.get_states().items():
            if v.get_role() == "DEFENDER":
                action[k] = handle_agent(
                    v, self.eaten_set, self.step, self.powerup_clock, attacker_location
                )
            else:
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
