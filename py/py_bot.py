import json
import argparse
import random
import marathon
import rust_perf

from game import Powerup

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


def attack(agent, enemies, powerup_clock) -> str:
    # record powerups
    if "passwall" in agent.get_self_agent().powerups:
        passwall = agent.get_self_agent().powerups["passwall"]
    else:
        passwall = 0

    current_pos = (agent.get_pos()["x"], agent.get_pos()["y"])

    # record locations have been arrived
    if current_pos in global_powerup_set:
        powerup_clock[current_pos] = 1

    cancel_key = []
    for powerup, clock in powerup_clock.items():
        if clock == 12:
            cancel_key.append(powerup)
    for k in cancel_key:
        del powerup_clock[k]

    path = rust_perf.catch_enemies_using_powerup(
        current_pos,
        passwall,
        enemies,
    )
    if len(path) == 0:
        return random.choice(ACTIONS)

    next_move = get_direction(current_pos, path[0])
    for powerup, _ in powerup_clock.items():
        powerup_clock[powerup] += 1
    return next_move


def defend(
    agent: marathon.Agent,
    eaten_set,
    step,
    powerup_clock,
    attacker_location,
    defender_scatter,
) -> str:
    current_pos = (agent.get_pos()["x"], agent.get_pos()["y"])

    # powerup within view
    for powerup in agent.get_powerups():
        if powerup["powerup"] == str(Powerup.SWORD):
            eaten_set.add((powerup["x"], powerup["y"]))
        elif (powerup["x"], powerup["y"]) in eaten_set:
            eaten_set.remove((powerup["x"], powerup["y"]))

    # owned powerup
    if "passwall" in agent.get_self_agent().powerups:
        passwall = agent.get_self_agent().powerups["passwall"]
    else:
        passwall = 0

    if "shield" in agent.get_self_agent().powerups:
        shield = agent.get_self_agent().powerups["shield"]
    else:
        shield = 0

    # scatter first
    if step < 7:
        if agent.get_self_agent().id not in defender_scatter:
            return rust_perf.get_direction(
                current_pos,
                random.choice([(0, 12), (18, 17), (11, 11)]),
                list(eaten_set),
            )
        return rust_perf.get_direction(
            current_pos,
            defender_scatter[agent.get_self_agent().id],
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

    other_agent_list = agent.get_other_agents()
    allies_location = []
    has_sword = False
    for other_agent in other_agent_list:
        if other_agent.get_role() == "DEFENDER":
            allies_location.append((other_agent.x, other_agent.y))
        else:
            if "sword" in other_agent["powerups"]:
                has_sword = True

    # strategy one (corner)
    if (len(attacker_location) >= 1 and shield <= 3) or has_sword:
        next_move = rust_perf.check_stay_or_not(
            current_pos, attacker_location, passwall
        )
        return next_move

    if agent.get_self_agent().id == 4:
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


class RealGame(marathon.Game):
    def __init__(self, match_id):
        super().__init__(match_id=match_id)
        self.step = 0
        self.eaten_set = set()
        self.powerup_clock = {}
        self.defender_scatter = {4: (0, 12), 5: (18, 17), 6: (11, 11), 7: (20, 9)}

    def on_game_start(self, data):
        self.step = 0
        self.eaten_set = set()
        self.powerup_clock = {}

    def on_game_state(self, data: marathon.MessageGameState):
        self.step += 1
        action = {}

        # TODO: powerup
        # prepare state
        attacker_locations = []
        defender_locations = []
        for _, agent_state in data.get_states().items():
            other_agent_list = agent_state.get_other_agents()
            for other_agent in other_agent_list:
                if other_agent.get_role() == "ATTACKER":
                    attacker_locations.append(
                        (other_agent.get_pos()["x"], other_agent.get_pos()["y"])
                    )
                elif other_agent.invulnerability_duration == 0:
                    defender_locations.append(
                        (other_agent.get_pos()["x"], other_agent.get_pos()["y"])
                    )

        for agent_id, agent_state in data.get_states().items():
            if agent_state.get_role() == "DEFENDER":
                action[agent_id] = defend(
                    agent_state,
                    self.eaten_set,
                    self.step,
                    self.powerup_clock,
                    attacker_locations,
                    self.defender_scatter,
                )
            else:
                action[agent_id] = attack(
                    agent_state, defender_locations, self.powerup_clock
                )
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
