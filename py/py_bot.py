from copy import deepcopy
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
    return "NO"


def attack(agent, enemies, powerup_clock, explore_paths, explore_paths_template) -> str:
    # record powerups
    if "passwall" in agent.get_self_agent().powerups:
        passwall = agent.get_self_agent().powerups["passwall"]
    else:
        passwall = 0

    current_pos = (
        agent.get_self_agent().get_pos()["x"],
        agent.get_self_agent().get_pos()["y"],
    )
    agent_id = agent.get_self_agent().id

    if len(enemies) == 0:
        explore_path = explore_paths[agent_id]
        next_point = explore_path.pop(0)
        if len(explore_path) == 0:
            explore_paths[agent_id] = deepcopy(explore_paths_template[agent_id])

        next_move = get_direction(current_pos, next_point)
        if next_move == "NO":
            explore_path.insert(0, next_point)
            return rust_perf.get_direction(current_pos, next_point, [])
        else:
            return next_move

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
    input_eaten_set,
    powerup_clock,
    other_target,
    attacker_locations,
) -> str:
    agent_id = agent.get_self_agent().id
    current_pos = (
        agent.get_self_agent().get_pos()["x"],
        agent.get_self_agent().get_pos()["y"],
    )

    # powerup within view
    for powerup in agent.get_powerups():
        if powerup["powerup"] == str(Powerup.SWORD):
            input_eaten_set.add((powerup["x"], powerup["y"]))
        elif (powerup["x"], powerup["y"]) in input_eaten_set:
            input_eaten_set.remove((powerup["x"], powerup["y"]))

    # record locations have been arrived
    if current_pos in global_coin_set:
        input_eaten_set.add(current_pos)
    if current_pos in global_powerup_set:
        input_eaten_set.add(current_pos)
        powerup_clock[current_pos] = 1

    # each agent has its own target coin
    eaten_set = deepcopy(input_eaten_set)
    rest_coin_count = len([p for p in global_coin_set if p not in input_eaten_set])
    if rest_coin_count > 4:
        for _, p in other_target.items():
            eaten_set.add(p)
    # owned powerup
    passwall = agent.get_self_agent().powerups.get("passwall", 0)
    shield = agent.get_self_agent().powerups.get("shield", 0)
    invisibility = agent.get_self_agent().powerups.get("invisibility", 0)

    cancel_key = []
    for powerup, clock in powerup_clock.items():
        if clock == 12:
            if powerup in eaten_set:
                eaten_set.remove(powerup)
            cancel_key.append(powerup)
    for k in cancel_key:
        del powerup_clock[k]

    nearest_enemy_dist = 1e8
    total_dist = 0
    for ep in attacker_locations:
        dist = rust_perf.shortest_path(current_pos, ep)
        total_dist += dist
        if dist < nearest_enemy_dist:
            nearest_enemy_dist = dist

    other_agent_list = agent.get_other_agents()
    # attacker_location = set()
    has_sword = False
    for other_agent in other_agent_list:
        if other_agent.get_role() != "DEFENDER":
            # attacker_location.add((other_agent.x, other_agent.y))
            if "sword" in other_agent["powerups"]:
                has_sword = True

    # strategy one (corner)
    if (
        len(attacker_locations) > 1
        and shield < 2
        and invisibility < 2
        and total_dist < 15
        and nearest_enemy_dist <= 3
    ) or has_sword:
        next_move = rust_perf.check_stay_or_not(
            current_pos, list(attacker_locations), passwall, eaten_set
        )
        return next_move

    elif nearest_enemy_dist > 2:
        # safe
        attacker_locations = set()

    target_coin, path, _ = rust_perf.collect_coins_using_powerup(
        current_pos, eaten_set, passwall, attacker_locations
    )
    other_target[agent_id] = target_coin
    for powerup, _ in powerup_clock.items():
        powerup_clock[powerup] += 1
    return get_direction(current_pos, path[0])


class RealGame(marathon.Game):
    def __init__(self, match_id):
        super().__init__(match_id=match_id)
        self.step = 0
        self.eaten_set = set()
        self.powerup_clock = {}
        self.defender_scatter = {4: (0, 12), 5: (18, 17), 6: (11, 11), 7: (20, 9)}

        self.explore_paths_template = {
            0: [(i, 0) for i in range(21, 0, -1)] + [(1, i) for i in range(1, 23)],
            1: [(i, 11) for i in range(23, 0, -1)] + [(i, 12) for i in range(1, 23)],
            2: [(12, i) for i in range(1, 23)] + [(11, i) for i in range(23, 0, -1)],
            3: [(i, 22) for i in range(23, 0, -1)] + [(i, 23) for i in range(1, 23)],
        }

        self.explore_paths = deepcopy(self.explore_paths_template)

    def on_game_start(self, data):
        self.step = 0
        self.eaten_set = set()
        self.powerup_clock = {}

    def on_game_state(self, data: marathon.MessageGameState):
        self.step += 1

        # TODO: powerup
        # prepare state
        attacker_locations = set()
        defender_locations = set()
        for _, agent_state in data.get_states().items():
            other_agent_list = agent_state.get_other_agents()
            for other_agent in other_agent_list:
                if other_agent.get_role() == "ATTACKER":
                    attacker_locations.add(
                        (other_agent.get_pos()["x"], other_agent.get_pos()["y"])
                    )
                else:
                    defender_locations.add(
                        (other_agent.get_pos()["x"], other_agent.get_pos()["y"])
                    )

        other_target = {}
        action = {}
        for agent_id, agent_state in data.get_states().items():
            if agent_state.get_role() == "DEFENDER":
                action[agent_id] = defend(
                    agent_state,
                    self.eaten_set,
                    self.powerup_clock,
                    other_target,
                    attacker_locations,
                )
            else:
                action[agent_id] = attack(
                    agent_state,
                    list(defender_locations),
                    self.powerup_clock,
                    self.explore_paths,
                    self.explore_paths_template,
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
