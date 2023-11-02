from copy import deepcopy
from itertools import product
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
global_walls_list = []
maze = [[0 for _ in range(24)] for _ in range(24)]
for cell in global_map:
    x = cell["x"]
    y = cell["y"]
    cell_type = cell["type"]
    maze[x][y] = cell_type

    if cell_type == "COIN":
        global_coin_set.add((x, y))
    if cell_type == "POWERUP":
        global_powerup_set.add((x, y))
    if cell_type == "WALL":
        global_walls_list.append((x, y))


def openness(x, y, grid):
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    count = 0
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] != "WALL":
            count += 1
    return count


openness_map = dict()
for i in range(24):
    for j in range(24):
        openness_map[(i, j)] = openness(i, j, maze)


def get_neighbors(pos):
    x, y = pos
    possible_moves = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    return [
        move
        for move in possible_moves
        if 0 <= move[0] < 24 and 0 <= move[1] < 24 and move not in global_walls_list
    ]


def compute_threat_for_position(pos, agents):
    # Calculate a threat score based on distance to all agents.
    # The closer an agent is, the higher the threat.
    return sum(
        1 / ((pos[0] - agent[0]) ** 2 + (pos[1] - agent[1]) ** 2 + 1e-6)
        for agent in agents
    )


def predict_enemy_move(enemy_pos, agents):
    neighbors = get_neighbors(enemy_pos)
    threats = {
        neighbor: compute_threat_for_position(neighbor, agents)
        for neighbor in neighbors
    }

    # Return the direction with the least threat
    move_to = min(threats, key=threats.get)
    delta_x = move_to[0] - enemy_pos[0]
    delta_y = move_to[1] - enemy_pos[1]
    return (delta_x, delta_y)


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


def get_all_nearby_pos(agent_pos: dict, map_in_heart: list):
    """
    agent_pos: {id1: (x1, y1), id2: (x2, y2)}
    """
    delta = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    new_pos = list()
    ids = list()
    for id, ap in agent_pos.items():
        ids.append(id)
        tmp = list()
        for d in delta:
            pos_tmp = (ap[0] + d[0], ap[1] + d[1])
            if (
                not (0 <= pos_tmp[0] < 24)
                or not (0 <= pos_tmp[1] < 24)
                or map_in_heart[pos_tmp[0]][pos_tmp[1]] < 0
            ):
                continue
            tmp.append(pos_tmp)
        new_pos.append(tmp)

    comb_pos = product(*new_pos)

    next_pos = list()
    for pos in comb_pos:
        pos_dict = dict()
        for i in range(len(ids)):
            pos_dict[ids[i]] = pos[i]
        next_pos.append(pos_dict)

    return next_pos


def attack(
    agent,
    enemies,
    powerup_clock,
    explore_paths,
    explore_paths_template,
    main_chase,
    defender_next_move,
) -> str:
    # record powerups
    passwall = agent.get_self_agent().powerups.get("passwall", 0)

    current_pos = (
        agent.get_self_agent().get_pos()["x"],
        agent.get_self_agent().get_pos()["y"],
    )
    agent_id = agent.get_self_agent().id

    if agent_id == 0:
        if current_pos in global_coin_set:
            return "STAY"

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

    enemies.sort(key=lambda x: rust_perf.shortest_path(current_pos, x))

    if (
        main_chase.get(agent_id, (-1, -1)) == enemies[0]
        or rust_perf.shortest_path(current_pos, enemies[0]) <= 4
    ):
        move_surround = False
    else:
        move_surround = True

    path = rust_perf.catch_enemies_using_powerup(
        current_pos,
        passwall,
        enemies,
        defender_next_move,
        move_surround,
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
    coin_cache,
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
        coin_cache.add(current_pos)
        input_eaten_set.add(current_pos)
    if current_pos in global_powerup_set:
        input_eaten_set.add(current_pos)
        powerup_clock[current_pos] = 1

    # each agent has its own target coin
    eaten_set = deepcopy(input_eaten_set)
    rest_coin_count = len([p for p in global_coin_set if p not in input_eaten_set])
    other_group_set = set([p for _, pl in other_target.items() for p in pl[:3]])
    rest_coin_count = len(
        [p for p in global_coin_set if p not in input_eaten_set.union(other_group_set)]
    )
    if rest_coin_count > 0:
        eaten_set = eaten_set.union(other_group_set)

    # owned powerup
    bkb = False
    passwall = agent.get_self_agent().powerups.get("passwall", 0)
    shield = agent.get_self_agent().powerups.get("shield", 0)
    invisibility = agent.get_self_agent().powerups.get("invisibility", 0)
    if shield == 0:
        if agent.get_self_agent().invulnerability_duration > 0:
            shield = agent.get_self_agent().invulnerability_duration
            bkb = True

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
    for _, ep in attacker_locations.items():
        dist = rust_perf.shortest_path(current_pos, ep)
        total_dist += dist
        if dist < nearest_enemy_dist:
            nearest_enemy_dist = dist

    other_agent_list = agent.get_other_agents()
    # attacker_location = set()
    has_sword = False
    enemy_nearby_count = 0
    enemies_in_vision = set()
    for other_agent in other_agent_list:
        if other_agent.get_role() != "DEFENDER":
            enemy_nearby_count += 1
            # attacker_location.add((other_agent.x, other_agent.y))
            enemies_in_vision.add((other_agent.x, other_agent.y))
            if "sword" in other_agent["powerups"]:
                has_sword = True

    if has_sword:
        shield = 0

    attacker_list = [v for v in attacker_locations.values()]
    for powerup, _ in powerup_clock.items():
        powerup_clock[powerup] += 1

    if len(coin_cache) == 87 and not bkb:
        if len(enemies_in_vision) == 0 or nearest_enemy_dist > 5:
            return "STAY"
        else:
            path = rust_perf.check_stay_or_not(
                current_pos, list(enemies_in_vision), passwall, eaten_set
            )
            return get_direction(current_pos, path[0])

    target_coin_group, path, _ = rust_perf.collect_coins_using_powerup(
        agent_id,
        current_pos,
        eaten_set,
        passwall,
        shield,
        invisibility,
        enemies_in_vision,
        set(attacker_list),
        openness_map,
        5,
    )
    other_target[agent_id] = target_coin_group
    if len(path) == 0:
        path = rust_perf.check_stay_or_not(
            current_pos, attacker_list, passwall, eaten_set
        )
        if len(path) == 0:
            return random.choice(ACTIONS)
    return get_direction(current_pos, path[0])


class RealGame(marathon.Game):
    def __init__(self, match_id):
        super().__init__(match_id=match_id)
        self.step = 0
        self.eaten_set = set()
        self.coin_cache = set()
        self.powerup_clock = {}
        self.defender_scatter = {4: (0, 12), 5: (18, 17), 6: (11, 11), 7: (20, 9)}
        self.map_in_heart = [[0.0 for _ in range(24)] for _ in range(24)]

        for x, y in global_walls_list:
            self.map_in_heart[x][y] = -1.0
        self.predicted_attacker_pos = {0: (22, 0), 1: (22, 1), 2: (23, 0), 3: (23, 1)}

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
        self.coin_cache = set()
        self.powerup_clock = {}

    def on_game_state(self, data: marathon.MessageGameState):
        self.step += 1

        # TODO: powerup
        # prepare state
        attacker_locations = dict()
        defender_locations = set()
        my_pos = {}
        my_pos_list = []
        main_chase = {}
        for k, agent_state in data.get_states().items():
            my_pos[int(k)] = (
                agent_state.get_self_agent().get_pos()["x"],
                agent_state.get_self_agent().get_pos()["y"],
            )
            my_pos_list.append(
                (
                    agent_state.get_self_agent().get_pos()["x"],
                    agent_state.get_self_agent().get_pos()["y"],
                )
            )
            other_agent_list = agent_state.get_other_agents()
            for other_agent in other_agent_list:
                if other_agent.get_role() == "ATTACKER":
                    attacker_locations[int(other_agent.id)] = (
                        other_agent.get_pos()["x"],
                        other_agent.get_pos()["y"],
                    )
                    self.predicted_attacker_pos[int(other_agent.id)] = (
                        other_agent.get_pos()["x"],
                        other_agent.get_pos()["y"],
                    )
                else:
                    main_chase[int(k)] = (
                        other_agent.get_pos()["x"],
                        other_agent.get_pos()["y"],
                    )
                    defender_locations.add(
                        (other_agent.get_pos()["x"], other_agent.get_pos()["y"])
                    )

        # all_pos = get_all_nearby_pos(self.predicted_attacker_pos, self.map_in_heart)
        # score_vec = rust_perf.predict_enemy([0, 1, 2, 3], all_pos, my_pos, {}, 0)
        # self.predicted_attacker_pos = all_pos[score_vec.index(max(score_vec))]

        for k, v in attacker_locations.items():
            self.predicted_attacker_pos[k] = v

        defender_next_move = {}
        for enemy in defender_locations:
            defender_next_move[enemy] = predict_enemy_move(enemy, my_pos_list)

        other_target = {i: [] for i in range(4, 8)}
        action = {}
        for agent_id, agent_state in data.get_states().items():
            if agent_state.get_role() == "DEFENDER":
                action[agent_id] = defend(
                    agent_state,
                    self.eaten_set,
                    self.powerup_clock,
                    other_target,
                    # self.predicted_attacker_pos,
                    attacker_locations,
                    self.coin_cache,
                )
            else:
                action[agent_id] = attack(
                    agent_state,
                    list(defender_locations),
                    self.powerup_clock,
                    self.explore_paths,
                    self.explore_paths_template,
                    main_chase,
                    defender_next_move,
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
