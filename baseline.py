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
from copy import deepcopy
from itertools import product
import json
import time

import numpy as np
from scipy.spatial import ConvexHull
from game import Game, Powerup
import random

import rust_perf

# print(rust_perf.get_direction_path((1, 1), (23, 22), []))
# raise Exception("e")

ACTIONS = ["STAY", "LEFT", "RIGHT", "DOWN", "UP"]

with open("map.json") as f:
    global_map = json.load(f)["map"]

global_coin_set = set()
global_powerup_set = set()
global_walls_list = []
global_portal_map = {}
ban_idx = set()
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
    if cell_type == "PORTAL":
        global_portal_map[(x, y)] = (cell["pair"]["x"], cell["pair"]["y"])
    if cell_type == "WALL":
        global_walls_list.append((x, y))


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


directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]


# Given position [i][j] find wall neighbors.
# Used in constructing "wall islands" for the hider
def neighbors(i, j, maze):
    n = []
    rows = len(maze)
    cols = len(maze[0])
    for dx, dy in directions:
        px = j + dx
        py = i + dy
        if px >= 0 and px < cols and py >= 0 and py < rows and maze[py][px] == "WALL":
            n.append((py, px))
    return n


# Explore graph function, for use in DFS
def explore(i, j, visited, maze):
    if (i, j) in visited:
        return None

    islands = [(i, j)]
    stack = [(i, j)]
    visited.add((i, j))

    while len(stack) > 0:
        cell = stack.pop()
        (cell_i, cell_j) = cell
        for neighbor in neighbors(cell_i, cell_j, maze):
            if neighbor not in visited:
                stack.append(neighbor)
                visited.add(neighbor)
                islands.append(neighbor)

    return islands


# Island class for keeping track of walled islands
class island_class:
    def __init__(self, points):
        self.points = points
        self.vertices = []
        self.volume = 0.0

    def set_vertices(self, v):
        self.vertices = v

    def set_volume(self, v):
        self.volume = v

    def get_volume(self):
        return self.volume

    def get_vertices(self):
        return self.vertices

    def __len__(self):
        return len(self.points)

    def __getitem__(self, position):
        return self.points[position]

    # So that we can hash this for use in dictionary
    def __hash__(self):
        return hash(self.points[0])  # Just use hash of point[0] tupple


# Retrieve all islands (connected wall components)  of maze
def all_islands(maze):
    components = []
    visited = set()
    rows = len(maze)
    cols = len(maze[0])

    # Loop through entire maze and attempt to run explore on each wall
    # Already-visited cells will be handled by explore()
    for i in range(rows):
        for j in range(cols):
            if maze[i][j] == "WALL":
                result = explore(i, j, visited, maze)
                if result != None:
                    components.append(island_class(result))

    # Filter out islands with size less than five
    # Probably could go as low 4 or 3 though
    valid = []
    for island in components:
        if len(island) >= 5:
            bad = False
            for py, px in island:
                if px == cols - 1 or py == rows - 1 or px == 0 or py == 0:
                    bad = True
                    break
            if not bad:
                valid.append(island)

    return valid


# Helper function to convert list of tupple positions (i,j) to numpy array
def tupple_to_np(tups):
    result = []
    for i, j in tups:
        result.append([i, j])

    return np.array(result)


# Create an outline of the every single island by adding adjacent points of all
#   walls. This gives us points to use for the convex hull.
# Note: This function will give you points even inside the island, but the hull
#   thankfully handles those and gives us the outer-most outline.
def outline(maze):
    outlines = []
    point_to_island = {}

    for island in all_islands(maze):
        marked = set()
        for i, j in island:
            for dx, dy in directions:
                op = (dy + i, dx + j)
                if op not in marked and maze[op[0]][op[1]] != "WALL":
                    marked.add(op)
                    point_to_island[op] = island
        outlines.append(list(marked))

    return (outlines, point_to_island)


# Will return all the islands with their vertices, and outline points
def hull(maze):
    rows = len(maze)
    cols = len(maze[0])
    (outlines, point_to_island) = outline(maze)
    corner_to_island = {}
    islands = []

    for out in outlines:
        points = tupple_to_np(out)
        h = ConvexHull(points)
        v = []
        for corner_index in h.vertices:
            corner = out[corner_index]
            i = corner[0]
            j = corner[1]
            v.append((i, j))

        island = point_to_island[v[0]]
        island.set_vertices(v)
        island.set_volume(h.volume)
        islands.append(island)

    return islands


# Find closest island from position (i,j) and choices of island_classes
# Admittedly, this is the closest island in terms of the convex hull vertices,
#   which means it not necessarily the closest. Also calculated with Euclidean
#   distance instead of steps, so even more margin for inacurracy.
def closest_island(i, j, choices):
    closest = None
    recommended_point = None
    distance = 1 << 15
    #    if random.randint(1,3) == 1:
    for island in choices:
        for corner in island.get_vertices():
            (ci, cj) = corner
            d = (i - ci) ** 2 + (j - cj) ** 2
            if d < distance:
                closest = island
                distance = d
                recommended_point = (ci, cj)
    return (closest, recommended_point, distance)


def longest_island(i, j, choices):
    closest = None
    recommended_point = None
    distance = 0
    #    if random.randint(1,3) == 1:
    for island in choices:
        for corner in island.get_vertices():
            (ci, cj) = corner
            d = (i - ci) ** 2 + (j - cj) ** 2
            if d > distance:
                closest = island
                distance = d
                recommended_point = (ci, cj)
    return (closest, recommended_point, distance)


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
# dist_map = defaultdict(dict)
# with open("dist.csv", "r", encoding="utf-8") as f:
#     for line in f.readlines():
#         fields = line.split("$")
#         dist_map[eval(fields[0])][eval(fields[1])] = int(fields[2])


# def get_distance(pos1, pos2):
#     if pos1 in dist_map:
#         if pos2 in dist_map[pos1]:
#             return dist_map[pos1][pos2]
#     return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


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

explore_paths_template = {
    0: [(i, 0) for i in range(21, 0, -1)] + [(1, i) for i in range(1, 23)],
    1: [(i, 11) for i in range(23, 0, -1)] + [(i, 12) for i in range(1, 23)],
    2: [(12, i) for i in range(1, 23)] + [(11, i) for i in range(23, 0, -1)],
    3: [(i, 22) for i in range(23, 0, -1)] + [(i, 23) for i in range(1, 23)],
}

explore_paths = deepcopy(explore_paths_template)


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


def use_attacker(agent, enemies, powerup_clock, main_chase, defender_next_move) -> str:
    # record powerups
    passwall = agent["self_agent"]["powerups"].get("passwall", 0)

    current_pos = (agent["self_agent"]["x"], agent["self_agent"]["y"])

    if agent["self_agent"]["id"] == 0:
        if current_pos in global_coin_set:
            return "STAY"

    # explore phrase
    if len(enemies) == 0:
        explore_path = explore_paths[agent["self_agent"]["id"]]
        next_point = explore_path.pop(0)
        if len(explore_path) == 0:
            explore_paths[agent["self_agent"]["id"]] = deepcopy(
                explore_paths_template[agent["self_agent"]["id"]]
            )

        next_move = get_direction(current_pos, next_point)
        if next_move == "NO":
            explore_path.insert(0, next_point)
            next_move = rust_perf.get_direction(current_pos, next_point, [])
            return next_move
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

    # print("haha")
    enemies.sort(key=lambda x: rust_perf.shortest_path(current_pos, x))
    # print("damn")
    if (
        main_chase.get(agent["self_agent"]["id"], (-1, -1)) == enemies[0]
        or rust_perf.shortest_path(current_pos, enemies[0]) <= 4
    ):
        move_surround = False
    else:
        move_surround = True

    # print("before")
    path = rust_perf.catch_enemies_using_powerup(
        current_pos,
        passwall,
        enemies,
        defender_next_move,
        move_surround,
    )
    # print("after")
    if len(path) == 0:
        return random.choice(ACTIONS)

    next_move = get_direction(current_pos, path[0])
    for powerup, _ in powerup_clock.items():
        powerup_clock[powerup] += 1
    return next_move


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


def use_defender(
    agent,
    input_eaten_set,
    powerup_clock,
    other_target,
    attacker_locations,
    coin_cache,
) -> str:
    # safe phrase
    agent_id = agent["self_agent"]["id"]
    current_pos = (agent["self_agent"]["x"], agent["self_agent"]["y"])

    for p in agent["powerups"]:
        if p["powerup"] == str(Powerup.SWORD):
            input_eaten_set.add((p["x"], p["y"]))
        elif (p["x"], p["y"]) in input_eaten_set:
            input_eaten_set.add((p["x"], p["y"]))

    # record locations have been arrived
    if current_pos in global_coin_set:
        coin_cache.add(current_pos)
        input_eaten_set.add(current_pos)
    if current_pos in global_powerup_set:
        input_eaten_set.add(current_pos)
        powerup_clock[current_pos] = 1

    # each agent has its own target coin
    eaten_set = deepcopy(input_eaten_set)
    other_group_set = set([p for _, pl in other_target.items() for p in pl[:3]])
    rest_coin_count = len(
        [p for p in global_coin_set if p not in eaten_set.union(other_group_set)]
    )
    if rest_coin_count > 0:
        eaten_set = eaten_set.union(other_group_set)

    # record powerups
    bkb = False
    passwall = agent["self_agent"]["powerups"].get("passwall", 0)
    shield = agent["self_agent"]["powerups"].get("shield", 0)
    invisibility = agent["self_agent"]["powerups"].get("invisibility", 0)
    if shield == 0:
        if agent["self_agent"]["invulnerability_duration"] > 0:
            shield = agent["self_agent"]["invulnerability_duration"]
            bkb = True

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
    has_sword = False
    nearest_enemy_dist = 1e8
    total_dist = 0
    for _, ep in attacker_locations.items():
        dist = rust_perf.shortest_path(current_pos, ep)
        total_dist += dist
        if dist < nearest_enemy_dist:
            nearest_enemy_dist = dist

    enemy_nearby_count = 0
    enemies_in_vision = set()
    for other_agent in other_agent_list:
        if other_agent["role"] != "DEFENDER":
            enemy_nearby_count += 1
            enemies_in_vision.add((other_agent["x"], other_agent["y"]))
            if "sword" in other_agent["powerups"]:
                has_sword = True

    if has_sword:
        shield = 0

    attacker_list = [v for v in attacker_locations.values()]

    for powerup, _ in powerup_clock.items():
        powerup_clock[powerup] += 1

    if len(coin_cache) == 87 and not bkb:
        if len(enemies_in_vision) >= 1:
            path = rust_perf.check_stay_or_not(
                current_pos, attacker_list, passwall, eaten_set
            )
            return get_direction(current_pos, path[0])

    # one function
    # print("----")
    # print("shield", shield)
    # print("sword", has_sword)
    # print("passwall", passwall)
    # print("eaten", len(eaten_set))
    # print(set(attacker_list))
    # print(current_pos, attacker_locations)
    # print(rest_coin_count)
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
        6,
    )
    other_target[agent_id] = target_coin_group
    if len(path) == 0:
        path = rust_perf.check_stay_or_not(
            current_pos, attacker_list, passwall, eaten_set
        )
        print("shield", shield)
        print("sword", has_sword)
        print("passwall", passwall)
        print("invisibility", invisibility)
        print("eaten", len(eaten_set))
        print(set(attacker_list))
        print(current_pos, attacker_locations)
        print(target_coin_group)
        print(rest_coin_count)
        # raise Exception("e")
        if len(path) == 0:
            return random.choice(ACTIONS)
    return get_direction(current_pos, path[0])

    # one enemy could not catch me
    if enemy_nearby_count < 2:
        target_coin_group, path, _ = rust_perf.collect_coins_using_powerup(
            agent_id,
            current_pos,
            eaten_set,
            passwall,
            shield,
            set(attacker_list),
            openness_map,
            7,
        )
        other_target[agent_id] = target_coin_group
        if len(path) == 0:
            if nearest_enemy_dist <= 2:
                path = rust_perf.check_stay_or_not(
                    current_pos, attacker_list, passwall, eaten_set
                )
                # print(attacker_list)
                # print(current_pos, path)
                return get_direction(current_pos, path[0])
                # print(current_pos, attacker_list)
                # print(target_coin_group)
                # print(rest_coin_count)
                # raise Exception("e")
            # print(current_pos, attacker_list)
            # raise Exception("e")
            return random.choice(ACTIONS)
        return get_direction(current_pos, path[0])

    else:
        # run away strategy
        path = rust_perf.check_stay_or_not(
            current_pos, attacker_list, passwall, eaten_set
        )
        # if agent_id == 4:
        #     print("-----")
        #     print("id: ", agent_id)
        #     print("pos", current_pos)
        #     print(attacker_locations)
        #     print(has_sword, shield)
        #     print("dist:", nearest_enemy_dist)
        #     print("score:", agent["self_agent"]["score"])
        #     print(path)

        if nearest_enemy_dist == 1:
            print("**********")
            print("id: ", agent_id)
            print(current_pos)
            print(attacker_locations)
            print(has_sword, shield)
            print("dist:", total_dist)
            print("score:", agent["self_agent"]["score"])
            print(path)
        return get_direction(current_pos, path[0])


if __name__ == "__main__":
    # load map
    with open("map.json") as f:
        map = json.load(f)
    game = Game(map)

    # init game
    win_count = 0
    attacker_score = 0
    defender_score = 0
    seeds = [random.randint(0, 1000000) for _ in range(2)]
    # seeds = [872914]
    for seed in seeds:
        print("=========start=======", seed)
        game.reset(attacker="attacker", defender="defender", seed=seed)

        eatten_set = set()
        coin_cache = set()
        step = 0
        defender_powerup_clock = {}
        attacker_powerup_clock = {}
        start_game_time = time.time()
        map_in_heart = [[0.0 for _ in range(24)] for _ in range(24)]
        for x, y in global_walls_list:
            map_in_heart[x][y] = -1.0
        predicted_attacker_pos = {0: (22, 0), 1: (22, 1), 2: (23, 0), 3: (23, 1)}

        # 16 islands
        islands = hull(maze)
        print(len(islands))

        prev_score = -1
        # game loop
        while not game.is_over():
            # get game state for player:
            attacker_state = game.get_agent_states_by_player("attacker")
            defender_state = game.get_agent_states_by_player("defender")

            # for k, v in defender_state.items():
            #     print("defender", k, v["self_agent"]["score"])

            # print(
            #     "a",
            #     [
            #         (k, v["self_agent"]["x"], v["self_agent"]["y"])
            #         for k, v in attacker_state.items()
            #     ],
            #     [v["self_agent"]["score"] for _, v in attacker_state.items()],
            # )
            # print(
            #     "d",
            #     [
            #         (k, v["self_agent"]["x"], v["self_agent"]["y"])
            #         for k, v in defender_state.items()
            #     ],
            #     [v["self_agent"]["score"] for _, v in defender_state.items()],
            # )
            # print("p", predicted_attacker_pos)

            defender_locations = set()
            my_pos = []
            main_chase = {}
            for k, v in attacker_state.items():
                my_pos.append((v["self_agent"]["x"], v["self_agent"]["y"]))
                other_agent_list = v["other_agents"]
                for other_agent in other_agent_list:
                    if other_agent["role"] == "DEFENDER":
                        defender_locations.add((other_agent["x"], other_agent["y"]))
                        main_chase[int(k)] = (other_agent["x"], other_agent["y"])

            defender_next_move = {}
            for enemy in defender_locations:
                defender_next_move[enemy] = predict_enemy_move(enemy, my_pos)
            # attacker_actions = {
            #     _id: random.choice(ACTIONS) for _id in attacker_state.keys()
            # }
            # attacker_actions = {_id: "STAY" for _id in attacker_state.keys()}
            attacker_actions = {
                _id: use_attacker(
                    attacker_state[_id],
                    list(defender_locations),
                    attacker_powerup_clock,
                    main_chase,
                    defender_next_move,
                )
                for _id in attacker_state.keys()
            }

            # defender_actions = {
            #     _id: random.choice(ACTIONS) for _id in defender_state.keys()
            # }

            my_pos = {}
            attacker_locations = dict()
            # least_score_id = 0
            # least_score = 1e8
            current_score = 0
            for k, v in defender_state.items():
                my_pos[int(k)] = (v["self_agent"]["x"], v["self_agent"]["y"])
                other_agent_list = v["other_agents"]
                current_score += v["self_agent"]["score"]
                # if v["self_agent"]["score"] < least_score:
                #     least_score = v["self_agent"]["score"]
                #     least_score_id = int(k)

                for other_agent in other_agent_list:
                    if other_agent["role"] == "ATTACKER":
                        attacker_locations[int(other_agent["id"])] = (
                            other_agent["x"],
                            other_agent["y"],
                        )
                        # predicted_attacker_pos[int(other_agent["id"])] = (
                        #     other_agent["x"],
                        #     other_agent["y"],
                        # )
            if current_score != prev_score:
                prev_score = current_score
                print("current_score: ", current_score)

            # all_pos = get_all_nearby_pos(predicted_attacker_pos, map_in_heart)
            # print(all_pos)
            # print(my_pos)
            # start = time.time()
            # if len(all_pos) > 100:
            #     all_pos = random.sample(all_pos, 100)
            # score_vec = rust_perf.predict_enemy([0, 1, 2, 3], all_pos, my_pos, {}, 0)
            # print(time.time() - start)
            # print(score_vec)
            # predicted_attacker_pos = all_pos[score_vec.index(max(score_vec))]
            for k, v in attacker_locations.items():
                predicted_attacker_pos[k] = v
            # print(predicted_attacker_pos)
            # raise Exception("e")

            # print(f"a, {step}/1152")
            other_target = {i: [] for i in range(4, 8)}
            defender_actions = {
                _id: use_defender(
                    defender_state[_id],
                    eatten_set,
                    defender_powerup_clock,
                    other_target,
                    attacker_locations,
                    coin_cache,
                    # predicted_attacker_pos,
                )
                for _id in defender_state.keys()
            }

            game.apply_actions(
                attacker_actions=attacker_actions, defender_actions=defender_actions
            )
            step += 1
            # print(f"d,{step}/1152")

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

        defender_state = game.get_agent_states_by_player("attacker")
        for k, v in defender_state.items():
            print("attacker", k, v["self_agent"]["score"])
        defender_state = game.get_agent_states_by_player("defender")
        for k, v in defender_state.items():
            print("defender", k, v["self_agent"]["score"])

    print("Win rate is ", win_count / len(seeds))
    print(f"Attacker score: {attacker_score} vs Defender score: {defender_score}")
