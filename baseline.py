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
import json
import time

import numpy as np
from scipy.spatial import ConvexHull
from game import Game, Powerup
import random

import rust_perf

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


# 16 islands
islands = hull(maze)
print(len(islands))

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
dist_map = defaultdict(dict)
with open("dist.csv", "r", encoding="utf-8") as f:
    for line in f.readlines():
        fields = line.split("$")
        dist_map[eval(fields[0])][eval(fields[1])] = int(fields[2])


def get_distance(pos1, pos2):
    if pos1 in dist_map:
        if pos2 in dist_map[pos1]:
            return dist_map[pos1][pos2]
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


explore_paths_template = {
    0: [(i, 0) for i in range(21, 0, -1)] + [(1, i) for i in range(1, 23)],
    1: [(i, 11) for i in range(23, 0, -1)] + [(i, 12) for i in range(1, 23)],
    2: [(12, i) for i in range(1, 23)] + [(11, i) for i in range(23, 0, -1)],
    3: [(i, 22) for i in range(23, 0, -1)] + [(i, 23) for i in range(1, 23)],
}

explore_paths = deepcopy(explore_paths_template)


def use_attacker(agent, enemies, powerup_clock) -> str:
    # record powerups
    if "passwall" in agent["self_agent"]["powerups"]:
        passwall = agent["self_agent"]["powerups"]["passwall"]
    else:
        passwall = 0
    current_pos = (agent["self_agent"]["x"], agent["self_agent"]["y"])

    # if agent["self_agent"]["id"] == 2:
    #     print("^^^^", current_pos)

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
        agent["self_agent"]["id"],
        current_pos,
        passwall,
        enemies,
    )
    if len(path) == 0:
        print(
            agent["self_agent"]["id"],
            current_pos,
            agent["self_agent"]["score"],
            passwall,
            enemies,
        )
        return random.choice(ACTIONS)

    # print(
    #     path,
    #     agent["self_agent"]["id"],
    #     current_pos,
    #     agent["self_agent"]["score"],
    #     passwall,
    #     enemies,
    # )
    next_move = get_direction(current_pos, path[0])
    for powerup, _ in powerup_clock.items():
        powerup_clock[powerup] += 1
    return next_move


def use_defender(
    agent, eaten_set, step, powerup_clock, attacker_location, defender_scatter
) -> str:
    # safe phrase
    agent_id = agent["self_agent"]["id"]

    for p in agent["powerups"]:
        if p["powerup"] == str(Powerup.SWORD):
            eaten_set.add((p["x"], p["y"]))
        elif (p["x"], p["y"]) in eaten_set:
            eaten_set.remove((p["x"], p["y"]))

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
    if step < 8:
        if agent_id not in defender_scatter:
            return rust_perf.get_direction(
                current_pos,
                random.choice([(0, 12), (18, 17), (11, 11)]),
                list(eaten_set),
            )
        return rust_perf.get_direction(
            current_pos,
            defender_scatter[agent_id],
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

    # enemies in out vision
    other_agent_list = agent["other_agents"]
    allies_location = []
    has_sword = False
    for other_agent in other_agent_list:
        if other_agent["role"] == "DEFENDER":
            allies_location.append((other_agent["x"], other_agent["y"]))
        else:
            if "sword" in other_agent["powerups"]:
                has_sword = True

    # strategy one (corner)
    if (len(attacker_location) >= 1 and shield < 3) or has_sword:
        next_move = rust_perf.check_stay_or_not(
            current_pos, attacker_location, passwall
        )
        # print(agent_id, current_pos, attacker_location, next_move, passwall)
        return next_move

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
    elif curr[0] == true_next[0]:
        if true_next[1] == curr[1] + 1:
            return "DOWN"
        elif true_next[1] == curr[1] - 1:
            return "UP"
    return "NO"


# load map
with open("map.json") as f:
    map = json.load(f)
game = Game(map)


# init game
win_count = 0
attacker_score = 0
defender_score = 0
seeds = [random.randint(0, 1000000) for _ in range(5)]
# seeds = [990986]
for seed in seeds:
    game.reset(attacker="attacker", defender="defender", seed=seed)

    eatten_set = set()
    step = 0
    powerup_clock = {}
    defender_scatter = {4: (0, 12), 5: (18, 17), 6: (11, 11), 7: (20, 9)}
    start_game_time = time.time()
    # game loop
    while not game.is_over():
        # get game state for player:
        attacker_state = game.get_agent_states_by_player("attacker")
        defender_state = game.get_agent_states_by_player("defender")

        attacker_locations = set()
        defender_locations = set()
        for k, v in attacker_state.items():
            other_agent_list = v["other_agents"]
            for other_agent in other_agent_list:
                if other_agent["role"] == "ATTACKER":
                    attacker_locations.add((other_agent["x"], other_agent["y"]))
                # elif other_agent["invulnerability_duration"] == 0:
                else:
                    defender_locations.add((other_agent["x"], other_agent["y"]))

        # attacker_actions = {
        #     _id: random.choice(ACTIONS) for _id in attacker_state.keys()
        # }
        attacker_actions = {
            _id: use_attacker(attacker_state[_id], list(defender_locations), {})
            for _id in attacker_state.keys()
        }

        attacker_locations = set()
        defender_locations = set()
        for k, v in defender_state.items():
            other_agent_list = v["other_agents"]
            for other_agent in other_agent_list:
                if other_agent["role"] == "ATTACKER":
                    attacker_locations.add((other_agent["x"], other_agent["y"]))
                elif other_agent["invulnerability_duration"] == 0:
                    defender_locations.add((other_agent["x"], other_agent["y"]))

        defender_actions = {
            _id: random.choice(ACTIONS) for _id in defender_state.keys()
        }
        # defender_actions = {
        #     _id: use_defender(
        #         defender_state[_id],
        #         eatten_set,
        #         step,
        #         powerup_clock,
        #         list(attacker_locations),
        #         defender_scatter,
        #     )
        #     for _id in defender_state.keys()
        # }

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
