# -*- coding: utf-8 -*-
"""Kunpeng.

__author__ = Yue Peng
__copyright__ = Copyright 2022
__version__ = 1.0.0
__maintainer__ = Yue Peng
__email__ = yuepaang@gmail.com
__status__ = Dev
__filename__ = new_baseline.py
__uuid__ = 6c063ddc-9087-4f1a-9879-c09208bef679
__date__ = 2023-10-12 15:15:56
__modified__ =
"""

from collections import defaultdict
from copy import deepcopy
from enum import Enum
import json
import time
from typing import Dict, List, Union

import numpy as np
from scipy.spatial import ConvexHull
from game import Game, Powerup
import random
from numba import jit

import rust_perf

ACTIONS = ["STAY", "LEFT", "RIGHT", "DOWN", "UP"]


class DIRECTION(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


with open("map.json") as f:
    map_json = json.load(f)


class Point:
    def __init__(self, _x: int, _y: int):
        self.x = _x  # type: int
        self.y = _y  # type: int
        self.next = dict()  # type: Dict[DIRECTION, Union[None, Point]]
        self.wall = False  # type: bool
        self.portal = None  # type: Union[None, Point]
        self.name = ""

    def __repr__(self):
        ret = dict()
        ret["x"] = self.x
        ret["y"] = self.y
        return json.dumps(ret)


@jit(nopython=True)
def floyd(node_num, dist, path, inf=9999):
    for k in range(node_num):
        for i in range(node_num):
            for j in range(node_num):
                if (
                    dist[i][k] + dist[k][j] < dist[i][j]
                    and dist[i][k] != inf
                    and dist[k][j] != inf
                    and i != j
                ):
                    dist[i][j] = dist[i][k] + dist[k][j]
                    path[i][j] = path[i][k]
    return dist


class Path:
    def __init__(self, _maps: Dict[int, Dict[int, Point]], _width: int, _height: int):
        self.maps = _maps  # type: Dict[int, Dict[int, Point]]
        self.width = _width  # type: int
        self.height = _height  # type: int
        self.node_num = self.width * self.height  # type: int
        self.inf = 9999
        self.dist = [
            [self.inf for _ in range(self.node_num)] for __ in range(self.node_num)
        ]  # type: List[List[int]]
        self.path = [
            [0 for _ in range(self.node_num)] for __ in range(self.node_num)
        ]  # type: List[List[int]]
        for i in range(self.node_num):
            for j in range(self.node_num):
                self.path[i][j] = j
        for i in range(self.node_num):
            for d in self.to_point(i).next:
                j = self.to_index(self.to_point(i).next[d])
                if i != j:
                    self.dist[i][j] = 1
                else:
                    self.dist[i][j] = 0
        self.dist = floyd(
            self.node_num, np.array(self.dist), np.array(self.path)
        ).tolist()

    def get_cost(self, start: Point, end: Point):
        return self.dist[self.to_index(start)][self.to_index(end)]

    def get_cost_index(self, start: int, end: int):
        return self.dist[start][end]

    def to_point(self, index: int) -> Point:
        return self.maps[index % self.width][index // self.width]

    def to_index(self, point: Point) -> int:
        return point.x + point.y * self.width

    def floyd(self):
        for k in range(self.node_num):
            for i in range(self.node_num):
                for j in range(self.node_num):
                    if (
                        self.dist[i][k] + self.dist[k][j] < self.dist[i][j]
                        and self.dist[i][k] != self.inf
                        and self.dist[k][j] != self.inf
                        and i != j
                    ):
                        self.dist[i][j] = self.dist[i][k] + self.dist[k][j]
                        self.path[i][j] = self.path[i][k]


class MapInfo:
    def __init__(self, _map_json, calculate_path=False):
        self.map_json = _map_json
        self.width = _map_json["map_conf"]["width"]  # type: int
        self.height = _map_json["map_conf"]["height"]  # type: int
        self.vision = _map_json["map_conf"]["vision_range"]  # type: int
        self.maps = dict()  # type: Dict[int, Dict[int, Point]]
        for x in range(self.width):
            if x not in self.maps:
                self.maps[x] = dict()
            for y in range(self.height):
                self.maps[x][y] = Point(x, y)
        self.load_map()
        self.construct_map()
        if calculate_path:
            self.path = Path(self.maps, self.width, self.height)

    def set_wall(self, x: int, y: int):
        self.maps[x][y].wall = True

    def set_portal(self, x: int, y: int, next_x: int, next_y: int, name: str):
        self.maps[x][y].portal = self.maps[next_x][next_y]
        self.maps[x][y].name = name

    def move_none(self, p: Point):
        if DIRECTION.STAY in p.next:
            return
        if p.portal:
            p.next[DIRECTION.STAY] = p.portal
        else:
            p.next[DIRECTION.STAY] = p

    def move_one_step(self, p: Point, direction: DIRECTION):
        next_x = p.x
        next_y = p.y
        if direction == DIRECTION.UP:
            next_y -= 1
        elif direction == DIRECTION.DOWN:
            next_y += 1
        elif direction == DIRECTION.LEFT:
            next_x -= 1
        elif direction == DIRECTION.RIGHT:
            next_x += 1
        if (
            0 <= next_x < self.width
            and 0 <= next_y < self.height
            and not self.maps[next_x][next_y].wall
        ):
            return self.maps[next_x][next_y]
        else:
            return p

    def load_map(self):
        wormholes = dict()  # type: Dict[str, List[int]]
        for cell in self.map_json["map"]:
            if cell["type"] == "WALL":
                self.set_wall(cell["x"], cell["y"])
            elif cell["type"] == "PORTAL":
                name = cell["name"]
                if name not in wormholes:
                    wormholes[name] = [
                        cell["x"],
                        cell["y"],
                        cell["pair"]["x"],
                        cell["pair"]["y"],
                    ]
        for name in wormholes:
            w = wormholes[name]
            assert len(w) == 4
            self.set_portal(w[0], w[1], w[2], w[3], name)
            self.set_portal(w[2], w[3], w[0], w[1], name)

    def construct_map(self):
        for x in self.maps:
            for y in self.maps[x]:
                self.move_none(self.maps[x][y])
        for x in self.maps:
            for y in self.maps[x]:
                for d in [
                    DIRECTION.UP,
                    DIRECTION.DOWN,
                    DIRECTION.LEFT,
                    DIRECTION.RIGHT,
                ]:
                    p = self.maps[x][y]
                    next_p = self.move_one_step(p, d)
                    p.next[d] = next_p.next[DIRECTION.STAY]
                    if p != next_p and next_p.portal:
                        p.next[d] = next_p.portal


map_info = MapInfo(map_json, calculate_path=True)
print(map_info.path.get_cost(map_info.maps[0][0], map_info.maps[23][23]))
print(map_info.maps[0][0].next)
print(map_info.maps[0][1].next)
print(map_info.path.get_cost(map_info.maps[0][23], map_info.maps[23][0]))
print("map_info done")


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
        if px >= 0 and px < cols and py >= 0 and py < rows and maze[py][px].wall:
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
            if maze[i][j].wall:
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
                if op not in marked and not maze[op[0]][op[1]].wall:
                    marked.add(op)
                    point_to_island[op] = island
        outlines.append(list(marked))

    return (outlines, point_to_island)


# Will return all the islands with their vertices, and outline points
def hull(maze):
    # rows = len(maze)
    # cols = len(maze[0])
    (outlines, point_to_island) = outline(maze)
    # corner_to_island = {}
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
islands = hull(map_info.maps)
print(len(islands))


def use_attacker(agent, enemies, powerup_clock) -> str:
    # record powerups
    if "passwall" in agent["self_agent"]["powerups"]:
        passwall = agent["self_agent"]["powerups"]["passwall"]
    else:
        passwall = 0
    current_pos = (agent["self_agent"]["x"], agent["self_agent"]["y"])
    # if agent["self_agent"]["id"] == 1:
    #     print("id: 1", current_pos, enemies)

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

    path = rust_perf.catch_enemies_using_powerup(
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

    next_move = get_direction(current_pos, path[0])
    for powerup, _ in powerup_clock.items():
        powerup_clock[powerup] += 1
    return next_move


def use_defender(agent, eaten_set, step, powerup_clock, defender_scatter) -> str:
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
    attacker_location = []
    allies_location = []
    has_sword = False
    for other_agent in other_agent_list:
        if other_agent["role"] == "DEFENDER":
            allies_location.append((other_agent["x"], other_agent["y"]))
        else:
            attacker_location.append((other_agent["x"], other_agent["y"]))
            if "sword" in other_agent["powerups"]:
                has_sword = True

    # strategy one (corner)
    if (len(attacker_location) >= 1 and shield < 3) or has_sword:
        next_move = rust_perf.check_stay_or_not(
            current_pos, attacker_location, passwall, eaten_set
        )
        # print(agent_id, current_pos, attacker_location, next_move, passwall)
        return next_move

    if agent_id in [4, 5, 7]:
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


# load map
with open("map.json") as f:
    map = json.load(f)
game = Game(map)


# init game
win_count = 0
attacker_score = 0
defender_score = 0
seeds = [random.randint(0, 1000000) for _ in range(5)]
# seeds = [268836]
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

        # defender_actions = {
        #     _id: random.choice(ACTIONS) for _id in defender_state.keys()
        # }
        defender_actions = {
            _id: use_defender(
                defender_state[_id],
                eatten_set,
                step,
                powerup_clock,
                defender_scatter,
            )
            for _id in defender_state.keys()
        }

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

    defender_state = game.get_agent_states_by_player("attacker")
    for k, v in defender_state.items():
        print("attacker", k, v["self_agent"]["score"])
    defender_state = game.get_agent_states_by_player("defender")
    for k, v in defender_state.items():
        print("defender", k, v["self_agent"]["score"])

print("Win rate is ", win_count / len(seeds))
print(f"Attacker score: {attacker_score} vs Defender score: {defender_score}")
