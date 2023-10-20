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
import itertools
import json
import math
import time
from typing import Dict, List, Union

import numpy as np
from scipy.spatial import ConvexHull
from game import Game, Powerup
import random
from numba import jit

import rust_perf


class DIRECTION(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STAY = "STAY"


with open("map.json") as f:
    map_json = json.load(f)


class Point:
    def __init__(self, _x: int, _y: int, _point: int):
        self.x = _x  # type: int
        self.y = _y  # type: int
        self.next = dict()  # type: Dict[DIRECTION, Union[None, Point]]
        self.wall = False  # type: bool
        self.portal = None  # type: Union[None, Point]
        self.name = ""
        self.point = _point

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


def max_xy(p1: Point, p2: Point):
    return max(abs(p1.x - p2.x), abs(p1.y - p2.y))


class Path:
    def __init__(
        self,
        _maps: Dict[int, Dict[int, Point]],
        _width: int,
        _height: int,
        _vision: int,
    ):
        self.maps = _maps  # type: Dict[int, Dict[int, Point]]
        self.width = _width  # type: int
        self.height = _height  # type: int
        self.vision = _vision  # type: int
        self.node_num = self.width * self.height  # type: int
        self.inf = float("inf")
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

        # cal danger index
        self.is_danger_index = [False for _ in range(self.node_num)]
        self.is_eat_danger_index = [False for _ in range(self.node_num)]
        for i in range(self.node_num):
            p = self.to_point(i)
            for d in [
                DIRECTION.UP,
                DIRECTION.DOWN,
                DIRECTION.LEFT,
                DIRECTION.RIGHT,
                DIRECTION.STAY,
            ]:
                if max_xy(p, p.next[d]) > self.vision:
                    self.is_danger_index[self.to_index(p.next[d])] = True
                    for dd in [
                        DIRECTION.UP,
                        DIRECTION.DOWN,
                        DIRECTION.LEFT,
                        DIRECTION.RIGHT,
                        DIRECTION.STAY,
                    ]:
                        self.is_eat_danger_index[
                            self.to_index(p.next[d].next[dd])
                        ] = True

    def get_cost(self, start: Point, end: Point):
        return self.dist[self.to_index(start)][self.to_index(end)]

    def get_cost_index(self, start: int, end: int):
        return self.dist[start][end]

    def to_point(self, index: int) -> Point:
        return self.maps[index % self.width][index // self.width]

    def to_index(self, point: Point) -> int:
        return point.x + point.y * self.width

    def to_index_xy(self, x: int, y: int) -> int:
        return x + y * self.width

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
                self.maps[x][y] = Point(x, y, 0)
        self.load_map()
        self.construct_map()
        if calculate_path:
            self.path = Path(self.maps, self.width, self.height, self.vision)

        self.vision_grids = [[] for _ in range(self.path.node_num)]
        self.vision_grids_1 = [[] for _ in range(self.path.node_num)]
        self.vision_grids_3 = [[] for _ in range(self.path.node_num)]
        self.vision_grids_cross = [[] for _ in range(self.path.node_num)]

        for n in range(self.path.node_num):
            p = self.path.to_point(n)
            x = p.x
            y = p.y
            for i in range(
                max(x - self.vision, 0), min(x + self.vision, self.width - 1) + 1
            ):
                for j in range(
                    max(y - self.vision, 0), min(y + self.vision, self.height - 1) + 1
                ):
                    self.vision_grids[n].append(self.path.to_index_xy(i, j))

            for i in range(max(x - 1, 0), min(x + 1, self.width - 1) + 1):
                for j in range(max(y - 1, 0), min(y + 1, self.height - 1) + 1):
                    self.vision_grids_3[n].append(self.path.to_index_xy(i, j))

            for d in [DIRECTION.UP, DIRECTION.DOWN, DIRECTION.LEFT, DIRECTION.RIGHT]:
                self.vision_grids_1[n].append(
                    self.path.to_index(self.path.to_point(n).next[d])
                )

            self.vision_grids_cross[n].extend(
                [
                    self.path.to_index_xy(x, y),
                    self.path.to_index_xy(max(x - 1, 0), y),
                    self.path.to_index_xy(min(x + 1, self.width), y),
                    self.path.to_index_xy(x, max(y - 1, 0)),
                    self.path.to_index_xy(x, min(y + 1, self.height)),
                ]
            )

        # for idx in self.vision_grids[0]:
        #     print(self.path.to_point(idx).__dict__)

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
            x = cell["x"]
            y = cell["y"]
            if cell["type"] == "WALL":
                self.set_wall(x, y)
            elif cell["type"] == "PORTAL":
                name = cell["name"]
                if name not in wormholes:
                    wormholes[name] = [
                        x,
                        y,
                        cell["pair"]["x"],
                        cell["pair"]["y"],
                    ]
            elif cell["type"] == "COIN":
                self.maps[x][y].point = 2
            # TODO: point for powerup
            elif cell["type"] == "POWERUP":
                self.maps[x][y].point = 6
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


class Naga:
    def __init__(self, map_info: MapInfo, agent_states, role: str) -> None:
        self.map_info = map_info
        self.agent_states = agent_states
        self.role = role
        self.id_to_agent = dict()
        for i in range(4):
            if role == "DEFENDER":
                self.id_to_agent[i] = i + 4
            else:
                self.id_to_agent[i] = i

        self.action = ["STAY", "LEFT", "RIGHT", "DOWN", "UP"]
        self.agent_count = len(agent_states)
        self.score_num = len(self.action) ** self.agent_count
        self.map_direction = []
        self.combinations = list(
            itertools.product(self.action, repeat=self.agent_count)
        )
        self.map_direction = [
            {self.id_to_agent[i]: move for i, move in enumerate(combination)}
            for combination in self.combinations
        ]

        self.power_scores = [0.0 for _ in range(map_info.path.node_num)]
        self.env_score_limit = [0.0 for _ in range(map_info.path.node_num)]
        self.visit_time = [0 for _ in range(map_info.path.node_num)]
        for i in range(map_info.path.node_num):
            p = map_info.path.to_point(i)
            if p.wall is False and p.portal is False:
                self.env_score_limit[i] = 1.0 / ((2 * map_info.vision + 1) ** 2)
        self.env_score = self.env_score_limit

        self.all_danger = [0.0 for _ in range(self.map_info.path.node_num)]
        self.vec_danger = [
            [100.0 for _ in range(self.map_info.path.node_num)] for _ in range(4)
        ]
        self.all_enemy_in_vision = False
        self.danger_in_vision = [False for _ in range(self.map_info.path.node_num)]
        self.danger_eat_in_vision = [False for _ in range(self.map_info.path.node_num)]
        self.vision_grids_index = [False for _ in range(self.map_info.path.node_num)]
        self.n_enemy = 0

        self.map_enemy_predict = dict()
        self.map_enemy_loc = dict()
        self.map_enemy_repeat = dict()

    def update_score(self, step: int):
        for i in range(self.map_info.path.node_num):
            if step - self.visit_time[i] > 15:
                self.env_score[i] = min(
                    self.env_score_limit[i] / 25 * (step - 15 - self.visit_time[i]),
                    self.env_score_limit[i],
                )

        self.other_agents = {}
        self.vision_grids_index = [False for _ in range(self.map_info.path.node_num)]
        for _, agent_state in self.agent_states.items():
            for other_agent in agent_state["other_agents"]:
                if other_agent["role"] != self.role:
                    if other_agent["id"] not in self.other_agents:
                        self.other_agents[other_agent["id"]] = other_agent

            for n in self.map_info.vision_grids[
                self.map_info.path.to_index_xy(
                    agent_state["self_agent"]["x"], agent_state["self_agent"]["y"]
                )
            ]:
                self.power_scores[n] = 0.0
                self.env_score[n] = 0.0
                self.visit_time[n] = step

            for x in self.map_info.maps:
                for y in self.map_info.maps[x]:
                    # powerup_index = self.map_info.path.to_index_xy(
                    #     powerup["x"], powerup["y"]
                    # )
                    # p = self.map_info.path.to_point(powerup_index)
                    powerup_index = self.map_info.path.to_index_xy(x, y)
                    p = self.map_info.maps[x][y]
                    if p.point == 0:
                        continue
                    self.env_score_limit[powerup_index] = p.point / self.map_info.vision
                    self.power_scores[powerup_index] = p.point

            # update map coin&power
            for powerup in agent_state["powerups"]:
                if powerup["powerup"] == "Powerup.SWORD":
                    powerup_index = self.map_info.path.to_index_xy(
                        powerup["x"], powerup["y"]
                    )
                    p = self.map_info.path.to_point(powerup_index)
                    p.point = 0

            # stand in point so no value
            agent_index = self.map_info.path.to_index_xy(
                agent_state["self_agent"]["x"], agent_state["self_agent"]["y"]
            )
            p = self.map_info.path.to_point(agent_index)
            p.point = 0

            # update danger
            for idx in self.map_info.vision_grids[agent_index]:
                v = self.map_info.path.to_point(idx)
                if v.wall:
                    continue
                self.vision_grids_index[idx] = True

            # all enemies in vision
            self.n_enemy = 0
            for o_id, other_agent in self.other_agents.items():
                pos = self.map_info.path.to_index_xy(other_agent["x"], other_agent["y"])
                self.n_enemy += 1

                danger = self.vec_danger[o_id]
                inside, outside = 0.0, 0.0
                outside_count = 0
                for i in range(self.map_info.path.node_num):
                    p = self.map_info.path.to_point(i)
                    if p.wall:
                        continue

                    if self.vision_grids_index[i]:
                        inside += danger[i]
                        danger[i] = 0.0
                    else:
                        outside += danger[i]
                        outside_count += 1

                # may be dead
                for i in range(self.map_info.path.node_num):
                    p = self.map_info.path.to_point(i)
                    if p.wall:
                        continue
                    if not self.vision_grids_index[i]:
                        danger[i] += inside / outside_count

                # print(f"id:{o_id}, inside:{inside}, outside:{outside}")
                # raise Exception("e")

            for o_id, other_agent in self.other_agents.items():
                pos = self.map_info.path.to_index_xy(other_agent["x"], other_agent["y"])
                # print("attacker: ", (other_agent["x"], other_agent["y"]))
                # FIXME:
                danger = self.vec_danger[o_id]
                inside, outside = 0.0, 0.0
                outside_count = 0
                danger = [0.0 for _ in range(self.map_info.path.node_num)]
                danger[pos] = self.map_info.path.node_num * 100.0

                danger_temp = [0.0 for _ in range(self.map_info.path.node_num)]
                next_index = []
                for i in range(self.map_info.path.node_num):
                    p = self.map_info.path.to_point(i)
                    if p.wall:
                        continue
                    if danger[i] == 0.0:
                        continue
                    next_index = []
                    for np in p.next.values():
                        next_index.append(self.map_info.path.to_index(np))
                    for ni in next_index:
                        danger_temp[ni] += danger[i] / len(next_index)

                    # print(danger_temp)
                    # print(danger)
                    # raise Exception("e")
                self.vec_danger[o_id] = danger_temp

            if self.role == "DEFENDER":
                for o_id, other_agent in self.other_agents.items():
                    danger_temp_temp = [0.0 for _ in range(self.map_info.path.node_num)]
                    next_index = []
                    for i in range(self.map_info.path.node_num):
                        p = self.map_info.path.to_point(i)
                        if p.wall:
                            continue
                        if danger[i] == 0.0:
                            continue
                        next_index = []
                        for np in p.next.values():
                            next_index.append(self.map_info.path.to_index(np))
                        for ni in next_index:
                            danger_temp_temp[ni] += danger[i] / len(next_index)

                        # print(danger_temp)
                        # print(danger)
                        # raise Exception("e")
                    self.vec_danger[o_id] = danger_temp_temp

            self.all_enemy_in_vision = False
            if self.n_enemy == 4:
                self.all_enemy_in_vision = True
            # TODO: passwall
            for o_id, other_agent in self.other_agents.items():
                pos = self.map_info.path.to_index_xy(other_agent["x"], other_agent["y"])
                p = self.map_info.path.to_point(pos)
                for d in [
                    DIRECTION.STAY,
                    DIRECTION.UP,
                    DIRECTION.DOWN,
                    DIRECTION.LEFT,
                    DIRECTION.RIGHT,
                ]:
                    next_idx = self.map_info.path.to_index(p.next[d])
                    self.danger_in_vision[next_idx] = True
                    if p.next[d].portal:
                        portal_idx = self.map_info.path.to_index(p.next[d].portal)
                        self.danger_in_vision[portal_idx] = True
                    for dd in [
                        DIRECTION.STAY,
                        DIRECTION.UP,
                        DIRECTION.DOWN,
                        DIRECTION.LEFT,
                        DIRECTION.RIGHT,
                    ]:
                        next_next_idx = self.map_info.path.to_index(p.next[d].next[dd])
                        self.danger_eat_in_vision[next_next_idx] = True
                        if p.next[d].next[dd].portal:
                            portal_next_idx = self.map_info.path.to_index(
                                p.next[d].next[dd].portal
                            )
                            self.danger_eat_in_vision[portal_next_idx] = True
                # print(self.danger_in_vision)
                # print(self.map_info.path.is_danger_index)
                # raise Exception("e")

    def update_dist(self):
        node_num = self.map_info.path.node_num
        G = [[float("inf") for _ in range(node_num)] for _ in range(node_num)]
        self.all_danger = [0.0 for _ in range(node_num)]
        for o_id, other_agent in self.other_agents.items():
            if other_agent["role"] == "ATTACKER":
                danger = self.vec_danger[o_id]
                for i in range(node_num):
                    self.all_danger[i] += danger[i]
        next_loc_set = set()
        for o_id, other_agent in self.other_agents.items():
            if other_agent["role"] == "ATTACKER":
                pos = self.map_info.path.to_index_xy(other_agent["x"], other_agent["y"])
                op = self.map_info.path.to_point(pos)
                for d in [
                    DIRECTION.STAY,
                    DIRECTION.UP,
                    DIRECTION.DOWN,
                    DIRECTION.LEFT,
                    DIRECTION.RIGHT,
                ]:
                    next_loc = self.map_info.path.to_index(op.next[d])
                    next_loc_set.add(next_loc)
        for i in range(node_num):
            for dp in self.map_info.path.to_point(i).next.values():
                j = self.map_info.path.to_index(dp)
                if i != j:
                    G[i][j] = 1
                    if (
                        self.map_info.path.is_eat_danger_index[j]
                        or self.map_info.path.is_danger_index[j]
                    ) and not self.all_enemy_in_vision:
                        G[i][j] = 100
                    if dp.portal:
                        pi = self.map_info.path.to_index(dp.portal)
                        if (
                            self.map_info.path.is_eat_danger_index[pi]
                            or self.map_info.path.is_danger_index[pi]
                        ) and not self.all_enemy_in_vision:
                            G[i][j] = 100
                    if j in next_loc_set:
                        G[i][j] = 100
                else:
                    G[i][j] = 0
        # print(G)
        # raise Exception("e")

    def predict_enemy(self, step: int):
        # self.map_enemy_predict = dict()
        # self.map_enemy_loc = dict()
        # self.map_enemy_repeat = dict()
        # repeat_times = 4
        # for oid, other_agent in self.other_agents.items():
        #     if other_agent["role"] == "DEFENDER":
        #         continue
        #     for repeat_interval in range(1, 5):
        #         if self.map_enemy_predict.get(oid,False):
        #             continue
        #         if repeat_times * repeat_interval >= step:
        #             continue
        #         is_break = False
        #         loc_record = []
        #         for i in range(repeat_times*repeat_interval):
        #             if oid in
        pass

    def eat_coin(self):
        agent_first_coin = dict()
        self.agent_pos = dict()

        for agent_id, agent_state in self.agent_states.items():
            agent_index = self.map_info.path.to_index_xy(
                agent_state["self_agent"]["x"], agent_state["self_agent"]["y"]
            )
            agent_p = self.map_info.path.to_point(agent_index)
            self.agent_pos[agent_id] = agent_p
            shortest_dist = float("inf")
            for coin_index, score in enumerate(self.power_scores):
                if score == 0:
                    continue
                continue_flag = False
                for id_, ci in agent_first_coin.items():
                    if id_ == agent_id:
                        continue
                    if ci == coin_index:
                        continue_flag = True
                        break
                if continue_flag:
                    continue

                dist = self.map_info.path.get_cost_index(coin_index, agent_index)
                if dist < shortest_dist:
                    shortest_dist = dist
                    agent_first_coin[agent_id] = coin_index

        single_direction_score = []
        for agent_id, coin_index in agent_first_coin.items():
            agent_p = self.agent_pos[agent_id]
            coin_p = self.map_info.path.to_point(coin_index)
            for d in [
                DIRECTION.UP,
                DIRECTION.DOWN,
                DIRECTION.LEFT,
                DIRECTION.RIGHT,
            ]:
                cost_a = self.map_info.path.get_cost(agent_p, agent_p.next[d])
                cost_b = self.map_info.path.get_cost(agent_p.next[d], coin_p)
                score = coin_p.point / (max(1.0, cost_a) + cost_b + 1)
                single_direction_score.append((agent_id, d, score))

        dir_score = [0.0 for _ in range(self.score_num)]
        for idx in range(self.score_num):
            md = self.map_direction[idx]
            for sds in single_direction_score:
                if md[sds[0]] == sds[1].value:
                    dir_score[idx] += sds[2]

        # print([s for s in dir_score if s == max(dir_score)])
        # raise Exception("e")
        self.avoid_enemy(dir_score)
        self.out_vision(dir_score)
        self.search_enemy(dir_score)
        self.remove_invalid(dir_score)
        return self.map_direction[dir_score.index(max(dir_score))]

    def avoid_enemy(self, dir_score):
        for idx in range(self.score_num):
            md = self.map_direction[idx]
            next_loc = []
            continue_flag = False
            for agent_id, agent in self.agent_states.items():
                p = self.map_info.path.to_point(
                    self.map_info.path.to_index_xy(
                        agent["self_agent"]["x"], agent["self_agent"]["y"]
                    )
                )
                next_point = p.next[DIRECTION(md[agent_id])]
                if md[agent_id] != DIRECTION.STAY.value and next_point == p:
                    continue_flag = True
                    break
                next_loc.append(self.map_info.path.to_index(next_point))

            if continue_flag:
                continue

            score = 0.0
            for loc in next_loc:
                if self.danger_in_vision[loc] or self.danger_eat_in_vision[loc]:
                    score -= 1e3
            dir_score[idx] += score

    def out_vision(self, dir_score):
        now_loc = []
        for agent_id, agent in self.agent_states.items():
            now_idx = self.map_info.path.to_index_xy(
                agent["self_agent"]["x"], agent["self_agent"]["y"]
            )
            now_loc.append(now_idx)

        for idx in range(self.score_num):
            md = self.map_direction[idx]
            next_loc = []
            continue_flag = False
            for agent_id, agent in self.agent_states.items():
                p = self.map_info.path.to_point(
                    self.map_info.path.to_index_xy(
                        agent["self_agent"]["x"], agent["self_agent"]["y"]
                    )
                )
                next_point = p.next[DIRECTION(md[agent_id])]
                if md[agent_id] != DIRECTION.STAY.value and next_point == p:
                    continue_flag = True
                    break
                next_loc.append(self.map_info.path.to_index(next_point))

            if continue_flag:
                continue

            score = 0.0
            for e_loc in range(self.map_info.path.node_num):
                danger = self.all_danger[e_loc]
                if danger < 1e-2:
                    continue
                for i_loc in next_loc:
                    i_dis = self.map_info.path.get_cost_index(e_loc, i_loc)
                    score -= (40 - i_dis) * danger
            dir_score[idx] += score

    def run_away(self, dir_score):
        reach_size = [
            self.map_info.path.node_num for _ in range(self.map_info.path.node_num)
        ]

    def search_enemy(self, dir_score):
        map_danger: Dict[int, float] = {}
        for c in range(4):
            max_score = float("-inf")
            max_score_loc = 0

            for n in range(map_info.path.node_num):
                point = map_info.path.to_point(n)
                if point.wall:
                    continue

                score = 0.0
                for g in map_info.vision_grids[n]:
                    score += self.all_danger[g]

                if score > max_score:
                    max_score = score
                    max_score_loc = n

            if max_score != 0:  # replace 'equal_double' function
                map_danger[max_score_loc] = max_score
                # print(f"max_score_loc: {max_score_loc} max_score: {max_score}")
                point = map_info.path.to_point(max_score_loc)
                for g in map_info.vision_grids[map_info.path.to_index(point)]:
                    self.all_danger[g] = 0.0

        mu_first_danger = dict()
        for idx in map_danger:
            mu_id = -1
            shortest_dist = float("inf")
            for agent_id, agent in self.agent_states.items():
                if agent_id in mu_first_danger:
                    continue
                pos = self.map_info.path.to_index_xy(
                    agent["self_agent"]["x"], agent["self_agent"]["y"]
                )
                dist = self.map_info.path.get_cost_index(
                    pos,
                    idx,
                )
                if dist < shortest_dist:
                    shortest_dist = dist
                    mu_id = agent_id

            if mu_id != -1 and mu_id not in mu_first_danger:
                mu_first_danger[mu_id] = idx

        single_direction_score = []
        for agent_id, e_index in mu_first_danger.items():
            agent_p = self.agent_pos[agent_id]
            e_p = self.map_info.path.to_point(e_index)
            for d in [
                DIRECTION.UP,
                DIRECTION.DOWN,
                DIRECTION.LEFT,
                DIRECTION.RIGHT,
            ]:
                cost = self.map_info.path.get_cost(agent_p, agent_p.next[d])
                score = map_danger.get(self.map_info.path.to_index(agent_p), 0) / (
                    cost + 1
                )
                single_direction_score.append((agent_id, d, cost, score))

        dir_score = [0.0 for _ in range(self.score_num)]
        for idx in range(self.score_num):
            md = self.map_direction[idx]
            for sds in single_direction_score:
                if md[sds[0]] == sds[1].value:
                    dir_score[idx] += sds[3] / (self.map_info.vision * 5)

    def remove_invalid(self, dir_score):
        for idx in range(self.score_num):
            md = self.map_direction[idx]
            continue_flag = False
            for agent_id, agent in self.agent_states.items():
                p = self.map_info.path.to_point(
                    self.map_info.path.to_index_xy(
                        agent["self_agent"]["x"], agent["self_agent"]["y"]
                    )
                )
                next_point = p.next[DIRECTION(md[agent_id])]
                if md[agent_id] != DIRECTION.STAY.value and next_point == p:
                    continue_flag = True
                    break

            if continue_flag:
                dir_score[idx] -= 1e6


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


# load map
with open("map.json") as f:
    map = json.load(f)
game = Game(map)

# init game
win_count = 0
attacker_score = 0
defender_score = 0
seeds = [random.randint(0, 1000000) for _ in range(2)]
# seeds = [170587]
for seed in seeds:
    game.reset(attacker="attacker", defender="defender", seed=seed)

    step = 0
    start_game_time = time.time()

    # manage the point in the map
    map_info = MapInfo(map_json, calculate_path=True)
    # 16 islands
    islands = hull(map_info.maps)
    print(len(islands))

    # print(map_info.path.get_cost(map_info.maps[0][0], map_info.maps[23][23]))
    # print(map_info.maps[0][0].next)
    # print(map_info.maps[0][1].next)
    # print(map_info.path.get_cost(map_info.maps[0][23], map_info.maps[23][0]))
    print("map_info done")

    # game loop
    while not game.is_over():
        # get game state for player:
        attacker_state = game.get_agent_states_by_player("attacker")
        defender_state = game.get_agent_states_by_player("defender")

        naga = Naga(map_info, defender_state, "DEFENDER")
        naga.update_score(step)
        naga.update_dist()
        # naga.eat_coin()
        # if sum(naga.power_scores) > 0:
        #     print(defender_state)
        #     print(naga.env_score_limit)
        #     print(naga.power_scores)
        #     raise Exception("b")

        attacker_actions = {
            _id: random.choice(naga.action) for _id in attacker_state.keys()
        }
        # attacker_actions = {_id: "STAY" for _id in attacker_state.keys()}

        attacker_locs = set()
        my_locs = {}
        for agent_id, agent in defender_state.items():
            my_locs[agent_id] = (agent["self_agent"]["x"], agent["self_agent"]["y"])
            for other_agent in agent["other_agents"]:
                if other_agent["role"] == "ATTACKER":
                    attacker_locs.add((other_agent["x"], other_agent["y"]))
        defender_actions = naga.eat_coin()
        # if len(attacker_locs) > 1:
        #     print(attacker_locs)
        #     print(my_locs)
        #     print(defender_actions)
        # defender_actions = {
        #     _id: random.choice(ACTIONS) for _id in defender_state.keys()
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

    defender_state = game.get_agent_states_by_player("attacker")
    for k, v in defender_state.items():
        print("attacker", k, v["self_agent"]["score"])
    defender_state = game.get_agent_states_by_player("defender")
    for k, v in defender_state.items():
        print("defender", k, v["self_agent"]["score"])

print("Win rate is ", win_count / len(seeds))
print(f"Attacker score: {attacker_score} vs Defender score: {defender_score}")
