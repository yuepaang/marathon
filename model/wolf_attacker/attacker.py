import random
import numpy as np
from math import *
from copy import deepcopy
from typing import List, Dict, Tuple
from itertools import product, combinations
from model.utils import *
import rust_perf


class Attacker:
    def __init__(self, map_size: Tuple[int, int], agents: List[int],
                 opponents: List[int], walls: List[Tuple]):
        self.map_size = map_size
        self.agents = agents
        self.opponents = opponents
        self.walls = walls
        self.portal = {
            (0, 0): (23, 23),
            (23, 23): (0, 0),
            (6, 17): (17, 6),
            (17, 6): (6, 17),
            (6, 6): (17, 17),
            (17, 17): (6, 6),
        }

        # ROLE
        self.ROLE = "ATTACKER"

        # ACTION
        self.STAY = "STAY"
        self.UP = "UP"
        self.DOWN = "DOWN"
        self.LEFT = "LEFT"
        self.RIGHT = "RIGHT"

        # TASK
        self.FREE = "free"
        self.PATROL = "patrol"
        self.CHASE = "chase"
        self.HUNT = "hunt"

        self.to_hunt = None
        self.in_sign = set()
        self.epsilon = 1e-6
        self.score = 0
        self.round = 0
        self.opp_state = dict()
        self.agt_state = dict()
        self.miss_count = dict()
        for id in agents:
            self.agt_state[id] = dict()
            self.agt_state[id]["pos"] = (0, 0)
            self.agt_state[id]["score"] = 0
            self.agt_state[id]["task"] = ""
            self.agt_state[id]["target"] = (-1, -1)

        for id in opponents:
            self.opp_state[id] = dict()
            self.opp_state[id]["esc_pos"] = None
            self.opp_state[id]["pos"] = (0, 0)
            self.opp_state[id]["score"] = 0
            self.opp_state[id]["vision_range"] = 2
            self.miss_count[id] = 0

        self.warmup_points = [(11, 12), (6, 13), (6, 1), (1, 1)]
        # self.warmup_points = [(11, 12), (4, 22), (6, 1), (22, 17)]
        self.patrol_points = [(11, 11), (3, 3), (20, 20), (3, 20), (20, 3)]
        self.patrol_points2 = [(11, 2), (11, 21), (2, 11), (21, 12)]

    def _coor_action(self, target: Tuple[int], start: Tuple[int]):
        if target[0] - start[0] > 0:
            return self.RIGHT
        if target[0] - start[0] < 0:
            return self.LEFT
        if target[1] - start[1] > 0:
            return self.DOWN
        if target[1] - start[1] < 0:
            return self.UP
        return self.STAY

    def _action_direction(self, action: str):
        if action == self.UP:
            return (0, -1)
        if action == self.DOWN:
            return (0, 1)
        if action == self.LEFT:
            return (-1, 0)
        if action == self.RIGHT:
            return (1, 0)
        return (0, 0)

    def _dist(self, coor1: Tuple[int], coor2: Tuple[int]):
        return abs(coor1[0] - coor2[0]) + abs(coor1[1] - coor2[1])

    def _mindist_match(self, coors: List[Tuple[int]],
                       blocked_pos: List[Tuple[int]]):
        res_agents = list(self.agents)
        res_coors = list(coors)
        match = dict()

        while len(res_agents) > 0 and len(res_coors) > 0:
            dist = np.zeros(shape=(len(res_agents), len(res_coors)))
            for i in range(len(res_agents)):
                id = res_agents[i]
                apos = self.agt_state[id]["pos"]
                for j in range(len(res_coors)):
                    # d = self._dist(apos, res_coors[j])
                    d, _ = self._safe_path(apos, res_coors[j], blocked_pos)
                    dist[i, j] = d

            # 每个agent的距离最短点
            agent_min = dist.argmin(axis=1)

            # 点找agent
            for j in range(len(res_coors)):
                ids = np.where(agent_min == j)[0]
                num = len(ids)
                # 唯一最短直接匹配
                if num == 1:
                    id = res_agents[ids[0]]
                    match[id] = res_coors[j]
                    res_agents.remove(id)
                    res_coors.remove(res_coors[j])
                    break
                # 多个最短选最短
                elif num > 1:
                    idx = dist[:, j].argmin()
                    id = res_agents[idx]
                    match[id] = res_coors[j]
                    res_agents.remove(id)
                    res_coors.remove(res_coors[j])
                    break

        return match

    def update(self, obs: Dict):
        self.to_hunt = None
        round_score = 0
        target_opp = None
        chase_id = list()
        chase_pos = list()
        self.in_sign = set()
        for id, view in obs.items():
            round_score += view["self_agent"]["score"]
            self.agt_state[id]["last_pos"] = self.agt_state[id]["pos"]
            self.agt_state[id]["pos"] = (view["self_agent"]["x"],
                                         view["self_agent"]["y"])
            self.agt_state[id]["score"] = view["self_agent"]["score"]
            if (self.agt_state[id]["task"] == self.PATROL and
                    self.agt_state[id]["pos"] == self.agt_state[id]["target"]):
                self.agt_state[id]["task"] = self.FREE
                self.agt_state[id]["target"] = (-1, -1)
            elif self.agt_state[id]["task"] in {self.CHASE, self.HUNT}:
                target_opp = self.agt_state[id]["target"]
                chase_id.append(id)
                chase_pos.append(self.agt_state[id]["pos"])

            for other_agent in view["other_agents"]:
                if other_agent["role"] != self.ROLE:
                    opp_id = other_agent["id"]
                    pos = (other_agent["x"], other_agent["y"])
                    self.opp_state[opp_id]["pos"] = pos
                    self.opp_state[opp_id]["score"] = other_agent["score"]
                    self.opp_state[opp_id]["vision_range"] = other_agent[
                        "vision_range"]
                    self.opp_state[opp_id]["esc_pos"] = None
                    self.in_sign.add(opp_id)
                    self.miss_count[opp_id] = 0

        if round_score > self.score and target_opp not in self.in_sign:
            # 分数增加而且追逐目标不在视野内
            target_opp = None
            self.to_hunt = None
            for id in self.agents:
                if self.agt_state[id]["task"] in {self.CHASE, self.HUNT}:
                    self.agt_state[id]["task"] = self.FREE

        if target_opp is not None:
            opp_pos = self.opp_state[target_opp]["pos"]
            x, y = list(zip(*chase_pos))
            if isin_quadrilaterals(x, y, opp_pos):
                self.to_hunt = target_opp
            # tris = combinations(chase_pos, 3)
            # for tri in tris:
            #     inside = set(all_valid_points(tri))
            #     if opp_pos in inside:
            #         self.to_hunt = target_opp
            #         break

        for opp_id in self.opponents:
            if opp_id in self.in_sign:
                continue
            portal = None
            for p1, p2 in self.portal.items():
                if self._dist(self.opp_state[opp_id]["pos"], p1) <= 1:
                    portal = p2
                    break
            if portal is not None:
                self.opp_state[opp_id]["pos"] = portal
            elif self.opp_state[opp_id]["esc_pos"] is not None:
                self.opp_state[opp_id]["pos"] = self.opp_state[opp_id][
                    "esc_pos"]
            self.miss_count[opp_id] += 1

        if target_opp is not None and self.miss_count[target_opp] >= 3:
            for id in self.agents:
                if self.agt_state[id]["task"] in {self.CHASE, self.HUNT}:
                    self.agt_state[id]["task"] = self.FREE

        # 第一轮对手在镜像位置
        if self.round == 0:
            for i in range(len(self.opponents)):
                opp_id = self.opponents[i]
                agt = self.agt_state[self.agents[i]]
                self.opp_state[opp_id]["pos"] = (self.map_size[0] - 1 -
                                                 agt["pos"][0],
                                                 self.map_size[1] - 1 -
                                                 agt["pos"][1])
                self.opp_state[opp_id]["score"] = 0
                self.opp_state[opp_id]["vision_range"] = 2

        self.score = round_score

    def _safe_area(self, coor: Tuple[int], vision_range: int):
        vision_range -= 1
        safe_area = set()
        xmin = max(coor[0] - vision_range, 0)
        xmax = min(coor[0] + vision_range, self.map_size[0] - 1)
        ymin = max(coor[1] - vision_range, 0)
        ymax = min(coor[1] + vision_range, self.map_size[1] - 1)
        for x in range(xmin, xmax + 1):
            for y in range(ymin, ymax + 1):
                if (x, y) in self.walls:
                    continue
                safe_area.add((x, y))

        surrounded_square = set()
        xmin = max(coor[0] - vision_range - 1, 0)
        xmax = min(coor[0] + vision_range + 1, self.map_size[0] - 1)
        ymin = max(coor[1] - vision_range - 1, 0)
        ymax = min(coor[1] + vision_range + 1, self.map_size[1] - 1)
        for x in range(xmin, xmax + 1):
            for y in [ymin, ymax]:
                if (x, y) in self.walls or (x, y) in safe_area:
                    continue
                surrounded_square.add((x, y))
        for x in [xmin, xmax]:
            for y in range(ymin, ymax + 1):
                if (x, y) in self.walls or (x, y) in safe_area:
                    continue
                surrounded_square.add((x, y))

        return list(safe_area), list(surrounded_square)

    def _escape_direction(self, coor: Tuple[int], vision_range: int):
        in_sign = list()
        xmin = coor[0] - vision_range
        xmax = coor[0] + vision_range
        ymin = coor[1] - vision_range
        ymax = coor[1] + vision_range
        # print(xmin, xmax, ymin, ymax)
        for id, state in self.agt_state.items():
            pos = state["pos"]
            if (xmin <= pos[0] <= xmax and ymin <= pos[1] <= ymax):
                in_sign.append(pos)

        direction = np.array((0.0, 0.0))
        for pos in in_sign:
            direction += (np.array(coor) -
                          np.array(pos)) / (self._dist(coor, pos) + 1e-6)

        return direction.tolist()

    def _crosspoint(self, coor: Tuple[int], vision_range: int,
                    direction: List[float], edge: List[Tuple[int]]):
        if abs(direction[0]) >= abs(direction[1]):
            alpha = 1
            if direction[0] < 0:
                alpha = -1
            x = coor[0] + alpha * (vision_range + 1)
            y = (x - coor[0]) * direction[0] + direction[1] + coor[1]
        else:
            alpha = 1
            if direction[1] < 0:
                alpha = -1
            y = coor[1] + alpha * (vision_range + 1)
            if direction[0] == 0:
                x = coor[0]
            else:
                x = (y - coor[1] - direction[1]) / direction[0] + coor[0]

        pos = (x, y)
        min_dist = self.map_size[0] + self.map_size[1] + 1
        crosspoint = edge[0]
        for point in edge[1:]:
            dist = self._dist(pos, point)
            if dist < min_dist:
                min_dist = dist
                crosspoint = point

        return crosspoint

    def _ang(self, a: Tuple[int], b: Tuple[int], c: Tuple[int]):
        ang = degrees(
            atan2(c[1] - b[1], c[0] - b[0]) - atan2(a[1] - b[1], a[0] - b[0]))
        return abs(ang)

    # def _area(self, a: Tuple[int], b: Tuple[int], c: Tuple[int]):
    #     # Area = 1/2[x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)]
    #     area = 0.5 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] *
    #                   (a[1] - b[1]))
    #     return area

    def _area(self, x, y):
        npx = np.array(x)
        npy = np.array(y)
        return 0.5 * np.abs(
            np.dot(npx, np.roll(npy, 1)) - np.dot(npy, np.roll(npx, 1)))

    def _surrounded_points_minmaxang(self, point: Tuple[int],
                                     edge: List[Tuple[int]]):
        if len(edge) <= 2:
            return [point] + edge

        points = [[point], edge, edge]
        candis = product(*points)
        min_ang = 361
        surrounded_points = None
        for combi in candis:
            if len(combi) > len(set(combi)):
                continue
            max_ang = -1
            for points in [
                    combi, (combi[1], combi[0], combi[2]),
                (combi[0], combi[2], combi[1])
            ]:
                ang = self._ang(*points)
                if ang > max_ang:
                    max_ang = ang
            if max_ang < min_ang:
                surrounded_points = combi
                min_ang = max_ang

        return surrounded_points

    def _surrounded_points_maxarea(self, point: Tuple[int],
                                   edge: List[Tuple[int]]):
        if len(edge) <= 3:
            return [point] + edge

        points = [[point], edge, edge, edge]
        candis = product(*points)
        max_area = -1e+6
        surrounded_points = None
        for combi in candis:
            if len(combi) > len(set(combi)):
                continue
            x, y = list(zip(*combi))
            area = self._area(x, y)
            if area > max_area:
                max_area = area
                surrounded_points = combi

        return surrounded_points

    def _single_agent_task(self, id: int, block_area: List[Tuple[int]]):
        action = dict()
        if self.agt_state[id]["task"] == self.PATROL:
            # step_num, path = rust_perf.get_direction_path(
            #     self.agt_state[id]["pos"], self.agt_state[id]["target"],
            #     block_area)
            step_num, path = self._safe_path(self.agt_state[id]["pos"],
                                             self.agt_state[id]["target"],
                                             block_area)
            action["action"] = path[0]
            action["target"] = self.agt_state[id]["target"]
            action["task"] = self.PATROL

        else:
            target = None
            max_dist = -1
            for point in self.patrol_points:
                dist = 0
                for _, state in self.agt_state.items():
                    dist += self._dist(point, state["pos"])
                if dist > max_dist:
                    max_dist = dist
                    target = point
            # step_num, path = rust_perf.get_direction_path(
            #     self.agt_state[id]["pos"], target, block_area)
            step_num, path = self._safe_path(self.agt_state[id]["pos"], target,
                                             [])
            action["action"] = path[0]
            action["target"] = target
            action["task"] = self.PATROL

        return action

    def _safe_path(self, start: Tuple[int], end: Tuple[int],
                   block_area: List[Tuple[int]]):
        _, pre_path = rust_perf.get_direction_path(start, end, [])
        safe_area_tmp = list(block_area)
        pre_coor = start
        for a in pre_path:
            if pre_coor in safe_area_tmp:
                safe_area_tmp.remove(pre_coor)
            d = self._action_direction(a)
            pre_coor = (pre_coor[0] + d[0], pre_coor[1] + d[1])

        try:
            step_num, path = rust_perf.get_direction_path(
                start, end, safe_area_tmp)
            if step_num == 0 or len(path) == 0:
                step_num, path = 0, ["STAY"]
        except Exception as e:
            print(e, start, end)
            step_num, path = 1e+2, ["STAY"]

        return step_num, path

    def _chase_target(self, opp_id: int, real_pos: bool):
        opp_action = dict()
        opp_pos = self.opp_state[opp_id]["pos"]
        vision_range = self.opp_state[opp_id]["vision_range"]
        safe_area, surrounded_square = self._safe_area(opp_pos, vision_range)
        blocked = list(safe_area)
        v = self._escape_direction(opp_pos, vision_range)
        # if real_pos:
        _, next_step = self._safe_area(opp_pos, 1)
        escape_pos = self._crosspoint(opp_pos, 0, v, next_step)
        self.opp_state[opp_id]["esc_pos"] = escape_pos

        crosspoint = self._crosspoint(opp_pos, vision_range, v,
                                      surrounded_square)
        surrounded = self._surrounded_points_maxarea(crosspoint,
                                                     surrounded_square)
        # print("surrounded", surrounded)
        # assert False
        match = self._mindist_match(surrounded, safe_area)

        if len(match) == 0:
            return -1, opp_action

        dists = list()
        ids = list()
        for id in self.agents:
            agent_pos = self.agt_state[id]["pos"]
            opp_step_num, opp_path = self._safe_path(agent_pos, opp_pos, [])
            dists.append(opp_step_num)
            ids.append(id)
        dists = list(zip(dists, ids))
        dists.sort()

        opp_step_num, opp_path = self._safe_path(
            self.agt_state[dists[0][1]]["pos"], opp_pos, [])
        opp_action[dists[0][1]] = dict()
        opp_action[dists[0][1]]["action"] = opp_path[0]
        opp_action[dists[0][1]]["task"] = self.CHASE
        opp_action[dists[0][1]]["target"] = opp_id
        direction = self._action_direction(opp_path[0])
        block_pos = (self.agt_state[dists[0][1]]["pos"][0] + direction[0],
                     self.agt_state[dists[0][1]]["pos"][1] + direction[1])
        blocked.append(block_pos)

        opp_step_num, opp_path = self._safe_path(
            self.agt_state[dists[1][1]]["pos"], opp_pos, [])
        opp_action[dists[1][1]] = dict()
        opp_action[dists[1][1]]["action"] = opp_path[0]
        opp_action[dists[1][1]]["task"] = self.CHASE
        opp_action[dists[1][1]]["target"] = opp_id
        direction = self._action_direction(opp_path[0])
        block_pos = (self.agt_state[dists[1][1]]["pos"][0] + direction[0],
                     self.agt_state[dists[1][1]]["pos"][1] + direction[1])
        blocked.append(block_pos)

        for d, id in dists[2:]:
            if d <= 2:
                opp_step_num, opp_path = self._safe_path(
                    self.agt_state[id]["pos"], opp_pos, [])
                opp_action[id] = dict()
                opp_action[id]["action"] = opp_path[0]
                opp_action[id]["task"] = self.CHASE
                opp_action[id]["target"] = opp_id
                direction = self._action_direction(opp_path[0])
                block_pos = (self.agt_state[dists[0][1]]["pos"][0] +
                             direction[0],
                             self.agt_state[dists[0][1]]["pos"][1] +
                             direction[1])
                blocked.append(block_pos)

        max_dist = -1
        for id, coor in match.items():
            if id in opp_action:
                continue
            opp_action[id] = dict()
            agent_pos = self.agt_state[id]["pos"]
            # if agent_pos in safe_area:
            #     if agent_pos in self.portal:
            #         path = self._coor_action(opp_pos, agent_pos)
            #         step_num, path = 1, [path]
            #     else:
            #         step_num, path = self._safe_path(agent_pos, opp_pos, [])
            #         # step_num, path = 0, ["STAY"]
            opp_step_num, opp_path = self._safe_path(agent_pos, opp_pos, [])
            if agent_pos in self.portal:
                step_num, path = opp_step_num, opp_path
            else:
                step_num, path = self._safe_path(agent_pos, coor, blocked)

            if step_num == 0 or len(path) == 0:
                step_num = 0
                opp_action[id]["action"] = "STAY"
            else:
                opp_action[id]["action"] = path[0]
            opp_action[id]["task"] = self.CHASE
            opp_action[id]["target"] = opp_id
            # opp_action[id]["chase_pos"] = coor
            if step_num > max_dist:
                max_dist = step_num

        max_dist = max(max_dist, 1e-6) + 1e-6

        for id_ in self.agents:
            if id_ in opp_action:
                continue
            opp_action[id_] = self._single_agent_task(id, safe_area)

        score_tmp = (self.opp_state[opp_id]["score"] / 2 + 4) / log(max_dist)
        return score_tmp, opp_action

    def _hunt_target(self, opp_id: int):
        opp_action = dict()
        opp_pos = self.opp_state[opp_id]["pos"]
        opp_next = list()
        for a in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            next_pos = (opp_pos[0] + a[0], opp_pos[1] + a[1])
            if next_pos in self.walls:
                continue
            opp_next.append(next_pos)

        for id, state in self.agt_state.items():
            if state["pos"] in opp_next:
                opp_next.remove(state["pos"])
            if state["target"] == opp_id:
                opp_action[id] = dict()
                # step_num, path = rust_perf.get_direction_path(
                #     state["pos"], opp_pos, [])
                step_num, path = self._safe_path(state["pos"], opp_pos, [])
                opp_action[id]["action"] = path[0]
                opp_action[id]["target"] = opp_id
                opp_action[id]["task"] = self.HUNT

        if len(opp_next) == 1:
            for id, state in self.agt_state.items():
                if self._dist(opp_next[0], state["pos"]) == 1:
                    opp_action[id]["action"] = self._coor_action(
                        opp_next[0], state["pos"])
                    opp_action[id]["target"] = opp_id
                    opp_action[id]["task"] = self.HUNT

        for id_ in self.agents:
            if id_ in opp_action:
                continue
            opp_action[id_] = self._single_agent_task(id, [])

        return opp_action

    def _continue_hunting_ang(self, hunters: List[int]):
        opp_pos = self.opp_state[self.to_hunt]
        combis = combinations(hunters, 2)
        maxang = -1
        for comb in combis:
            ang = self._ang(self.agt_state[comb[0]]["pos"], opp_pos,
                            self.agt_state[comb[1]]["pos"])
            if ang > maxang:
                maxang = ang
        return maxang >= 90

    def _continue_hunting_cover(self, hunters: List[int]):
        to_hunt = False
        opp_pos = self.opp_state[self.to_hunt]
        hunt_pos = list()
        for id in hunters:
            hunt_pos.append(self.agt_state[id]["pos"])

        x, y = list(zip(*hunt_pos))
        if isin_quadrilaterals(x, y, opp_pos):
            to_hunt = True

        # tris = combinations(hunt_pos, 3)
        # for tri in tris:
        #     inside = set(all_valid_points(tri))
        #     if opp_pos in inside:
        #         to_hunt = True
        #         break
        return to_hunt

    def step(self):
        print("round:", self.round, "score", self.score)
        action = dict()

        hunt = self.to_hunt is not None
        chase = False
        hunters = list()
        for id, state in self.agt_state.items():
            print("state", id, state)
            if state["task"] == self.HUNT:
                self.to_hunt = state["target"]
                hunt = True
                hunters.append(id)
            chase = chase | (state["task"] == self.CHASE)

        if self.to_hunt is not None and not self._continue_hunting_cover(
                hunters):
            hunt = False
            chase = True

        for id, state in self.opp_state.items():
            print("state", id, state)
        # print("in_sign", self.in_sign, chase, hunt, self.to_hunt)

        if self.round < 20 and len(
                self.in_sign) == 0 and not chase and not hunt:
            match = self._mindist_match(self.warmup_points, [])
            for id, coor in match.items():
                # step_num, path = rust_perf.get_direction_path(
                #     self.agt_state[id]["pos"], coor, [])
                step_num, path = self._safe_path(self.agt_state[id]["pos"],
                                                 coor, [])
                if step_num == 0 or len(path) == 0:
                    action[id] = "STAY"
                else:
                    action[id] = path[0]
                self.agt_state[id]["task"] = self.PATROL
                self.agt_state[id]["target"] = coor
                # print("  ", id, "->", coor, path, self.PATROL)

        elif hunt:
            opp_action = self._hunt_target(self.to_hunt)
            for id, state in opp_action.items():
                action[id] = state["action"]
                self.agt_state[id]["task"] = state["task"]
                self.agt_state[id]["target"] = state["target"]

        elif chase or len(self.in_sign) > 0:
            action_tmp = list()
            score = list()
            for opp_id in self.in_sign:
                score_tmp, opp_action = self._chase_target(opp_id, True)
                score.append(score_tmp)
                action_tmp.append(opp_action)

            if len(score) == 0:
                opp_id = None
                for state in self.agt_state.values():
                    if state["target"] in self.opponents:
                        opp_id = state["target"]
                        break
                score_tmp, opp_action = self._chase_target(opp_id, False)
                score.append(score_tmp)
                action_tmp.append(opp_action)

            max_score = -1e+10
            max_action = None
            for i in range(len(score)):
                if score[i] > max_score:
                    max_score = score[i]
                    max_action = action_tmp[i]

            for id, state in max_action.items():
                action[id] = state["action"]
                self.agt_state[id]["task"] = state["task"]
                self.agt_state[id]["target"] = state["target"]
                # print("  target:", state["target"], "agent:", id, "->",
                #   state["action"], state["task"])

        else:
            # print("====== DISPERSE")
            stay = set()
            match = self._mindist_match(self.patrol_points, [])
            for id, coor in match.items():
                # step_num, path = rust_perf.get_direction_path(
                #     self.agt_state[id]["pos"], coor, [])
                step_num, path = self._safe_path(self.agt_state[id]["pos"],
                                                 coor, [])
                if step_num == 0 or len(path) == 0:
                    action[id] = self.STAY
                else:
                    action[id] = path[0]
                if action[id] == self.STAY:
                    stay.add(id)
                self.agt_state[id]["task"] = self.PATROL
                self.agt_state[id]["target"] = coor
                # print("  ", id, "->", coor, path, self.PATROL)

            if len(stay) > 0:
                match = self._mindist_match(self.patrol_points2, [])
                for id, coor in match.items():
                    if id not in stay:
                        continue
                    # step_num, path = rust_perf.get_direction_path(
                    #     self.agt_state[id]["pos"], coor, [])
                    step_num, path = self._safe_path(self.agt_state[id]["pos"],
                                                     coor, [])
                    if step_num == 0 or len(path) == 0:
                        action[id] = self.STAY
                    else:
                        action[id] = path[0]
                    if action[id] == self.STAY:
                        stay.add(id)
                    self.agt_state[id]["task"] = self.PATROL
                    self.agt_state[id]["target"] = coor

        self.round += 1
        # print(action)

        return action
