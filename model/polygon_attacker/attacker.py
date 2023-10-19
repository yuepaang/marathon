import numpy as np
from math import *
from copy import deepcopy
from typing import List, Dict, Tuple
from collections import defaultdict


class PolygonAttacker:
    def __init__(self, map_size: Tuple[int, int], agents: List[int],
                 opponents: List[int], walls: List[Tuple]):
        self.map_size = map_size
        self.agents = agents
        self.opponents = opponents
        self.walls = set(walls)
        self.obstacles = list()
        self._set_obstacles(walls)

        self.ROLE = "ATTACKER"
        self.STAY = "STAY"
        self.UP = "UP"
        self.DOWN = "DOWN"
        self.LEFT = "LEFT"
        self.RIGHT = "RIGHT"
        self.epsilon = 1e-6
        self.rotation = np.zeros((len(agents) + 1, 4))
        self.score = 0
        self.round = 0
        self.opp_state = defaultdict(dict)
        self.agt_state = defaultdict(dict)
        for id in agents:
            self.agt_state[id]["pos"] = (0, 0)
            self.agt_state[id]["score"] = 0

    def _set_obstacles(self, obstacles):
        for obstacle in obstacles:
            # self.obstacles.append((obstacle[0], obstacle[1], 0.5))
            coors = [
                (obstacle[0] - 0.25, obstacle[1] - 0.25),
                (obstacle[0] + 0.25, obstacle[1] - 0.25),
                (obstacle[0] - 0.25, obstacle[1] + 0.25),
                (obstacle[0] + 0.25, obstacle[1] + 0.25),
                (obstacle[0], obstacle[1] - 0.25),
                (obstacle[0], obstacle[1] + 0.25),
                (obstacle[0] - 0.25, obstacle[1]),
                (obstacle[0] + 0.25, obstacle[1]),
            ]
            for coor in coors:
                self.obstacles.append((coor[0], coor[1], 0.25))

    def _empty_map(self):
        return np.zeros(self.map_size)

    def _dist(self, coor1, coor2):
        return sqrt(((coor1[0] - coor2[0])**2) + ((coor1[1] - coor2[1])**2))

    def _ang(self, coor1, coor2):
        angle = np.arctan2((coor1[1] - coor2[1]), (coor1[0] - coor2[0]))
        if angle < 0:
            angle = 2 * np.pi + angle
        return (angle)

    def _wrap(self, x):
        while x < 0:
            x += 2 * np.pi
        while x > 2 * np.pi:
            x -= 2 * np.pi
        return x

    def update(self, obs: Dict):
        round_score = 0
        for id, view in obs.items():
            round_score += view["self_agent"]["score"]
            self.agt_state[id]["last_pos"] = self.agt_state[id]["pos"]
            self.agt_state[id]["pos"] = (view["self_agent"]["x"],
                                         view["self_agent"]["y"])

            for other_agent in view["other_agents"]:
                if other_agent["role"] != self.ROLE:
                    pos = (other_agent["x"], other_agent["y"])
                    self.opp_state[other_agent["id"]]["pos"] = pos
                    self.opp_state[
                        other_agent["id"]]["score"] = other_agent["score"]

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

        self.score = round_score

    def _x_action(self, x, id):
        action = self.STAY
        next_pos = self.agt_state[id]["pos"]
        if x > 0:
            action = self.RIGHT
            next_pos = (self.agt_state[id]["pos"][0] + 1,
                        self.agt_state[id]["pos"][1])
        else:
            action = self.LEFT
            next_pos = (self.agt_state[id]["pos"][0] - 1,
                        self.agt_state[id]["pos"][1])
        return action, next_pos

    def _y_action(self, y, id):
        action = self.STAY
        next_pos = self.agt_state[id]["pos"]
        if y > 0:
            action = self.DOWN
            next_pos = (self.agt_state[id]["pos"][0],
                        self.agt_state[id]["pos"][1] + 1)
        else:
            action = self.UP
            next_pos = (self.agt_state[id]["pos"][0],
                        self.agt_state[id]["pos"][1] - 1)
        return action, next_pos

    def _pdca(self, ids: List[int]):
        actions = dict()
        num_play = len(ids)
        points = list()
        for id in ids[:-1]:
            points.append(list(self.agt_state[id]["pos"]))
        points.append(list(self.opp_state[ids[-1]]["pos"]))

        pos = 0
        bottom = points[0]
        for i in range(1, num_play):
            if points[i][1] <= bottom[1]:
                bottom = points[i]
                pos = i

        neigh = list()
        for i in range(0, num_play):
            if i != pos:
                neigh.append([self._ang(bottom, points[i]), i])
        else:
            pass

        neigh.append([0, pos])  #soring angles to form fig
        neigh.sort()
        area = 0
        peri = 0

        for i in range(num_play):
            if i != num_play - 1:
                peri += dist(points[neigh[i][1]], points[neigh[i + 1][1]])
                area += points[neigh[i][1]][0] * points[neigh[i + 1][1]][
                    1] - points[neigh[i + 1][1]][0] * points[neigh[i][1]][1]
                peri += dist(points[neigh[i][1]], points[neigh[i + 1][1]])
            else:
                peri += dist(points[neigh[i][1]], points[neigh[0][1]])
                area += points[neigh[i][1]][0] * points[neigh[0][1]][
                    1] - points[neigh[0][1]][0] * points[neigh[i][1]][1]
                peri += dist(points[neigh[i][1]], points[neigh[0][1]])

        area = abs(area) / 2
        area = max(area, 1e-6)
        peri = max(peri, 1e-6)

        centx = 0
        centy = 0
        for i in range(num_play):
            centx += points[i][0]
            centy += points[i][1]
        centx /= num_play
        centy /= num_play

        for i in range(num_play):
            if neigh[i][1] == num_play - 1:
                phi = 0
                pass
            else:
                ##################    PERI*(K*DIST+(1-K)*CENT)/AREA ALGORITHM    ################
                k = 0.5
                d = dist(points[neigh[i][1]], points[num_play - 1])
                d = max(d, 1e-6)
                dc = dist([centx, centy], points[num_play - 1])
                dc = max(dc, 1e-6)
                f = k * d + (1 - k) * dc
                if i != 0 and i != num_play - 1:
                    dn = dist(points[neigh[i][1]], points[neigh[i + 1][1]])
                    dp = dist(points[neigh[i][1]], points[neigh[i - 1][1]])
                    dn = max(dn, 1e-6)
                    dp = max(dp, 1e-6)
                    nu = (k *
                          (points[neigh[i][1]][1] - points[num_play - 1][1]) /
                          d + (1 - k) * (centy - points[num_play - 1][1]) /
                          (num_play * dc)) / f + (
                              (points[neigh[i][1]][1] -
                               points[neigh[i + 1][1]][1]) / dn +
                              (points[neigh[i][1]][1] -
                               points[neigh[i - 1][1]][1]) / dp) / (peri) + (
                                   points[neigh[i + 1][1]][0] -
                                   points[neigh[i - 1][1]][0]) / (2 * area)
                    de = (k *
                          (points[neigh[i][1]][0] - points[num_play - 1][0]) /
                          d + (1 - k) * (centx - points[num_play - 1][0]) /
                          (num_play * dc)) / f + (
                              (points[neigh[i][1]][0] -
                               points[neigh[i + 1][1]][0]) / dn +
                              (points[neigh[i][1]][0] -
                               points[neigh[i - 1][1]][0]) / dp) / (peri) - (
                                   points[neigh[i + 1][1]][1] -
                                   points[neigh[i - 1][1]][1]) / (2 * area)
                    beta = atan2(nu, de)
                    #peri2 = peri-dist(points[neigh[i][1]],points[neigh[i+1][1]])-dist(points[neigh[i][1]],points[neigh[i-1][1]])
                    next = neigh[i + 1][1]
                    prev = neigh[i - 1][1]
                elif i == num_play - 1:
                    dn = dist(points[neigh[i][1]], points[neigh[0][1]])
                    dp = dist(points[neigh[i][1]], points[neigh[i - 1][1]])
                    dn = max(dn, 1e-6)
                    dp = max(dp, 1e-6)
                    nu = (k *
                          (points[neigh[i][1]][1] - points[num_play - 1][1]) /
                          d + (1 - k) * (centy - points[num_play - 1][1]) /
                          (num_play * dc)) / f + (
                              (points[neigh[i][1]][1] - points[neigh[0][1]][1])
                              / dn +
                              (points[neigh[i][1]][1] -
                               points[neigh[i - 1][1]][1]) / dp) / (peri) + (
                                   points[neigh[0][1]][0] -
                                   points[neigh[i - 1][1]][0]) / (2 * area)
                    de = (k *
                          (points[neigh[i][1]][0] - points[num_play - 1][0]) /
                          d + (1 - k) * (centx - points[num_play - 1][0]) /
                          (num_play * dc)) / f + (
                              (points[neigh[i][1]][0] - points[neigh[0][1]][0])
                              / dn +
                              (points[neigh[i][1]][0] -
                               points[neigh[i - 1][1]][0]) / dp) / (peri) - (
                                   points[neigh[0][1]][1] -
                                   points[neigh[i - 1][1]][1]) / (2 * area)
                    beta = atan2(nu, de)
                    #peri2 = peri-dist(points[neigh[i][1]],points[neigh[0][1]])-dist(points[neigh[i][1]],points[neigh[i-1][1]])
                    next = neigh[0][1]
                    prev = neigh[i - 1][1]
                elif i == 0:
                    dn = dist(points[neigh[i][1]], points[neigh[i + 1][1]])
                    dp = dist(points[neigh[i][1]],
                              points[neigh[num_play - 1][1]])
                    dn = max(dn, 1e-6)
                    dp = max(dp, 1e-6)
                    nu = (k *
                          (points[neigh[i][1]][1] - points[num_play - 1][1]) /
                          d + (1 - k) * (centy - points[num_play - 1][1]) /
                          (num_play * dc)) / f + (
                              (points[neigh[i][1]][1] -
                               points[neigh[i + 1][1]][1]) / dn +
                              (points[neigh[i][1]][1] -
                               points[neigh[num_play - 1][1]][1]) / dp
                          ) / (peri) + (points[neigh[i + 1][1]][0] - points[
                              neigh[num_play - 1][1]][0]) / (2 * area)
                    de = (k *
                          (points[neigh[i][1]][0] - points[num_play - 1][0]) /
                          d + (1 - k) * (centx - points[num_play - 1][0]) /
                          (num_play * dc)) / f + (
                              (points[neigh[i][1]][0] -
                               points[neigh[i + 1][1]][0]) / dn +
                              (points[neigh[i][1]][0] -
                               points[neigh[num_play - 1][1]][0]) / dp
                          ) / (peri) - (points[neigh[i + 1][1]][1] - points[
                              neigh[num_play - 1][1]][1]) / (2 * area)
                    beta = atan2(nu, de)
                    #peri2 = peri-dist(points[neigh[i][1]],points[neigh[i+1][1]])-dist(points[neigh[i][1]],points[neigh[num_play-1][1]])
                    next = neigh[i + 1][1]
                    prev = neigh[num_play - 1][1]

                if beta < 0:
                    beta = 2 * np.pi + beta

                if (points[neigh[i][1]][0] <= 0 and -cos(beta) < 0) or (
                        points[neigh[i][1]][0] >= self.map_size[0] - 1
                        and -cos(beta) > 0):
                    if -sin(beta) > 0:
                        beta = -np.pi / 2 - self.epsilon
                    else:
                        beta = np.pi / 2 + self.epsilon
                if (points[neigh[i][1]][1] <= 0 and -sin(beta) < 0) or (
                        points[neigh[i][1]][1] >= self.map_size[1] - 1
                        and -sin(beta) > 0):
                    if -cos(beta) > 0:
                        beta = np.pi - self.epsilon
                    else:
                        beta = 0 + self.epsilon

                flag = 0

                for j in range(len(self.obstacles)):
                    obst_ang = self._ang(
                        [self.obstacles[j][0], self.obstacles[j][1]],
                        points[neigh[i][1]])
                    obst_ang = self._wrap(obst_ang)
                    if dist(points[neigh[i][1]], [
                            self.obstacles[j][0] - self.obstacles[j][2] *
                            cos(obst_ang), self.obstacles[j][1] -
                            self.obstacles[j][2] * sin(obst_ang)
                    ]) < 0.65 and dist([
                            points[neigh[i][1]][0] - cos(beta),
                            points[neigh[i][1]][1] - sin(beta)
                    ], [
                            self.obstacles[j][0] - self.obstacles[j][2] *
                            cos(obst_ang), self.obstacles[j][1] -
                            self.obstacles[j][2] * sin(obst_ang)
                    ]) < 0.65:
                        c = 0
                        d = 0
                        k = 0
                        flag += 1
                        print("Touch", self.obstacles[j], ids[neigh[i][1]])
                        if self.round == 0 or self.rotation[i][0] == 0:
                            if abs(obst_ang) < np.pi / 2:
                                # print("YES")
                                x1 = points[neigh[i][1]][0] - cos(
                                    (3 * np.pi / 2) + obst_ang)
                                y1 = points[neigh[i][1]][1] - sin(
                                    (3 * np.pi / 2) + obst_ang)
                                x2 = points[neigh[i][1]][0] - cos((np.pi / 2) +
                                                                  obst_ang)
                                y2 = points[neigh[i][1]][1] - sin((np.pi / 2) +
                                                                  obst_ang)
                                if dist([x1, y1], points[num_play - 1]) < dist(
                                    [x2, y2], points[num_play - 1]):
                                    c += -cos(3 * np.pi / 2 + obst_ang)
                                    d += -sin(3 * np.pi / 2 + obst_ang)
                                    self.rotation[i][0] = 1
                                else:
                                    c += -cos(np.pi / 2 + obst_ang)
                                    d += -sin(np.pi / 2 + obst_ang)
                                    self.rotation[i][0] = 2
                            else:
                                x1 = points[neigh[i][1]][0] - cos(
                                    (-np.pi / 2) + obst_ang)
                                y1 = points[neigh[i][1]][1] - sin(
                                    (-np.pi / 2) + obst_ang)
                                x2 = points[neigh[i][1]][0] - cos((np.pi / 2) +
                                                                  obst_ang)
                                y2 = points[neigh[i][1]][1] - sin((np.pi / 2) +
                                                                  obst_ang)
                                if dist([x1, y1], points[num_play - 1]) - dist(
                                    [x2, y2], points[num_play - 1]) < 0:
                                    c += -cos(obst_ang - np.pi / 2)
                                    d += -sin(obst_ang - np.pi / 2)
                                    self.rotation[i][0] = 1
                                else:
                                    c += -cos(obst_ang + np.pi / 2)
                                    d += -sin(obst_ang + np.pi / 2)
                                    self.rotation[i][0] = 2
                        elif self.rotation[i][0] == 1:
                            c += -cos(obst_ang - np.pi / 2)
                            d += -sin(obst_ang - np.pi / 2)
                        elif self.rotation[i][0] == 2:
                            c += -cos(obst_ang + np.pi / 2)
                            d += -sin(obst_ang + np.pi / 2)
                        self.rotation[i][1] = neigh[i][1]
                        self.rotation[i][2] = c
                        self.rotation[i][3] = d

                if flag == 0 and dist(points[neigh[i][1]], [
                        self.obstacles[j][0] - self.obstacles[j][2] *
                        cos(obst_ang), self.obstacles[j][1] -
                        self.obstacles[j][2] * sin(obst_ang)
                ]) > 2 and dist([
                        points[neigh[i][1]][0] - cos(beta),
                        points[neigh[i][1]][1] - sin(beta)
                ], [
                        self.obstacles[j][0] - self.obstacles[j][2] *
                        cos(obst_ang), self.obstacles[j][1] -
                        self.obstacles[j][2] * sin(obst_ang)
                ]) > 2:
                    self.rotation[i][0] = 0
                    self.rotation[i][1] = neigh[i][1]
                    self.rotation[i][2] = 10
                    self.rotation[i][3] = 10

                if self.rotation[i][0] == 1 or self.rotation[i][0] == 2:
                    #print("CAC",c,d)
                    points[neigh[i][1]][0] += self.rotation[i][2] / sqrt(
                        self.rotation[i][2]**2 + self.rotation[i][3]**2)
                    points[neigh[i][1]][1] += self.rotation[i][3] / sqrt(
                        self.rotation[i][2]**2 + self.rotation[i][3]**2)
                else:
                    points[neigh[i][1]][0] -= cos(beta)
                    points[neigh[i][1]][1] -= sin(beta)

                agent_id = ids[neigh[i][1]]
                last_pos = self.agt_state[agent_id]["pos"]
                delta_x = points[neigh[i][1]][0] - last_pos[0]
                delta_y = points[neigh[i][1]][1] - last_pos[1]
                # print("move", agent_id, last_pos, points[neigh[i][1]], delta_x,
                #       delta_y)
                if abs(delta_x) <= 1e-3 and abs(delta_y) <= 1e-3:
                    action = "STAY"
                    next_pos = self.agt_state[agent_id]["pos"]
                elif abs(delta_x) > abs(delta_y):
                    action, next_pos = self._x_action(delta_x, agent_id)
                    # print("delta_x next_pos", next_pos, next_pos in self.walls)
                    if (next_pos in self.walls
                            or not (0 <= next_pos[0] <= self.map_size[0] - 1)
                            or not (0 <= next_pos[1] <= self.map_size[1] - 1)):
                        action, next_pos = self._y_action(delta_y, agent_id)
                else:
                    action, next_pos = self._y_action(delta_y, agent_id)
                    # print("_y_action next_pos", next_pos,
                    #       next_pos in self.walls)
                    if (next_pos in self.walls
                            or not (0 <= next_pos[0] <= self.map_size[0] - 1)
                            or not (0 <= next_pos[1] <= self.map_size[1] - 1)):
                        action, next_pos = self._x_action(delta_x, agent_id)

                # print(next_pos, action)
                # print()
                actions[agent_id] = action

        return actions, (centx, centy)

    def step(self):
        max_score = -1
        max_action = dict()
        target = None
        for opp in self.opponents:
            ids = self.agents + [opp]
            action, center = self._pdca(ids)
            d = self._dist(self.opp_state[opp]["pos"], center) + 1e-6
            score = (self.opp_state[opp]["score"] / 2 + 4) / log(d)
            if score > max_score:
                max_score = score
                max_action = deepcopy(action)
                target = opp

        print("=====>", self.round, self.score)
        print("\ttarget:", target, "max_score:", max_score, "action:",
              max_action)

        self.round += 1
        return max_action
