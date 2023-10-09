import random
import numpy as np
from copy import deepcopy
from map.map import Map


class Model:
    def __init__(self, agents: list, opponents: list, map: Map):
        '''
        agents: id [0, 1, 2, 3]
        opponents: id [4, 5, 6, 7]
        map: map.map.Map
        '''
        self.agents = agents
        self.opponents = opponents
        self.role = ""
        self.ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
        # 自己位置
        self.my_pos = dict()
        for id in agents:
            self.my_pos[id] = (0, 0)
        self.my_last_pos = dict()
        for id in agents:
            self.my_last_pos[id] = (0, 0)
        # 敌人位置分数
        self.pos_score = np.zeros(
            shape=[len(opponents), map.size[0], map.size[1]])
        # 探索视野能获得的得分 - 视野内的格子X回合内无得分
        self.vision_score = np.zeros(shape=map.size)
        # 道具分数
        self.power_up_score = np.zeros(shape=map.size)
        self.map_score = np.zeros(shape=map.size)
        self.map = map
        self.round = 0
        self.total_score = 0
        return

    def step(self):
        '''
        返回每个agent的动作: 如["UP", "LEFT", "UP", "RIGHT"]
        '''
        self.round += 1
        return {_id: random.choice(self.ACTIONS) for _id in self.agents}

    def score_sum(self):
        '''
        地图分数相加
        '''
        return

    def get_all_nearby_pos(self, agent_pos: dict, visited: dict):
        '''
        agent_pos: {id1: (x1, y1), id2: (x2, y2)}
        '''
        delta = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        new_pos = dict()
        for id in self.agents:
            new_pos[id] = list()
            for d in delta:
                pos_tmp = (agent_pos[id][0] + d[0], agent_pos[id][1] + d[1])
                if (not (0 <= pos_tmp[0] < self.map.size[0])
                        or not (0 <= pos_tmp[1] < self.map.size[1])
                        or self.map.wall_map[pos_tmp] == 1
                        or pos_tmp in visited[id]):
                    continue
                new_pos[id].append(pos_tmp)

        next_pos = list()
        for p0 in new_pos[self.agents[0]]:
            for p1 in new_pos[self.agents[1]]:
                for p2 in new_pos[self.agents[2]]:
                    for p3 in new_pos[self.agents[3]]:
                        next_pos.append({
                            self.agents[0]: p0,
                            self.agents[1]: p1,
                            self.agents[2]: p2,
                            self.agents[3]: p3,
                        })

        return next_pos

    def explore(self, position: dict, map_score: np.ndarray, trace: dict,
                reward: float, exploration: dict, visited: dict, step: int,
                max_step: int):
        '''
        探索每个方向并扩展模拟回合进行直到N个回合, 汇总每个扩展分支的得分
        position: {id1: (x1, y1), id2: (x2, y2)}
        map_score: self.map_score 
        trace: position of first step
        reward: total reward of chosen route
        exploration: {"reward": max_reward, "trace": the trace of max_reward}
        step: 
        '''
        if step >= max_step:
            return

        map_copy = np.copy(map_score)
        visited_copy = dict()
        for id, v in visited.items():
            visited_copy[id] = set(v)
        # visited_copy = deepcopy(visited)

        pos = list()
        crowd = 0.0
        for id in self.agents:
            agent_pos = (position[id][0], position[id][1])
            visited_copy[id].add(agent_pos)
            reward += map_copy[agent_pos] * pow(0.99, step)
            map_copy[agent_pos] = 0

            for other in pos:
                if self.map.distance(other, agent_pos) < 3:
                    crowd += 0.1

        reward -= crowd

        if reward > exploration["reward"]:
            exploration["reward"] = reward
            exploration["trace"] = trace

        next_pos = self.get_all_nearby_pos(position, visited_copy)
        for p in next_pos:
            self.explore(p, map_copy, trace, reward, exploration, visited_copy,
                         step + 1, max_step)

    def certain_pos(self, id: int, pos: tuple):
        '''
        对手的确定位置
        '''
        index = self.opponents.index(id)
        self.pos_score[index] = 0
        self.pos_score[index, pos[0], pos[1]] = 1
        return

    def estimate_pos(self, id: int):
        '''
        看不到对手时估计对手位置, 每一步污染相邻的一格
        '''
        index = self.opponents.index(id)
        last_pos = np.transpose(self.pos_score[index].nonzero())
        pos = list()
        for p in last_pos:
            nearby = self.map.get_nearby_coor(p)
            pos.extend(nearby)
        rows, cols = zip(*pos)
        self.pos_score[index][rows, cols] = 1
        return

    def update(self, obs: dict):
        '''
        1. 更新分数
        2. 更新自身位置
        3. 更新对手位置概率
        4. 更新视野得分
        '''
        ### 1. 更新分数/自身位置
        opp_in_view = dict()  # 视野内的对手
        view_pos = list()  # 视野内格子坐标
        total_score = 0
        for id, view in obs.items():
            # 更新自身位置
            self.my_last_pos[id] = self.my_pos[id]
            my_pos = self.map.obs_to_map_coor(
                (view["self_agent"]["x"], view["self_agent"]["y"]))
            self.my_pos[id] = my_pos
            total_score += view["self_agent"]["score"]

            for other_agent in view["other_agents"]:
                if other_agent["role"] != self.role:
                    map_pos = self.map.obs_to_map_coor(
                        (other_agent["x"], other_agent["y"]))
                    opp_in_view[other_agent["id"]]: map_pos

            view_pos.extend(
                self.map.get_view_coor(my_pos,
                                       view["self_agent"]["vision_range"]))

        self.total_score = total_score
        view_pos = list(set(view_pos))
        rows, cols = zip(*view_pos)

        ### 2. 更新对手位置概率
        if self.round == 0:
            # 第一轮对手在镜像位置
            for index in range(len(self.opponents)):
                opp_pos = (self.map.size[0] - 1 -
                           self.my_pos[self.agents[index]][0],
                           self.map.size[1] - 1 -
                           self.my_pos[self.agents[index]][1])
                self.certain_pos(self.opponents[index], opp_pos)
        else:
            # 根据视野更新对手位置或估算对手位置
            for index in range(len(self.opponents)):
                id = self.opponents[index]
                if id in opp_in_view:
                    self.certain_pos(id, opp_in_view[id])
                else:
                    self.estimate_pos(id)
                    self.pos_score[index][rows, cols] = 0

        return
