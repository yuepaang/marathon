import random
import numpy as np
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
        #预测每个对手在地图每个格子的概率
        self.pos_score = np.zeros(
            shape=[len(opponents), map.size[0], map.size[1]])
        # 探索视野能获得的得分 - 视野内的格子X回合内无得分
        self.vision_score = np.zeros(shape=map.size)
        self.map = map
        self.round = 0
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

    def explore(self):
        '''
        探索每个方向并扩展模拟回合进行直到N个回合, 汇总每个扩展分支的得分
        '''
        return

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
        1. 更新自身位置
        2. 更新对手位置概率
        3. 更新视野得分
        '''
        opp_in_view = dict()  # 视野内的对手
        view_pos = list()  # 视野内格子坐标
        for id, view in obs.items():
            # 更新自身位置
            self.my_last_pos[id] = self.my_pos[id]
            my_pos = self.map.obs_to_map_coor(
                (view["self_agent"]["x"], view["self_agent"]["y"]))
            self.my_pos[id] = my_pos

            for other_agent in view["other_agents"]:
                if other_agent["role"] != self.role:
                    map_pos = self.map.obs_to_map_coor(
                        (other_agent["x"], other_agent["y"]))
                    opp_in_view[other_agent["id"]]: map_pos

            view_pos.extend(
                self.map.get_view_coor(my_pos,
                                       view["self_agent"]["vision_range"]))

        view_pos = list(set(view_pos))
        rows, cols = zip(*view_pos)

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
