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
        self.ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
        #预测每个对手在地图每个格子的概率
        self.pos_score = np.zeros(
            shape=[len(opponents), map.size[0], map.size[1]])
        # 探索视野能获得的得分 - 视野内的格子X回合内无得分
        self.vision_score = np.zeros(shape=map.size)
        self.map = map
        return

    def step(self):
        '''
        返回每个agent的动作: 如["UP", "LEFT", "UP", "RIGHT"]
        '''
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

    def update(self, obs: dict):
        '''
        1. 更新对手位置概率
        2. 更新视野得分
        '''
        return
