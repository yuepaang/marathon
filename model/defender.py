import random
import numpy as np
from model.model import Model
from map.map import Map


class Defender(Model):
    def __init__(self, agents: list, opponents: list, map: Map):
        super().__init__(agents, opponents, map)
        self.role = "DEFENDER"

    def step(self):
        '''
        主模型逻辑
        1. 地图每个格子各种分数相加 -- 攻方对手位置概率得分为负
        2. 探索每种action组合并扩展模拟回合进行直到N个回合, 汇总每个扩展分支的得分到每个action组合
        3. 选择得分最高的方向并返回: 如["UP", "LEFT", "UP", "RIGHT"]
        '''
        self.round += 1
        self.score_sum()
        print("map_score")
        print(self.map_score.astype(int))
        return {_id: random.choice(self.ACTIONS) for _id in self.agents}

    def score_sum(self):
        '''
        地图分数相加
        '''
        self.map_score = np.zeros(shape=self.map.size)
        # 金币
        self.map_score += self.map.coin_map * 2
        # # 敌人位置
        # opp_score = self.pos_score / self.pos_score.sum(axis=(1, 2),
        #                                                 keepdims=True)
        # self.map_score -= opp_score.sum(axis=0) * 2
        # 道具
        self.map_score += self.power_up_score
