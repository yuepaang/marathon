import numpy as np
from model.model import Model
from map.map import Map


class Attacker(Model):
    # def step(self):
    #     '''
    #     主模型逻辑
    #     1. 地图每个格子各种分数相加 -- 攻方对手位置概率得分为正
    #     2. 探索每个方向并扩展模拟回合进行直到N个回合, 汇总每个扩展分支的得分
    #     3. 选择得分最高的方向并返回: 如["UP", "LEFT", "UP", "RIGHT"]
    #     '''
    #     pass
    
    def score_sum(self):
        '''
        地图分数相加
        '''
        pass
