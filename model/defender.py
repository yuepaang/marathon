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
        action = dict()
        self.round += 1

        if self.round <= 5:
            return {4: "UP", 5: "UP", 6: "RIGHT", 7: "RIGHT"}

        self.score_sum()

        visited = dict()
        groups = list()
        for id, pos in self.my_pos.items():
            visited[id] = set([pos])

            # 相距远的agent单独决策
            isolate = True
            for i in range(len(groups)):
                near = False
                for other in groups[i]:
                    if self.map.distance(pos, self.my_pos[other]) <= 4:
                        near = True
                        break
                if near:
                    groups[i].append(id)
                    isolate = False
                    break
            if isolate:
                groups.append([id])

        for group in groups:
            exploration = {"reward": -1000, "trace": {}}
            agent_pos = dict()
            for id in group:
                agent_pos[id] = self.my_pos[id]
            next_pos = self.get_all_nearby_pos(agent_pos, visited)
            for pos in next_pos:
                self.explore(pos, self.map_score, pos, 0, exploration, visited,
                             1, 10)
            for id, pos in exploration["trace"].items():
                action[id] = self.map.to_action(self.my_pos[id], pos)

        return action

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
