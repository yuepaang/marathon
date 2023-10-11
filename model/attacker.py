import time
import numpy as np
import rust_perf
from model.model import Model
from map.map import Map


class Attacker(Model):
    def __init__(self, agents: list, opponents: list, map: Map):
        super().__init__(agents, opponents, map)
        self.role = "ATTACKER"

    def step(self):
        """
        主模型逻辑
        1. 地图每个格子各种分数相加 -- 攻方对手位置概率得分为负
        2. 探索每种action组合并扩展模拟回合进行直到N个回合, 汇总每个扩展分支的得分到每个action组合
        3. 选择得分最高的方向并返回: 如["UP", "LEFT", "UP", "RIGHT"]
        """
        self.round += 1
        self.score_sum()
        exploration = {"reward": -1000, "trace": {}}
        visited = dict()
        for id, pos in self.my_pos.items():
            visited[id] = set([pos])
        next_pos = self.get_all_nearby_pos(self.my_pos, visited)

        # for pos in next_pos:
        #     self.explore(pos, self.map_score, pos, 0, exploration, visited, 1, 6)

        start = time.time()
        rewards = rust_perf.explore(
            self.agents,
            next_pos,
            self.map_score.tolist(),
            visited,
            4,
        )
        print("duration: ", time.time() - start)
        # print(rewards)
        # print(len(rewards))
        raise Exception("e")
        for i, reward in enumerate(rewards):
            if reward > exploration["reward"]:
                exploration["reward"] = reward
                exploration["trace"] = next_pos[i]
        ############# py ##############

        action = dict()
        for id, pos in exploration["trace"].items():
            action[id] = self.map.to_action(self.my_pos[id], pos)
        return action

    def score_sum(self):
        """
        地图分数相加
        """
        self.map_score = np.zeros(shape=self.map.size)
        # 金币
        self.map_score += self.map.coin_map * 2
        # # 敌人位置
        # opp_score = self.pos_score / self.pos_score.sum(axis=(1, 2),
        #                                                 keepdims=True)
        # self.map_score -= opp_score.sum(axis=0) * 2
        # 道具
        self.map_score += self.power_up_score
