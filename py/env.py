# -*- coding: utf-8 -*-
"""Env for DI-ENGINE.

__author__ = Yue Peng
__copyright__ = Copyright 2023
__version__ = 1.0.0
__maintainer__ = Yue Peng
__email__ = yuepaang@gmail.com
__status__ = Dev
__filename__ = env.py
__uuid__ = ba84bb0f-e226-4c5f-ae50-b4e4c7c103c1
__date__ = 2023-09-18 21:30:14
__modified__ =
"""

from collections import namedtuple
import enum
import random
from easydict import EasyDict
import numpy as np

from game import CellType, Game

ORIGINAL_AGENT = "attacker"
OPPONENT_AGENT = "defender"


class Direction(enum.IntEnum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    STAY = 4


class Marathon(Game):
    def __init__(self, map_data: dict) -> None:
        super().__init__(map_data)
        self._seed = None
        self.map_x = map_data["map_conf"]["width"]
        self.map_y = map_data["map_conf"]["height"]

    def seed(self, seed):
        self._seed = seed

    def _launch(self):
        """Generate Game Map"""
        self.reset_game("attacker", "defender", self.seed)
        self._launch_env_flag = True

    def reset(self) -> np.ndarray:
        self._final_eval_fake_reward = 0.0

        print("*************LAUNCH MARATHON GAME********************")
        self._launch()

        picked_id = None
        picked_agent = None
        for a_id, agent in self.agents.items():
            if agent.role == "DEFENDER":
                picked_agent = agent
                picked_id = a_id
                break
        # attacker_state = self.get_agent_states_by_player("attacker")
        defender_state = self.get_agent_states_by_player("defender")
        # raise Exception("e")

        own_feats_dim = 128

        obs = np.zeros(own_feats_dim, dtype=np.float32)

        obs[0] = picked_agent.pos[0] / self.map_x
        obs[1] = picked_agent.pos[1] / self.map_y
        obs[2] = picked_agent.score
        ind = 3
        state = defender_state[picked_id]
        for wall in state["walls"]:
            obs[ind] = wall["x"]
            ind += 1
            obs[ind] = wall["y"]
            ind += 1

        for coin in state["coins"]:
            obs[ind] = coin["x"]
            ind += 1
            obs[ind] = coin["y"]
            ind += 1

        for oa in state["other_agents"]:
            obs[ind] = oa["score"]
            ind += 1
            obs[ind] = 1 if oa["role"] == "DEFENDER" else 0
            ind += 1
            obs[ind] = oa["x"]
            ind += 1
            obs[ind] = oa["y"]

        return obs, {"a_id": picked_id}

    def step(self, actions):
        attacker_state = self.get_agent_states_by_player("attacker")
        defender_state = self.get_agent_states_by_player("defender")
        ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
        attacker_actions = {
            _id: random.choice(ACTIONS) for _id in attacker_state.keys()
        }

        attacker_score = 0
        for k in attacker_state.keys():
            attacker_score += attacker_state[k]["self_agent"]["score"]

        total_score = 0
        for k in defender_state.keys():
            total_score += defender_state[k]["self_agent"]["score"]
            if k not in actions:
                actions[k] = random.choice(ACTIONS)
        self.apply_actions(attacker_actions, actions)

        picked_id = None
        picked_agent = None
        for a_id, agent in self.agents.items():
            if agent.role == "DEFENDER":
                picked_agent = agent
                picked_id = a_id
                break

        # attacker_state = self.get_agent_states_by_player("attacker")
        defender_state = self.get_agent_states_by_player("defender")

        own_feats_dim = 128

        obs = np.zeros(own_feats_dim, dtype=np.float32)

        obs[0] = picked_agent.pos[0] / self.map_x
        obs[1] = picked_agent.pos[1] / self.map_y
        obs[2] = picked_agent.score
        ind = 3
        state = defender_state[picked_id]
        for wall in state["walls"]:
            obs[ind] = wall["x"]
            ind += 1
            obs[ind] = wall["y"]
            ind += 1
        for oa in state["other_agents"]:
            obs[ind] = oa["score"]
            ind += 1
            obs[ind] = 1 if oa["role"] == "DEFENDER" else 0
            ind += 1
            obs[ind] = oa["x"]
            ind += 1
            obs[ind] = oa["y"]

        return (
            obs,
            attacker_score - total_score,
            self.is_over(),
            self.is_over(),
            False,
        )

    def get_obs(self):
        pass