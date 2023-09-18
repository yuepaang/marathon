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
from ding.envs import BaseEnv
from ding.utils import ENV_REGISTRY, deep_merge_dicts
from easydict import EasyDict
import numpy as np

from game import Game

ORIGINAL_AGENT = "attacker"
OPPONENT_AGENT = "defender"

FORCE_RESTART_INTERVAL = 50000


class Direction(enum.IntEnum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3


@ENV_REGISTRY.register("marathon")
class Marathon(Game, BaseEnv):
    MarathonTimestep = namedtuple(
        "MarathonTimestep", ["obs", "reward", "done", "info", "episode_steps"]
    )
    config = dict(
        two_player=False,
        mirror_opponent=False,
        reward_type="original",
        save_replay_episodes=None,
        difficulty=7,
        reward_death_value=10,
        reward_win=200,
        obs_alone=False,
        game_steps_per_episode=None,
        reward_only_positive=True,
        death_mask=False,
        special_global_state=False,
        # add map's center location point or not
        add_center_xy=True,
        independent_obs=False,
        # add agent's id information or not in special global state
        state_agent_id=True,
    )

    def __init__(self, cfg: dict, map_data: dict) -> None:
        super().__init__(map_data)
        cfg = deep_merge_dicts(EasyDict(self.config), cfg)
        self.cfg = cfg

        self._episode_steps = 0
        self._seed = None
        self._launch_env_flag = True
        self.agents = {}
        self.enemies = {}
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self._next_reset_steps = FORCE_RESTART_INTERVAL

    def seed(self, seed, dynamic_seed=False):
        self._seed = seed

    def _launch(self):
        """Generate Game Map"""
        self.reset_game("attacker", "defender", self.seed)
        self._launch_env_flag = True

    def reset(self):
        self._final_eval_fake_reward = 0.0
        old_unit_tags = set(u.tag for u in self.agents.values()).union(
            set(u.tag for u in self.enemies.values())
        )

        if self.just_force_restarts:
            old_unit_tags = set()
            self.just_force_restarts = False

        if self._launch_env_flag:
            # Launch StarCraft II
            print("*************LAUNCH MARATHON GAME********************")
            self._launch()
            self._launch_env_flag = False
        elif (self._total_steps > self._next_reset_steps) or (
            self.save_replay_episodes is not None
        ):
            # Avoid hitting the real episode limit of SC2 env
            print(
                "We are full restarting the environment! save_replay_episodes: ",
                self.save_replay_episodes,
            )
            self.full_restart()
            old_unit_tags = set()
            self._next_reset_steps += FORCE_RESTART_INTERVAL
        else:
            self._restart_episode()

        # Information kept for counting the reward
        self.win_counted = False
        self.defeat_counted = False

        self.action_helper.reset()

        self.previous_ally_units = None
        self.previous_enemy_units = None

        # if self.heuristic_ai:
        #     self.heuristic_targets = [None] * self.n_agents

        count = 0
        while count <= 5:
            self._update_obs()
            # print("INTERNAL INIT UNIT BEGIN")
            init_flag = self.init_units(old_unit_tags)
            # print("INTERNAL INIT UNIT OVER", init_flag)
            count += 1
            if init_flag:
                break
            else:
                old_unit_tags = set()
        if count >= 5:
            raise RuntimeError("reset 5 times error")

        self.reward_helper.reset(self.max_reward)

        assert all(u.health > 0 for u in self.agents.values())
        assert all(u.health > 0 for u in self.enemies.values())

        return {
            "agent_state": {
                ORIGINAL_AGENT: self.get_obs(),
                OPPONENT_AGENT: self.get_obs(True),
            },
            "global_state": {
                ORIGINAL_AGENT: self.get_state(),
                OPPONENT_AGENT: self.get_state(True),
            },
            "action_mask": {
                ORIGINAL_AGENT: self.get_avail_actions(),
                OPPONENT_AGENT: self.get_avail_actions(True),
            },
        }

    def get_obs(self, is_opponent=False):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralized execution.
        """
        agents_obs_list = [
            self.get_obs_agent(i, is_opponent) for i in range(self.n_agents)
        ]

        if self.mirror_opponent and is_opponent:
            assert not self.flatten_observation
            new_obs = list()
            for agent_obs in agents_obs_list:
                new_agent_obs = dict()
                for key, feat in agent_obs.items():
                    feat = feat.copy()

                    if key == "move_feats":
                        can_move_right = feat[2]
                        can_move_left = feat[3]
                        feat[3] = can_move_right
                        feat[2] = can_move_left

                    elif key == "enemy_feats" or key == "ally_feats":
                        for unit_id in range(feat.shape[0]):
                            # Relative x
                            feat[unit_id, 2] = -feat[unit_id, 2]

                    new_agent_obs[key] = feat
                new_obs.append(new_agent_obs)
            agents_obs_list = new_obs

        if not self.flatten_observation:
            agents_obs_list = self._flatten_obs(agents_obs_list)
        if self.obs_alone:
            (
                agents_obs_list,
                agents_obs_alone_list,
                agents_obs_alone_padding_list,
            ) = list(zip(*agents_obs_list))
            return (
                np.array(agents_obs_list).astype(np.float32),
                np.array(agents_obs_alone_list).astype(np.float32),
                np.array(agents_obs_alone_padding_list).astype(np.float32),
            )
        else:
            return np.array(agents_obs_list).astype(np.float32)

    def step(self, actions):
        self.apply_actions(actions["attacker"], actions["defender"])

    def get_unit_by_id(self, a_id, is_opponent=False):
        """Get unit by ID."""
        if is_opponent:
            return self.enemies[a_id]
        return self.agents[a_id]

    def get_obs_agent(self, agent_id, is_opponent=False):
        unit = self.get_unit_by_id(agent_id, is_opponent=is_opponent)

        # TODO All these function should have an opponent version
        enemy_feats_dim = self.get_obs_enemy_feats_size()
        ally_feats_dim = self.get_obs_ally_feats_size()
        own_feats_dim = self.get_obs_own_feats_size()

        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)

        move_feats = self.action_helper.get_movement_features(
            agent_id, self, is_opponent
        )

        if unit.health > 0:  # otherwise dead, return all zeros
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)
            avail_actions = self.action_helper.get_avail_agent_actions(
                agent_id, self, is_opponent
            )

            # Enemy features
            if is_opponent:
                enemy_items = self.agents.items()
            else:
                enemy_items = self.enemies.items()
            for e_id, e_unit in enemy_items:
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = distance(x, y, e_x, e_y)

                if dist < sight_range and e_unit.health > 0:  # visible and alive
                    # Sight range > shoot range
                    enemy_feats[e_id, 0] = avail_actions[
                        self.action_helper.n_actions_no_attack + e_id
                    ]  # available
                    enemy_feats[e_id, 1] = dist / sight_range  # distance
                    enemy_feats[e_id, 2] = (e_x - x) / sight_range  # relative X
                    enemy_feats[e_id, 3] = (e_y - y) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        enemy_feats[e_id, ind] = (
                            e_unit.health / e_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_enemy > 0:
                            max_shield = self.unit_max_shield(e_unit, not is_opponent)
                            enemy_feats[e_id, ind] = (
                                e_unit.shield / max_shield
                            )  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        # If enemy is computer, than use ally=False, but since now we use
                        #  agent for enemy, ally=True
                        if self.two_player:
                            type_id = self.get_unit_type_id(
                                e_unit, True, not is_opponent
                            )
                        else:
                            type_id = self.get_unit_type_id(e_unit, False, False)
                        enemy_feats[e_id, ind + type_id] = 1  # unit type

            # Ally features
            al_ids = [
                al_id
                for al_id in range(
                    (self.n_agents if not is_opponent else self.n_enemies)
                )
                if al_id != agent_id
            ]
            for i, al_id in enumerate(al_ids):
                al_unit = self.get_unit_by_id(al_id, is_opponent=is_opponent)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = distance(x, y, al_x, al_y)

                if dist < sight_range and al_unit.health > 0:  # visible and alive
                    ally_feats[i, 0] = 1  # visible
                    ally_feats[i, 1] = dist / sight_range  # distance
                    ally_feats[i, 2] = (al_x - x) / sight_range  # relative X
                    ally_feats[i, 3] = (al_y - y) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        ally_feats[i, ind] = (
                            al_unit.health / al_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit, is_opponent)
                            ally_feats[i, ind] = al_unit.shield / max_shield  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(al_unit, True, is_opponent)
                        ally_feats[i, ind + type_id] = 1
                        ind += self.unit_type_bits

                    # LJ fix
                    # if self.obs_last_action:
                    #     ally_feats[i, ind:] = self.action_helper.get_last_action(is_opponent)[al_id]

            # Own features
            ind = 0
            if self.obs_own_health:
                own_feats[ind] = unit.health / unit.health_max
                ind += 1
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(unit, is_opponent)
                    own_feats[ind] = unit.shield / max_shield
                    ind += 1

            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(unit, True, is_opponent)
                own_feats[ind + type_id] = 1
                ind += self.unit_type_bits
            if self.obs_last_action:
                own_feats[ind:] = self.action_helper.get_last_action(is_opponent)[
                    agent_id
                ]

        if is_opponent:
            agent_id_feats = np.zeros(self.n_enemies)
        else:
            agent_id_feats = np.zeros(self.n_agents)
        agent_id_feats[agent_id] = 1
        # Only set to false by outside wrapper
        if self.flatten_observation:
            agent_obs = np.concatenate(
                (
                    move_feats.flatten(),
                    enemy_feats.flatten(),
                    ally_feats.flatten(),
                    own_feats.flatten(),
                    agent_id_feats,
                )
            )
            if self.obs_timestep_number:
                agent_obs = np.append(
                    agent_obs, self._episode_steps / self.episode_limit
                )
            if self.obs_alone:
                agent_obs_alone = np.concatenate(
                    (
                        move_feats.flatten(),
                        enemy_feats.flatten(),
                        own_feats.flatten(),
                        agent_id_feats,
                    )
                )
                agent_obs_alone_padding = np.concatenate(
                    (
                        move_feats.flatten(),
                        enemy_feats.flatten(),
                        np.zeros_like(ally_feats.flatten()),
                        own_feats.flatten(),
                        agent_id_feats,
                    )
                )
                if self.obs_timestep_number:
                    agent_obs_alone = np.append(
                        agent_obs_alone, self._episode_steps / self.episode_limit
                    )
                    agent_obs_alone_padding = np.append(
                        agent_obs_alone_padding,
                        self._episode_steps / self.episode_limit,
                    )
                return agent_obs, agent_obs_alone, agent_obs_alone_padding
            else:
                return agent_obs
        else:
            agent_obs = dict(
                move_feats=move_feats,
                enemy_feats=enemy_feats,
                ally_feats=ally_feats,
                own_feats=own_feats,
                agent_id_feats=agent_id_feats,
            )
            if self.obs_timestep_number:
                agent_obs["obs_timestep_number"] = (
                    self._episode_steps / self.episode_limit
                )

        return agent_obs
