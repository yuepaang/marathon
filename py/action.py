from collections import namedtuple
import enum

import numpy as np


ORIGINAL_AGENT = "attacker"
OPPONENT_AGENT = "defender"


class Direction(enum.IntEnum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    STAY = 4


def distance(x1, y1, x2, y2):
    """Distance between two points."""
    return abs(x1 - x2) + abs(y1 - y2)


class Action:
    info_template = namedtuple(
        "EnvElementInfo",
        ["shape", "value", "to_agent_processor", "from_agent_processor"],
    )

    def __init__(self, agents, enemies):
        self._move_amount = 1
        self.n_actions_move = 5
        self.n_actions = 5
        self.map_x = 0
        self.map_y = 0

        # Status tracker
        self.agents = agents
        self.enemies = enemies
        self.n_agents = len(agents)
        self.n_enemies = len(enemies)
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self.last_action_opponent = np.zeros((self.n_enemies, self.n_actions))

    def reset(self):
        self.last_action.fill(0)
        self.last_action_opponent.fill(0)

    def update(self, map_x, map_y):
        self.map_x = map_x
        self.map_y = map_y

    def _parse_single(self, action_dict, is_opponent=False):
        action_id = sorted([k for k in action_dict.keys()])
        actions = [action_dict[k] for k in action_id]
        actions = np.asarray(actions, dtype=np.int)
        assert len(actions) == (self.n_enemies if is_opponent else self.n_agents)

        actions_int = [int(a) for a in actions]
        # Make them one-hot
        if is_opponent:
            self.last_action_opponent = np.eye(self.n_actions)[np.array(actions_int)]
        else:
            self.last_action = np.eye(self.n_actions)[np.array(actions_int)]

        sc_actions = []
        for a_id, action in enumerate(actions_int):
            sc_action = self.get_agent_action(a_id, action, is_opponent)
            if sc_action:
                sc_actions.append(sc_action)
        return sc_actions

    def get_action(self, actions):
        """
        {"attacker": {1: "UP}, "defender": {2: "DOWN"}}
        """
        # ========= Two player mode ==========
        assert isinstance(actions, dict)
        assert ORIGINAL_AGENT in actions
        assert OPPONENT_AGENT in actions

        sc_actions_me = self._parse_single(actions[ORIGINAL_AGENT], is_opponent=False)
        sc_actions_opponent = self._parse_single(
            actions[OPPONENT_AGENT], is_opponent=True
        )

        return {ORIGINAL_AGENT: sc_actions_me, OPPONENT_AGENT: sc_actions_opponent}

    def get_unit_by_id(self, a_id, is_opponent=False):
        """Get unit by ID."""
        if is_opponent:
            return self.enemies[a_id]
        return self.agents[a_id]

    def get_agent_action(self, action: int):
        """Construct the action for agent a_id."""

        if action == 0:
            cmd = "UP"
        elif action == 1:
            cmd = "DOWN"
        elif action == 2:
            cmd = "RIGHT"
        elif action == 3:
            cmd = "LEFT"
        elif action == 4:
            cmd = "STAY"

        return cmd

    def get_avail_agent_actions(self, agent_id, is_opponent=False):
        """Returns the available actions for agent_id."""
        avail_actions = [0] * self.n_actions
        unit = self.get_unit_by_id(agent_id, is_opponent)
        # see if we can move
        if self.can_move(unit, Direction.UP):
            avail_actions[0] = 1
        if self.can_move(unit, Direction.DOWN):
            avail_actions[1] = 1
        if self.can_move(unit, Direction.RIGHT):
            avail_actions[2] = 1
        if self.can_move(unit, Direction.LEFT):
            avail_actions[3] = 1

        # if self.can_move(unit, Direction.STAY):
        avail_actions[4] = 1

        return avail_actions

    def can_move(self, unit, direction):
        """Whether a unit can move in a given direction."""
        m = self._move_amount
        ux, uy = unit["self_agent"]["x"], unit["self_agent"]["y"]
        if direction == Direction.UP:
            x, y = ux, uy + m
        elif direction == Direction.DOWN:
            x, y = ux, uy - m
        elif direction == Direction.RIGHT:
            x, y = ux + m, uy
        else:
            x, y = ux - m, uy

        if self.check_bounds(x, y):
            return True

        return False

    def check_bounds(self, x, y):
        """Whether a point is within the map bounds."""
        return 0 <= x < self.map_x and 0 <= y < self.map_y

    def get_movement_features(self, agent_id, is_opponent=False):
        move_feats_dim = self.get_obs_move_feats_size()
        move_feats = np.zeros(move_feats_dim, dtype=np.float32)

        # Movement features
        avail_actions = self.get_avail_agent_actions(agent_id, is_opponent=is_opponent)
        for m in range(self.n_actions_move):
            move_feats[m] = avail_actions[m]

        return move_feats

    def get_obs_move_feats_size(self):
        """Returns the size of the vector containing the agents's movement-related features."""
        return 5

    def get_last_action(self, is_opponent=False):
        if is_opponent:
            ret = self.last_action_opponent
        else:
            ret = self.last_action
        return ret

    def get_avail_actions(self, is_opponent=False):
        if is_opponent:
            return [
                self.get_avail_agent_actions(agent_id, is_opponent=is_opponent)
                for agent_id in self.enemies.keys()
            ]
        else:
            return [
                self.get_avail_agent_actions(agent_id, is_opponent=is_opponent)
                for agent_id in self.agents.keys()
            ]

    def info(self):
        shape = (self.n_actions,)
        value = {"min": 0, "max": 1}
        return Action.info_template(shape, value, None, None)


if __name__ == "__main__":
    print(distance(0, 1, 2, 2))
    print(int(Direction.UP))
