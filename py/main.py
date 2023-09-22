import argparse
import marathon
import random
import rust_perf
from typing import List

ROLES = ["DEFENDER", "ATTACKER"]


def get_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def handle_agent(agent: marathon.Agent) -> str:
    current_pos = (agent.get_pos["x"], agent.get_pos["y"])
    coins_list = agent.get_coins()
    wall_list = agent.get_walls()
    other_agent_list: List[marathon.AgentDef] = agent.get_other_agents()

    block_list = []
    for wall in wall_list:
        block_list.append((wall["x"], wall["y"]))
    for other_agent in other_agent_list:
        if other_agent.get_role() == "ATTACKER":
            block_list.append((other_agent.get_pos["x"], other_agent.get_pos["y"]))

    if len(coins_list) > 0:
        coins_distance_list = [
            get_distance(current_pos, (coin["x"], coin["y"])) for coin in coins_list
        ]
        nearest_coin_idx = coins_distance_list.index(min(coins_distance_list))
        move_to = (
            coins_list[nearest_coin_idx]["x"],
            coins_list[nearest_coin_idx]["y"],
        )
        return rust_perf.get_direction(current_pos, move_to, block_list)
    else:
        direction = ["UP", "DOWN", "LEFT", "RIGHT"]
        return random.choice(direction)


class RealGame(marathon.Game):
    def __init__(self, match_id):
        super().__init__(match_id=match_id)
        self.picked_coin_set = set()

    def on_game_start(self, data):
        pass

    def on_game_state(self, data: marathon.MessageGameState):
        action = {}
        direction = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
        for k, v in data.get_states().items():
            # print(k,v.get_pos())
            # print(k,v.get_role())
            # print(k,v)
            # action[k] = random.choice(direction)
            action[k] = handle_agent(v)
        return action

    def on_game_end(self, data):
        pass
        # print(data)

    def on_game_over(self, data):
        pass
        # print(data)


if __name__ == "__main__":
    ps = argparse.ArgumentParser()
    ps.add_argument("-room", type=str, required=True)
    args = ps.parse_args()
    room = args.room
    g = RealGame(match_id=room)
    # 给一个同步的run
    g.sync_run()
