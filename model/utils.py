import numpy as np
from model.model import Model


def print_map_state(player: Model, opp_pos: dict):
    map_state = np.zeros(shape=player.map.size)
    # map_state += player.pos_score.sum(axis=0)
    map_state = map_state.astype(int).astype(str)

    # 空格
    empty_pos = np.transpose((map_state == "0").nonzero())
    rows, cols = zip(*empty_pos)
    map_state[rows, cols] = "·"

    # 墙
    wall_pos = np.transpose(player.map.wall_map.nonzero())
    rows, cols = zip(*wall_pos)
    map_state[rows, cols] = "#"

    # 金币
    coin_pos = np.transpose(player.map.coin_map.nonzero())
    rows, cols = zip(*coin_pos)
    map_state[rows, cols] = "$"

    print("===>", player.role, player.round)
    rows, cols = map_state.shape
    map_state = map_state.tolist()

    # 自己位置
    for i in range(len(player.agents)):
        agent_pos = player.my_pos[player.agents[i]]
        map_state[agent_pos[0]][agent_pos[1]] = str(player.agents[i])

    # 敌人位置
    for id, pos in opp_pos.items():
        map_state[pos[0]][pos[1]] = str(id)

    for i in range(rows):
        for j in range(cols):
            print(map_state[i][j], end=" ")
        print()

    print()
