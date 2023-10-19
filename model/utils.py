import numpy as np
from model.model import Model

# def print_map_state(player: Model, opp_pos: dict):
#     map_state = np.zeros(shape=player.map.size)
#     # map_state += player.pos_score.sum(axis=0)
#     map_state = map_state.astype(int).astype(str)

#     # 空格
#     empty_pos = np.transpose((map_state == "0").nonzero())
#     rows, cols = zip(*empty_pos)
#     map_state[rows, cols] = "·"

#     # 墙
#     wall_pos = np.transpose(player.map.wall_map.nonzero())
#     rows, cols = zip(*wall_pos)
#     map_state[rows, cols] = "#"

#     # 金币
#     coin_pos = np.transpose(player.map.coin_map.nonzero())
#     rows, cols = zip(*coin_pos)
#     map_state[rows, cols] = "$"

#     print("===>", player.role, player.round, player.total_score)
#     rows, cols = map_state.shape
#     map_state = map_state.tolist()

#     # 自己位置
#     for i in range(len(player.agents)):
#         agent_pos = player.my_pos[player.agents[i]]
#         map_state[agent_pos[0]][agent_pos[1]] = str(player.agents[i])

#     # 敌人位置
#     for id, pos in opp_pos.items():
#         map_state[pos[0]][pos[1]] = str(id)

#     for i in range(rows):
#         for j in range(cols):
#             print(map_state[i][j], end=" ")
#         print()


def print_map(map):
    print("⊠⊠" * (len(map) + 1))
    for y in range(len(map[0])):
        print("⊠", end="")
        for x in range(len(map)):
            print(map[x][y], end=" ")
        print("⊠")
    print("⊠⊠" * (len(map) + 1))


def print_map_state(map_state, map_size):
    '''
    {
        "agents": [
            {     
                "id": agent_id //agent_id
                "x": 1, //x坐标
                "y": 1, //y坐标
                "powerups": { //所有持有道具的持续状态，没持有某个道具，则key不存在,value为持续时间
                    "invisibility": 5,
                    "passwall": 2,
                    "extravision": 3,
                    "shield": 6,
                    "sword": 4
                }  # 当前持有的道具
                "role": role  //agent当前角色，ATTACKER 或 DEFENDER
                "player_id" : player_id //用户id
                "vision_range" : 5 //视野范围
                "score" : 0 //持有的分数
                "invulnerability_duration" : 0 //无敌的回合
            },
            ...
        ],
        "walls": [
            {"x": 0,"y": 1},
            ...
        ],
        "portals": [
            {"x": 0,"y": 1,"pair": {"x":3,"y":4},"name": "A"},
            ...
        ],
        "powerups": [
            {"x": 0,"y": 5,"powerup": "invisibility"},
            ...
        ],
        "coins": [
            {"x": 0,"y": 5,"score": 6},
            ...
        ]
    }
    '''

    map = list()
    for i in range(map_size[0]):
        row = list()
        for j in range(map_size[1]):
            row.append("·")
        map.append(row)

    for wall in map_state["walls"]:
        map[wall["x"]][wall["y"]] = "⊠"

    for coin in map_state["coins"]:
        map[coin["x"]][coin["y"]] = "$"

    for portal in map_state["portals"]:
        map[portal["x"]][portal["y"]] = portal["name"]

    for powerup in map_state["powerups"]:
        map[powerup["x"]][powerup["y"]] = powerup["powerup"][0]

    for agent in map_state["agents"]:
        map[agent["x"]][agent["y"]] = agent["id"]

    print_map(map)


def get_walls():
    # return [(23, 23)]
    walls = [
        (2, 2), (3, 2), (4, 2), (5, 2), (7, 2), (8, 2),
        (9, 2), (10, 2), (13, 2), (14, 2), (15, 2), (16, 2), (18, 2), (19, 2),
        (20, 2), (21, 2), (2, 3), (10, 3), (13, 3), (21, 3), (2, 4), (5, 4),
        (6, 4), (10, 4), (13, 4), (16, 4), (21, 4), (2, 5), (8, 5), (10, 5),
        (13, 5), (16, 5), (18, 5), (19, 5), (21, 5), (4, 6), (8, 6), (2, 7),
        (4, 7), (10, 7), (13, 7), (15, 7), (16, 7), (18, 7), (21, 7), (2, 8),
        (6, 8), (7, 8), (10, 8), (13, 8), (18, 8), (21, 8), (2, 9), (10, 9),
        (13, 9), (21, 9), (2, 10), (3, 10), (4, 10), (5, 10), (7, 10), (8, 10),
        (9, 10), (10, 10), (13, 10), (14, 10), (15, 10), (16, 10), (18, 10),
        (19, 10), (20, 10), (21, 10), (2, 13), (3, 13), (4, 13), (5, 13),
        (7, 13), (8, 13), (9, 13), (10, 13), (13, 13), (14, 13), (15, 13),
        (16, 13), (18, 13), (19, 13), (20, 13), (21, 13), (2, 14), (10, 14),
        (13, 14), (21, 14), (2, 15), (5, 15), (7, 15), (10, 15), (13, 15),
        (16, 15), (18, 15), (21, 15), (2, 16), (4, 16), (8, 16), (10, 16),
        (13, 16), (15, 16), (16, 16), (18, 16), (19, 16), (21, 16), (2, 18),
        (4, 18), (8, 18), (10, 18), (13, 18), (15, 18), (16, 18), (18, 18),
        (19, 18), (21, 18), (2, 19), (5, 19), (7, 19), (10, 19), (13, 19),
        (16, 19), (18, 19), (21, 19), (2, 20), (10, 20), (13, 20), (21, 20),
        (2, 21), (3, 21), (4, 21), (5, 21), (7, 21), (8, 21), (9, 21),
        (10, 21), (13, 21), (14, 21), (15, 21), (16, 21), (18, 21), (19, 21),
        (20, 21), (21, 21)
    ]
    protal = [(0, 0), (6, 6), (17, 6), (6, 17), (17, 17), (23, 23)]
    return walls  # + protal
