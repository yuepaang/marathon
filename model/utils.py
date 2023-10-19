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


def x_min_max(verts):
    MIN, MAX = float('inf'), -float('inf')

    for vert in verts:
        if vert[0] < MIN:
            MIN = vert[0]
        if vert[0] > MAX:
            MAX = vert[0]

    return MIN, MAX


def y_min_max(verts):
    MIN, MAX = float('inf'), -float('inf')

    for vert in verts:
        if vert[1] < MIN:
            MIN = vert[1]
        if vert[1] > MAX:
            MAX = vert[1]

    return MIN, MAX


def get_line(a, b):
    '''
    returns A, B, C for standard form equation
    for line from point a to point b
    '''
    if b[0] == a[0]:
        m = 0
    else:
        m = (b[1] - a[1]) / (b[0] - a[0])

    x, y = a
    a = -m
    b = 1.0
    c = -x * m + y

    return (a, b, c)


def get_y(line, x):
    '''
    returns the y value at x on the line
    where line is (A, B, C) for standard form of the line
        (given by get_line())
    '''
    a, b, c = line
    y = (c - (a * x)) / b

    return y


def above_line(point, line, ab):
    '''
    returns if point is above a line
    true if right of vertical line
    true if above horizontal line
    returns null if on the line (useful for comparing two points)
    '''
    # Avoids divide by zero error
    if line[0] == 0:
        a, b = ab
        if a[0] == b[0]:
            x = a[0]
            if point[0] == x:
                return None
            else:
                return point[0] > x
        else:
            y = a[1]
            if point[1] == y:
                return None
            else:
                return point[1] > y

    else:
        y_on_line = get_y(line, point[0])
        if point[1] == y_on_line:
            return None
        else:
            return point[1] > y_on_line


def valid_point(point, tri):
    '''
    returns true if a point is inside the lines between
    the three points making the triangle
    '''
    a, b, c = tri
    ab = get_line(a, b)
    ac = get_line(a, c)
    bc = get_line(b, c)

    x_ab = above_line(point, ab, (a, b))
    x_ac = above_line(point, ac, (a, c))
    x_bc = above_line(point, bc, (b, c))

    if x_ab == None or x_ab == above_line(c, ab, (a, b)):
        if x_ac == None or x_ac == above_line(b, ac, (a, c)):
            if x_bc == None or x_bc == above_line(a, bc, (b, c)):
                return True

    return False


def all_valid_points(tri):
    '''
    gets all valid whole number coordinate pairs that are inside triangle
    triangle points must be in ints
    '''
    all_values = list(sum(tri, ()))
    for value in all_values:
        assert type(value) == int, "all triange points should be ints"
    x_min, x_max = x_min_max(tri)
    y_min, y_max = y_min_max(tri)
    # ab = get_line(tri[0], tri[1])
    # ac = get_line(tri[0], tri[2])
    # bc = get_line(tri[1], tri[2])

    valid = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            point = (x, y)
            v = valid_point(point, tri)

            if v or point in tri:
                valid.append(point)

    return valid


def isin_quadrilaterals(vertx, verty, point):
    c = False
    j = len(vertx) - 1

    for i in range(len(vertx)):
        if (((verty[i] > point[1]) != (verty[j] > point[1]))
                and (point[0] < (vertx[j] - vertx[i]) * (point[1] - verty[i]) /
                     (verty[j] - verty[i]) + vertx[i])):
            c = not c

            j = i

    return c
