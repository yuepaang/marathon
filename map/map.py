import numpy as np


class Map:
    def __init__(self):
        '''
        地图基本信息
        '''
        self.size = [24, 24]
        self.coin_map = np.zeros(self.size)
        self.wall_map = np.zeros(self.size)
        self.power_up_map = np.zeros(self.size)
        self.portal = dict()
        self.paths_no_walls = dict()
        self.paths_with_walls = dict()
        self.power_up_map = {
            "Powerup.INVISIBILITY": 2,
            "Powerup.PASSWALL": 3,
            "Powerup.EXTRAVISION": 4,
            "Powerup.SHIELD": 5,
            "Powerup.SWORD": 6,
        }
        return

    def load_map(self, map: dict):
        for cell in map:
            x = cell["x"]
            y = cell["y"]
            cell_type = cell["type"]
            map_coor = self.obs_to_map_coor([x, y])
            if cell_type == "COIN":
                self.coin_map[map_coor[0], map_coor[1]] = 1
            if cell_type == "POWERUP":
                self.power_up_map[map_coor[0], map_coor[1]] = 1
            if cell_type == "WALL":
                self.wall_map[map_coor[0], map_coor[1]] = 1
            if cell_type == "PORTAL":
                self.portal[(map_coor[0], map_coor[1])] = self.obs_to_map_coor(
                    (cell["pair"]["x"], cell["pair"]["y"]))
        return

    def obs_to_map_coor(self, coor: tuple):
        return (coor[1], coor[0])

    def map_to_obs_coor(self, coor: tuple):
        return (coor[1], coor[0])

    def to_action(self, ori_coor: tuple, new_coor: tuple):
        if new_coor[0] - ori_coor[0] < 0:
            return "UP"
        if new_coor[0] - ori_coor[0] > 0:
            return "DOWN"
        if new_coor[1] - ori_coor[1] < 0:
            return "LEFT"
        if new_coor[1] - ori_coor[1] > 0:
            return "RIGHT"
        return "STAY"

    def get_nearby_coor(self, coor: tuple):
        '''
        相邻格子坐标
        '''
        nearby = list()
        for diff in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            pos = (coor[0] + diff[0], coor[1] + diff[1])
            if pos[0] < 0 or pos[0] >= self.size[0] or pos[1] < 0 or pos[
                    1] >= self.size[1]:
                continue
            if self.wall_map[pos[0], pos[1]] != 1:
                nearby.append(pos)
        return nearby

    def distance(self, coor1: tuple, coor2: tuple):
        return abs(coor1[0] - coor2[0]) + abs(coor1[1] - coor2[1])

    def get_view_coor(self, coor: tuple, view_range: int):
        '''
        视野内格子的坐标
        '''
        view = list()
        for x in range(max(coor[0] - view_range, 0),
                       min(coor[0] + view_range, self.size[0] - 1) + 1):
            for y in range(max(coor[1] - view_range, 0),
                           min(coor[1] + view_range, self.size[1] - 1) + 1):
                view.append((x, y))
        return view

    def update_map(self, obs: dict):
        '''
        根据观测值更新地图信息,剩余金币、道具等
        '''
        for id, view in obs.items():
            pos = self.obs_to_map_coor(
                (view["self_agent"]["x"], view["self_agent"]["y"]))
            # 吃当前位置的金币/道具
            self.coin_map[pos[0], pos[1]] = 0
            self.power_up_map[pos[0], pos[1]] = 0
            # 金币
            for p in view["coins"]:
                mp = self.obs_to_map_coor((p["x"], p["y"]))
                self.coin_map[mp[0], mp[1]] = 1
            # 道具
            for p in view["powerups"]:
                mp = self.obs_to_map_coor((p["x"], p["y"]))
                self.power_up_map[mp[0],
                                  mp[1]] = self.power_up_map[p["powerup"]]
        return
