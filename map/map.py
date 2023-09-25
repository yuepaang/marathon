import numpy as np


class Map:
    def __init__(self):
        '''
        地图基本信息
        '''
        size = [24, 24]
        self.coins = np.zeros(size)
        self.walls = np.zeros(size)
        self.power_up = np.zeros(size)
        self.paths_no_walls = dict()
        self.paths_with_walls = dict()
        pass

    def load_map(self):
        pass

    def update_map(self, obs):
        '''
        根据观测值更新地图信息,剩余金币、道具等
        '''
        pass
