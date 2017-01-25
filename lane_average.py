from lane import Lane
import numpy as np

class LaneAverage:
    def __init__(self):
        self.__xs = []
        self.__ys = []

    def update(self, lane):
        self.__xs.append(lane.xs)
        self.__ys.append(lane.ys)

        # limit arrays to last 3 lanes only
        self.__xs = self.__xs[-3:]
        self.__ys = self.__ys[-3:]

        self.lane = Lane(self.xs, self.ys)

    @property
    def pixels(self):
        return self.lane.pixels

    @property
    def meters(self):
        return self.lane.meters

    @property
    def length(self):
        return len(self.__xs) + len(self.__ys)

    @property
    def xs(self):
        if len(self.__xs) > 0:
            return np.concatenate(self.__xs)
        else:
            return self.__xs

    @property
    def ys(self):
        if len(self.__ys) > 0:
            return np.concatenate(self.__ys)
        else:
            return self.__ys

