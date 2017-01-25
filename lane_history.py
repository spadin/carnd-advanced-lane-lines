import numpy as np

def difference(a, b):
    return abs(a - b)

class LaneHistory:
    def __init__(self):
        self.history = []
        self.curvature_history = []
        self.fit_history = []

    def update(self, lane):
        self.history.append(lane)

        A, B, C = lane.fit()
        self.curvature_history.append(lane.curvature())
        self.fit_history.append([A, B, C])

        # if(len(self.curvature_history) > 0):
        #     if(difference(self.mean_curvature(), lane.curvature()) < 70):
        #         self.curvature_history.append(lane.curvature())
        #         self.fit_history.append([A, B, C])
        # else:
        #     self.curvature_history.append(lane.curvature())
        #     self.fit_history.append([A, B, C])

    @property
    def last_lane(self):
        if len(self.history) > 0:
            return self.history[-1]
        else:
            return None

    def mean_curvature(self):
        return np.mean(self.curvature_history)

    def curvature(self):
        return self.mean_curvature()

    def mean_fit(self):
        return np.mean(self.fit_history, axis=0)

    def fit(self):
        return self.mean_fit()

    def distance_from_center(self, other_lane_history, width):
        return self.last_lane.distance_from_center(other_lane_history.last_lane, width)

