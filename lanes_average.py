from lane_average import LaneAverage
from lanes import Lanes

class LanesAverage:
    def __init__(self):
        self.left = LaneAverage()
        self.right = LaneAverage()

    def update(self, lanes):
        self.left.update(lanes.left)
        self.right.update(lanes.right)

        self.lanes = Lanes(self.left, self.right)

    def distance_from_center(self, center):
        return self.lanes.distance_from_center(center)

    def lane_distance(self, y):
        return self.lanes.lane_distance(y)

    def lanes_parallel(self, height, samples=50):
        return self.lanes.lanes_parallel(height, samples)

