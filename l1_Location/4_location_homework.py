import numpy as np

from l1_Location.models.motion_models import MotionModel1D, MotionModel2D
from l1_Location.models.sensor_models import SensorModel2D


class ProbabilisticRobot2D:
    def __init__(self, world: list, sensor_info: dict, motion_info: dict):
        self.verbose = True
        self.world = world
        self.motion_model = MotionModel2D(motion_info, self.verbose)
        self.sensor_model = SensorModel2D(world, sensor_info, self.verbose)
        # init:
        pinit = 1.0 / float(len(world)) / float(len(world[0]))
        p = [[pinit for row in range(len(world[0]))] for col in range(len(world))]
        self.p = np.array(p)

    def sense(self, measurement):
        self.p = self.sensor_model.sense(self.p, measurement)
        return self.p

    def normalize(self):
        s = np.sum(self.p)
        self.p /= s

    def accurate_move(self, U):
        """
        shift p with U
        :param U: direction 1=right, -1=left, -3=3xleft
        :return:
        """
        U = U % len(self.p)
        self.p = np.roll(self.p, U)
        return self.p

    def inaccurate_move(self, U):
        """
        shift p with U
        :param U: direction 1=right, -1=left, -3=3xleft
        :return:
        """

        self.p = self.motion_model.move(self.p, U)
        if self.verbose:
            self.compute_entropy()
        return self.p

    def compute_entropy(self):
        entropy = np.sum(-self.p * np.log(self.p))
        if self.verbose:
            print(f"current entropy: {entropy}")
        return entropy


if __name__ == '__main__':
    world = [['R', 'G', 'G', 'R', 'R'],
             ['R', 'R', 'G', 'R', 'R'],
             ['R', 'R', 'G', 'G', 'R'],
             ['R', 'R', 'R', 'R', 'R']]
    sensor_info = {
        "pHit": 0.7
    }
    motion_info = {
        "p_move": 0.8
    }
    pr = ProbabilisticRobot2D(world, sensor_info, motion_info)
    measurements = ['G', 'G', 'G', 'G', 'G']
    motions = [[0, 0], [0, 1], [1, 0], [1, 0], [0, 1]]

    for i in range(len(measurements)):
        print(f"step {i}")
        pr.inaccurate_move(motions[i])
        pr.sense(measurements[i])
        print(f"p: {pr.p}")
