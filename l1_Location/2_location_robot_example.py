import numpy as np

from l1_Location.models.motion_models import MotionModel1D


class ProbabilisticRobot:
    def __init__(self, world: list, sensor_accuracy: dict):
        self.world = world
        self.pHit = sensor_accuracy["hit_ratio"]
        self.pMiss = sensor_accuracy["miss_ratio"]
        self.motion_model = MotionModel1D()
        self.verbose = True
        # init:
        self.p = np.full((len(self.world)), 1. / len(self.world))

    def sense(self, sensor_measurement):
        for i in range(len(self.p)):
            hit = sensor_measurement == self.world[i]
            self.p[i] *= (hit * self.pHit + (1 - hit) * self.pMiss)
        self.normalize()
        if self.verbose:
            print(f"Sense with data {sensor_measurement}")
            self.compute_entropy()
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
        self.normalize()
        if self.verbose:
            print(f"Move with data {U}")
            self.compute_entropy()
        return self.p

    def compute_entropy(self):
        entropy = np.sum(-self.p * np.log(self.p))
        if self.verbose:
            print(f"current entropy: {entropy}")
        return entropy


if __name__ == '__main__':
    world = ["green", "red", "red", "green", "green"]
    sensor_accuracy = {
        "hit_ratio": 0.6,
        "miss_ratio": 0.2
    }
    pr = ProbabilisticRobot(world, sensor_accuracy)
    measurements = ["red", "green"]
    motions = [1, 1]

    for i in range(len(measurements)):
        print(f"step {i}")
        pr.sense(measurements[i])
        pr.inaccurate_move(motions[i])
        print(f"p: {pr.p}")
