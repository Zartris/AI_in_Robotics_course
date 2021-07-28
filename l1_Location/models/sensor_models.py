import numpy as np


class SensorModel1D:
    def __init__(self, grid_map, sensor_accuracy, verbose):
        self.grid_map = grid_map
        self.verbose = verbose
        self.pHit = sensor_accuracy["pHit"]
        self.pMiss = sensor_accuracy["pMiss"]

    def sense(self, p, sensor_measurement):
        for i in range(len(self.grid_map)):
            hit = sensor_measurement == self.grid_map[i]
            p[i] *= (hit * self.pHit + (1 - hit) * self.pMiss)
        if self.verbose:
            print(f"Sense with data {sensor_measurement}")
        return p


class SensorModel2D:
    def __init__(self, grid_map, sensor_accuracy, verbose):
        self.grid_map = grid_map
        self.verbose = verbose
        self.pHit = sensor_accuracy["pHit"]
        if "pMiss" in sensor_accuracy:
            self.pMiss = sensor_accuracy["pMiss"]
        else:
            self.pMiss = 1 - self.pHit

    def sense(self, p, sensor_measurement, normalize=True):
        for i in range(len(self.grid_map)):  # row
            for j in range(len(self.grid_map[0])):  # col
                hit = sensor_measurement == self.grid_map[i][j]
                p[i][j] *= (hit * self.pHit + (1 - hit) * self.pMiss)
        if normalize:
            p = self.normalize(p)
        if self.verbose:
            print(f"Sense with data {sensor_measurement}")
            print(p)
        return p

    def np_sense(self, p, sensor_measurement, normalize=True):
        for i in range(len(self.grid_map)):  # row
            for j in range(len(self.grid_map[0])):  # col
                hit = sensor_measurement == self.grid_map[i][j]
                p[i, j] *= (hit * self.pHit + (1 - hit) * self.pMiss)
        if normalize:
            p = self.normalize(p)
        if self.verbose:
            print(f"Sense with data {sensor_measurement}")
            print(p)

        return p

    @staticmethod
    def normalize(p):
        s = np.sum(p)
        p /= s
        return p


if __name__ == '__main__':
    colors = [['R', 'G', 'G', 'R', 'R'],
              ['R', 'R', 'G', 'R', 'R'],
              ['R', 'R', 'G', 'G', 'R'],
              ['R', 'R', 'R', 'R', 'R']]
    measurements = ['G', 'G', 'G', 'G', 'G']
    pinit = 1.0 / float(len(colors)) / float(len(colors[0]))
    p = [[pinit for row in range(len(colors[0]))] for col in range(len(colors))]
    sensor_right = 0.7
    sm = SensorModel2D(colors, {"pHit": sensor_right}, True)
    for m in measurements:
        p = sm.sense(p, m)
