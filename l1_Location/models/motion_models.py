import numpy as np


class MotionModel1D:
    def __init__(self, pExact=0.8, pOvershoot=0.1, pUndershoot=0.1):
        self.pExact = pExact
        self.pOvershoot = pOvershoot
        self.pUndershoot = pUndershoot

    def move(self, p: np.array, U):
        q = np.zeros((len(p)))
        U = U % len(p)
        for i in range(len(p)):
            new_pos = i + U
            # handle undershoot
            q[(new_pos - 1) % len(p)] += p[i] * self.pUndershoot
            # handle exact
            q[new_pos % len(p)] += p[i] * self.pExact
            # handle overshoot
            q[(new_pos + 1) % len(p)] += p[i] * self.pOvershoot
        return q


class MotionModel2D:
    """https://classroom.udacity.com/courses/cs373/lessons/48684821/concepts/487362110923"""

    def __init__(self, p_move=0.8):
        self.p_move = p_move

    def move(self, p: np.ndarray, U):
        q = np.zeros((len(p)))
        U = U % len(p)
        for i in range(len(p)):
            new_pos = i + U
            # handle undershoot
            q[(new_pos - 1) % len(p)] += p[i] * self.pUndershoot
            # handle exact
            q[new_pos % len(p)] += p[i] * self.pExact
            # handle overshoot
            q[(new_pos + 1) % len(p)] += p[i] * self.pOvershoot
        return q
