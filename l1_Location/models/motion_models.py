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
    """https://classroom.udacity.com/courses/cs373/lessons/48684821/concepts/487362110923
    # Motion actions:
    #  [0,0] - stay
    #  [0,1] - right
    #  [0,-1] - left
    #  [1,0] - down
    #  [-1,0] - up
    """

    def __init__(self, motion_info: dict, verbose=True):
        self.p_move = motion_info["p_move"]
        self.p_still = 1 - self.p_move
        self.verbose = verbose

    def move(self, p: np.ndarray, U):
        # Handling U bigger than the world
        m_horizontal = U[1] % len(p[0])
        m_vertical = U[0] % len(p)
        q = p * self.p_still  # Computing the prob of not moving
        for i in range(len(p)):  # row
            for j in range(len(p[0])):  # col
                new_row = (i + m_vertical) % len(p)
                new_col = (j + m_horizontal) % len(p[0])
                q[new_row, new_col] += p[i, j] * self.p_move
        if self.verbose:
            print(f"Move with data {U}")
            print(q)
        return q

    def np_move(self, p: np.ndarray, U):
        m_horizontal = U[1] % len(p[0])
        m_vertical = U[0] % len(p)
        q_still = p * self.p_still  # Computing the prob of not moving
        q_move = np.roll(np.roll(p, m_vertical, axis=0), m_horizontal, axis=1) * self.p_move
        q = q_still + q_move
        if self.verbose:
            print(q)
        return q


if __name__ == '__main__':
    mm = MotionModel2D({"p_move": 1})
    stay = [0, 0]
    right = [0, 1]
    left = [0, -1]
    down = [1, 0]
    up = [-1, 0]
    world = np.array([[0, 1, 2], [0, 1, 0], [0, 1, 0], [0, 1, 1]])
    print(world)
    print("\nright")
    p = mm.np_move(world, right)
    p2 = mm.move(world, right)

    print("\nup")
    p = mm.np_move(p, up)
    p2 = mm.move(p2, up)

    print("\nleft")
    p = mm.np_move(p, left)
    p2 = mm.move(p2, left)

    print("\ndown")
    p = mm.np_move(p, down)
    p2 = mm.move(p2, down)
    debug = 0
    # p = mm.np_move(world, right)
