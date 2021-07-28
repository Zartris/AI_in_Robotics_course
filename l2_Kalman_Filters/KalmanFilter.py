import numpy as np


class KalmanFilter:
    def __init__(self,
                 init_state: np.ndarray,
                 init_covariance_P: np.ndarray,
                 F: np.ndarray,
                 H: np.ndarray,
                 R: np.ndarray,
                 u: np.ndarray):
        """

        :param init_state: # N array
        :param init_covariance_P: # NxN matrix
        :param F: # next state function -- NxN matrix
        :param H: # measurement function -- Telling us what var we can observe (fx. x,y but not x_v, y_v) - N array
        :param R: # measurement uncertainty -- number of observable variable x number of observable variable matrix
        :param u: # external motion -- like if we were hit by a car -- N array
        """
        self.state = init_state
        self.covariance_P = init_covariance_P
        self.F = F
        self.H = H
        self.R = R
        self.u = u

    def __str__(self):
        return f"State:\n{str(self.state)}\nCovariance:\n{str(self.covariance_P)}"

    def update(self, z_measurement: np.ndarray):
        # @ is for matrix multiplication
        error_y = z_measurement.transpose() - (self.H @ self.state).transpose()
        # The system uncertainty (error) is projected into matrix S
        S = ((self.H @ self.covariance_P) @ self.H.transpose()) + self.R
        # From this is the Kalman gain computed
        K = self.covariance_P @ self.H.transpose() @ np.linalg.inv(S)
        new_state_x = self.state + (K @ error_y.transpose())
        new_covariance_P = (np.identity(len(self.covariance_P)) - K @ self.H) @ self.covariance_P
        self.state = new_state_x
        self.covariance_P = new_covariance_P
        return self.state, self.covariance_P

    def predict(self):
        # @ is for matrix multiplication
        new_state_x = (self.F @ self.state) + self.u
        new_covariance_P = (self.F @ self.covariance_P) @ self.F.transpose()
        self.state = new_state_x
        self.covariance_P = new_covariance_P
        return new_state_x, new_covariance_P


if __name__ == '__main__':
    print("### 4-dimensional example ###")
    measurements = [[5., 10.], [6., 8.], [7., 6.], [8., 4.], [9., 2.], [10., 0.]]
    # initial state (location and velocity)
    initial_xy = np.array([[4., 12., 0., 0.]]).transpose()
    u = np.array([[0.], [0.], [0.], [0.]])  # external motion
    dt = 0.1
    # initial uncertainty: 0 for positions x and y, 1000 for the two velocities
    P = np.array([[0., 0., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 1000., 0.],
                  [0., 0., 0., 1000.]])
    # next state function: generalize the 2d version to 4d
    F = np.array([[1., 0., dt, 0.],
                  [0., 1., 0., dt],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]])
    # measurement function: reflect the fact that we observe x and y but not the two velocities
    H = np.array([[1., 0., 0., 0.],
                  [0., 1., 0., 0.]])
    # measurement uncertainty: use 2x2 matrix with 0.1 as main diagonal
    R = np.array([[0.1, 0.], [0., 0.1]])
    # 4d identity matrix

    kf = KalmanFilter(init_state=initial_xy,
                      init_covariance_P=P,
                      F=F,
                      H=H,
                      R=R,
                      u=u)

    for n in range(len(measurements)):
        # prediction
        kf.predict()
        # measurement update
        kf.update(np.array(measurements[n]))
    print(str(kf))
