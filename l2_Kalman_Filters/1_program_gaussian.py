from math import *


class Gaussian:
    def __init__(self, init_mu, init_sigma):
        self.mu = init_mu  # Remember mu is the mean of the distribution
        self.sigma2 = init_sigma  # Remember sigma is the variance of the distribution

    def __str__(self):
        return f"mu: {self.mu}, sigma2: {self.sigma2}"

    def __repr__(self):
        return str(self)


class kalmanfilter1D(Gaussian):
    def __init__(self, init_mu, init_sigma):
        super().__init__(init_mu, init_sigma)

    def f(self, x):
        normalize_constant = 1 / (sqrt(2 * pi * self.sigma2))
        return normalize_constant * exp(-0.5 * (((x - self.mu) ** 2) / self.sigma2))

    def update(self, g):
        self.mu = (g.sigma2 * self.mu + self.sigma2 * g.mu) / (self.sigma2 + g.sigma2)
        self.sigma2 = 1 / (1 / g.sigma2 + 1 / self.sigma2)
        return self

    def predict(self, g):
        self.mu = self.mu + g.mu
        self.sigma2 = self.sigma2 + g.sigma2
        return self


if __name__ == '__main__':
    measurements = [5., 6., 7., 9., 10.]
    motion = [1., 1., 2., 1., 1.]
    measurement_sig = 4.
    motion_sig = 2.
    mu = 0.
    sigma2 = 10000.
    kf = kalmanfilter1D(mu, sigma2)
    for i in range(len(measurements)):
        print(kf.update(Gaussian(measurements[i], measurement_sig)))
        print(kf.predict(Gaussian(motion[i], motion_sig)))
