import numpy as np


class PoissonDelay():
    def __init__(self, lam=1.0):
        self.lam = lam

    def get_rand_delay(self):
        return int(1 + np.random.poisson(self.lam))