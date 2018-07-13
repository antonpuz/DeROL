import numpy as np


class UniformDelay():
    def __init__(self, par=1.0):
        self.par = par

    def get_rand_delay(self):
        return int(np.random.uniform(self.par + 1.1)) #1.1 is used to make inclusive selection