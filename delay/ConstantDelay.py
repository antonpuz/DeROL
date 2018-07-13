class ConstantDelay():
    def __init__(self, delay=1.0):
        self.delay = delay

    def get_rand_delay(self):
        return int(1 + self.delay)