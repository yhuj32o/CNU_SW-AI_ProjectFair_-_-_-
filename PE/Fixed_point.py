class Fixed_point:
    def __init__(self):
        self.result = 1

    def start_mul(self, weight, activation):
        for i in range(0, 7):
            self.result += weight[i] * activation

    def output(self):
        return self.result
