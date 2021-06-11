'''
fixed-point
1 : 부호 비트  4:정수비트 3:소수비트
'''
from NPU.PE.Fixed_point import Fixed_point

'''
A = a * 2^L         B = b * 2^L
A + B = (a + b)*2^L             
A * B = (a * b) * 2^(2L) 
'''
class MAC:
    def __init__(self):
        self.weight[0:8] = 0;
        self.activation[0:8] = 0;
        self.output = 0;
        self.fixed_point = Fixed_point();
        return 0

    def start(self):
        for i in range(0,7):
            Fixed_point.start_mul(self.weight[i], self.activation[i])

        self.output = Fixed_point.output();

    def insert(self, input, weight):
        self.activaiton = input
        self.weight = weight
