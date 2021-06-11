from Memory.LocalBuffer_Input import *
from Memory.LocalBuffer_Filter import *
from Memory.Memory import *
import numpy as np

class Register:
    def __init__(self):
        self.reg = np.zeros(8)
        self.reg_input = 0
        self.reg_weight = 0
        self.reg_bias=0
        self.reg_output = 0
        # input, filter, bias, output 순
# 128 byte -> Input 8개 , Weight 8개 => 16개

# 256 byte -> Input 16개, weight 16개?
#PE 16개 128*16=

# PE 내부에 8개 의 MAC
    #MAC 당 Input,weight 각각 8byte    16 byte
    #128byte

        #reg _8 - 직접 접근하는 레지스터
        #reg_data - 메모리계층에서 사용하는 레지스터
        #reg_data 1개당 64bit씩 수용
    def open(self, addr):
        return self.reg[addr]

    def open_data(self):
        return [self.reg_input, self.reg_weight,self.reg_bias]

    def open_output_data(self):
        return self.reg_output
    def close(self, rD, imm) :
        self.reg[rD] = imm

    def close_data(self, output):
        self.reg_output = output

    def close_input_data(self, input):
        self.reg_input = input

    def close_filter_data(self, filter):
        self.reg_weight = filter

    def close_output_data(self, output):
        self.reg_output = output


