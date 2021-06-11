import numpy as np

class IMEM :
    def __init__(self):
        self.file = 'test_program.txt'
        f = open(self.file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)
        self.code = f.readlines()
        self.line = np.zeros((len(self.code)))
        for i in range(0,len(self.code)) :          
            value = 0
            for j in range(0, 32):
                value <<= 1
                value = value + 0 if self.code[i][j] == '0' else value + 1
            self.line[i] = int(value)



