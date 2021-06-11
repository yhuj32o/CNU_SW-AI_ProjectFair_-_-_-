from IMEM import *

class Decode:
    def __init__(self):
        self.opcode = 0
        self.rA = 0
        self.rB = 0
        self.rC = 0
        self.rD = 0
        self.rE = 0
        self.rF = 0
        self.imm26 = 0
        self.pool = 0
        self.fc = 0
        self.dw = 0
        self.pw = 0
        self.stride = 0
        self.imm_pool = 0
        self.zero_padding = 0
        self.pe_usage = 0

    def setting(self, line):
        self.line = int(line)
        self.opcode = (self.line & 0xF0000000) >> 28
        self.rA = (self.line & 0x0E000000) >> 25
        self.rB = (self.line & 0x01C00000) >> 22
        self.rC = (self.line & 0x003F0000) >> 19
        self.rD = (self.line & 0x00070000) >> 16
        self.rE = (self.line & 0x0000E000) >> 13
        self.rF = (self.line & 0x00001C00) >> 10
        self.imm26 = (self.line & 0x01FFFFFF)
        self.pool = 1 if self.opcode == 6 else 0
        self.fc = 1 if self.opcode == 5 else 0
        self.dw = (self.line & 0x00001000) >> 12
        self.pw = (self.line & 0x00000800) >> 11
        self.stride = (self.line & 0x00000400) >> 10
        self.imm_pool = (self.line & 0x003C0000) >> 18
        self.zero_padding = ~(self.pw)
        #self.pe_usage = self.rC if self.opcode == 4 else self.rB if self.opcode == 5 else self.rD if self.opcode == 6 else 0
#1010 1011 1100 1101 1110 1111 #ABCDEF
# OPCODE - 3bit  LOAD 1 STORE 2 MOVE 3 CONV 4 FC 5 POOL 6   0,7- X
# LOAD  1 rD(rA 사용)(3bit)  Input OR Filter (3bit *3) rW_H,rC,rT(rB rC rD)
# STORE 2 rS(rA 사용)(3bit)  rI_W_H, rI_C(rB rC)
# MOVE  3 rD(rA 사용)(3bit)  Imm26
# CONV  4 rFW_H(rA) rF_C(rB) rF_T(rC) rI_W_H(rD) rI_C(rE) dw pw stride
# FC    5 rF_W,rF_H, rI_W_H, rI_C rB_W, rB_H
# POOL  6 rI_W_H, rI_C imm_ pool_3


