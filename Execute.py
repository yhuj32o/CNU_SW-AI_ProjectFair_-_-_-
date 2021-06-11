import numpy as np
class Execute :
    def __init__(self, reg, PE, D_memory):
        self.Reg = reg
        self.PE=PE
        self.D_memory = D_memory

    def LOAD(self, rD, rW_H, rC, rT) :

        RF_rD = self.Reg.open(rD)
        RF_rW_H = self.Reg.open(rW_H)
        RF_rC = self.Reg.open(rC)
        RF_rT = self.Reg.open(rT)
        if RF_rD == 0:
            if np.all(self.Reg.reg_input) ==0:
                self.Reg.reg_input = self.D_memory.upper(1, 0, RF_rD, RF_rW_H, RF_rC, RF_rT)
        elif RF_rW_H*RF_rC==1000:
                self.Reg.reg_bias = self.D_memory.upper(1, 0, RF_rD, RF_rW_H, RF_rC, RF_rT)
        else :
                self.Reg.reg_weight = self.D_memory.upper(1, 0, RF_rD, RF_rW_H, RF_rC, RF_rT)

        if RF_rW_H*RF_rC==1024000:
            print("LOAD ", int(RF_rD), " to ", int(int(RF_rD) + ((RF_rW_H * RF_rC))))
        elif RF_rW_H*RF_rC==1000:
            print("LOAD ", int(RF_rD), " to ", int(int(RF_rD) + ((RF_rW_H * RF_rC * RF_rT))))
        else:
            print("LOAD ", int(RF_rD), " to ", int(int(RF_rD) + ((RF_rW_H * RF_rW_H * RF_rC * RF_rT))))
    def STORE(self, rS, rI_W_H, rI_C) :
        RF_rS = self.Reg.open(rS)
        RF_rI_W_H = self.Reg.open(rI_W_H)
        RF_rI_C = self.Reg.open(rI_C)

        #Mem.D_memory_storage.lower(1, 1, RF_rS, RF_rI_W_H, RF_rI_C)

        print("STORE " , RF_rS + " to " , ((RF_rI_W_H * RF_rI_C) + RF_rS))

    def MOVE(self, rD, Imm26) :
        self.Reg.close(rD, Imm26)

        print("MOVE ", "r", rD, " ", Imm26)

    def CONV(self, rF_W_H, rF_C, rF_T, rI_W_H, rI_C, dw, pw, stride):

        RF_rF_W_H = self.Reg.open(rF_W_H)
        RF_rF_C = self.Reg.open(rF_C)
        RF_rF_T = self.Reg.open(rF_T)
        RF_rI_W_H = self.Reg.open(rI_W_H)
        RF_rI_C = self.Reg.open(rI_C)
        RF_data = self.Reg.open_data()

        # RF_rF_W, RF_rF_H, RF_rF_C, RF_rF_T, RF_rI_W_H, RF_rI_C, rB_W, rB_H, RF_data,
        if dw:
            self.Reg.close_data(
                self.PE.CAL(RF_rI_W_H,RF_rI_C,RF_rF_W_H,RF_rF_W_H,RF_rF_C,RF_rF_T,0,0,stride,dw,pw,0,0,RF_data[0], RF_data[1],0))
        elif pw:
            self.Reg.close_data(
                self.PE.CAL(RF_rI_W_H, RF_rI_C, RF_rF_W_H, RF_rF_W_H, RF_rF_C, RF_rF_T, 0, 0, stride, dw, pw, 0, 0,RF_data[0], RF_data[1],0))
        else:
            self.Reg.close_data(
                self.PE.CAL(RF_rI_W_H, RF_rI_C, RF_rF_W_H, RF_rF_W_H, RF_rF_C, RF_rF_T, 0, 0, stride, dw, pw, 0, 0,RF_data[0], RF_data[1],0))

        self.Reg.close_input_data(self.Reg.reg_output)
    def FC(self, rF_W, rF_H, rI_W_H, rI_C, rB_W, rB_H):
        RF_rF_W = self.Reg.open(rF_W)
        RF_rF_H = self.Reg.open(rF_H)
        RF_rI_W_H = self.Reg.open(rI_W_H)
        RF_rI_C = self.Reg.open(rI_C)
        RF_rB_W = self.Reg.open(rB_W)
        RF_rB_H = self.Reg.open(rB_H)
        RF_data = self.Reg.open_data()
        self.Reg.close_data(
            self.PE.CAL(RF_rI_W_H, RF_rI_C, RF_rF_W, RF_rF_H, 0, 0, RF_rB_W, RF_rB_H, 0, 0, 0, 0, 1,RF_data[0], RF_data[1],RF_data[2]))
        self.Reg.close_input_data(self.Reg.reg_output)
    def POOL(self, rI_W_H, rI_C, imm_POOL_3):
        RF_rI_W_H = self.Reg.open(rI_W_H)
        RF_rI_C = self.Reg.open(rI_C)
        RF_data = self.Reg.open_data()
        self.Reg.close_data(
            self.PE.CAL(RF_rI_W_H, RF_rI_C, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,RF_data[0], 0,0))
        self.Reg.close_input_data(self.Reg.reg_output)


            



