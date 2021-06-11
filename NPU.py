from PE.PE import *
from Memory.Memory import *
from Execute import *
from IMEM import *
from Decode import *
from Register import*


IMEM = IMEM()
Decode = Decode()
Reg = Register()
Mem = Memory()
D_memory = D_memory()
PE = PE()
Exe = Execute(Reg, PE, D_memory)



for i in range(len(IMEM.line)) :
    Decode.setting(IMEM.line[i])
    #LOAD, STORE, MOVE, CONV, FC, POOL
    if Decode.opcode == 1 :
        Exe.LOAD(Decode.rA, Decode.rB, Decode.rC, Decode.rD)
    elif Decode.opcode == 2 :
        Exe.STORE(Decode.rA, Decode.rB, Decode.rC)
    elif Decode.opcode == 3 :
        Exe.MOVE(Decode.rA, Decode.imm26)
    elif Decode.opcode == 4 :
        Exe.CONV(Decode.rA, Decode.rB, Decode.rC, Decode.rD, Decode.rE, Decode.dw, Decode.pw, Decode.stride)
    elif Decode.opcode == 5 :
        Exe.FC(Decode.rA, Decode.rB, Decode.rC, Decode.rD, Decode.rE, Decode.rF)
    elif Decode.opcode == 6 :
        Exe.POOL(Decode.rA, Decode.rB, Decode.imm_pool)



max = 0
for i in range(1, Reg.open_output_data().shape[0]) :
    max = max if Reg.open_output_data()[max] > Reg.open_output_data()[i] else i

print("최종결과 : ",max)
