import PE


'''
fixed-point
1 : 부호 비트  2:정수비트 5:소수비트
'''


'''
PE_Block
 -PE 16개
    -MAC 8개

'''


class PE_Block:
    def __init__(self):
        ############# Control #############
        self.RF_rF_W = 0
        self.RF_rF_H = 0
        self.RF_rF_C = 0
        self.RF_rF_T = 0
        self.RF_rI_W_H = 0
        self.RF_rI_C = 0
        self.RF_rB_W = 0
        self.RF_rB_H = 0
        self.RF_data = 0
        self.dw = 0
        self.pw = 0
        self.stride = 0
        self.pe_usage = 0
        self.pool = 0
        self.fc = 0

        self.PE_iter = 0
        self.PE_left = 0
        self.MAC_num = 0
        self.MAC_final = 0
        self.MAC_iter = 0
        self.MAC_operator = 0

        ################ PE ###############

        PE_0 = PE()

        PE_1 = PE()
        PE_2 = PE()
        PE_3 = PE()
        PE_4 = PE()
        PE_5 = PE()
        PE_6 = PE()
        PE_7 = PE()
        PE_8 = PE()
        PE_9 = PE()
        PE_10 = PE()
        PE_11 = PE()
        PE_12 = PE()
        PE_13 = PE()
        PE_14 = PE()
        PE_15 = PE()

        #################

    def setting(self, RF_rF_W, RF_rF_H, RF_rF_C, RF_rF_T, RF_rI_W_H, RF_rI_C, RF_rB_W, RF_rB_H, RF_data, pool, fc, dw,
                pw, stride, pe_usage):
        self.RF_rF_W = RF_rF_W
        self.RF_rF_H = RF_rF_H
        self.RF_rF_C = RF_rF_C
        self.RF_rF_T = RF_rF_T
        self.RF_rI_W_H = RF_rI_W_H
        self.RF_rI_C = RF_rI_C
        self.RF_rB_W = RF_rB_W
        self.RF_rB_H = RF_rB_H
        self.RF_data = RF_data
        self.pool = pool
        self.fc = fc
        self.dw = dw
        self.pw = pw
        self.stride = stride
        self.pe_usage = pe_usage

        self.PE_iter = (RF_rI_C >> 3) >> 4 if self.pool else 0  # 16개의 PE가 몇번 반복되는지
        self.PE_left = (RF_rI_C >> 3) & 0x0000000F if self.pool else 0  # 마지막 PE는 몇개가 남는지
        self.MAC_num = self.RF_rI_C if self.pool else 0 # 총 몇개의 MAC이 사용되는지
        self.MAC_final = self.RF_rI_C & 0x00000007 if self.pool else 0 # 마지막 MAC은 몇개가 인지
        self.MAC_iter = self.RF_rI_W_H if self.pool else 0 # MAC당 몇번 반복되는지
        self.MAC_operator = self.RF_rI_W_H if self.pool else  0# MAC의 operaotr는 몇개가 사용되는지
        '''
        self.MAC_iter = RF_rF_W if self.fc else self.RF_rF_H if self.pool else self.RF_rF_H * self.RF_rF_W
        self.MAC_num = RF_rF_W if self.fc else self.RF_rF_H if self.pool else RF_rF_C
        '''


