
from Memory.D_memory import *
from Memory.GlobalBuffer import *
from Memory.LocalBuffer_Filter import *
from Memory.LocalBuffer_Input import *


class Memory:
    def __init__(self):
        self.D_memory = D_memory()
        self.GlobalBuffer = GlobalBuffer()
        self.LocalBuffer_Input = LocalBuffer_Input()
        self.LocalBuffer_Filter = LocalBuffer_Filter()

    def upper(self, mem_enable, read_write,type_num ,RF_rD, RF_rW_H, RF_rC, RF_rT):
            #type_num= 0 - D_mem -> global, 1 global -> Local_input, 2 global -> Local_filter
            self.type_num=type_num
            self.RF_rD = RF_rD
            self.RF_rW_H = RF_rW_H
            self.RF_rC = RF_rC
            self.RF_rT = RF_rT
            self.All_data_num = self.RF_rW_H ** 2 * self.RF_rC * self.RF_rT
            #self.data = np.zeros(self.RF_rW_H ** 2 * self.RF_rC * self.RF_rT)

            if self.type_num ==0:
                #print(self.data.shape)

                for num in range(self.All_data_num):  # 0 ~ data_end
                    width = num % self.RF_rW_H
                    height = int(num / self.RF_rW_H) % self.RF_rW_H
                    channel= int(num / (self.RF_rW_H ** 2)) % self.RF_rC
                    tensor = int(num / (self.RF_rW_H * self.RF_rW_H * self.RF_rC))
                    target_address = self.RF_rD + self.RF_rW_H * (self.RF_rW_H * (self.RF_rC * tensor + channel) + height) + width
                    #print(target_address)
                    self.GlobalBuffer.storage[((RF_rD+num)%self.GlobalBuffer.storage.shape[0])] = self.D_memory.storage[target_address]


            elif self.type_num==1:

                for num in range(self.All_data_num):  # 0 ~ data_end
                    width = num % self.RF_rW_H
                    height = int(num / self.RF_rW_H) % self.RF_rW_H
                    channel = int(num / (self.RF_rW_H ** 2)) % self.RF_rC
                    tensor = int(num / (self.RF_rW_H * self.RF_rW_H * self.RF_rC))
                    target_address = self.RF_rD + self.RF_rW_H * (self.RF_rW_H * (self.RF_rC * tensor + channel) + height) + width

                    self.LocalBuffer_Input.storage[num]=self.GlobalBuffer.storage[((RF_rD+target_address)%self.GlobalBuffer.storage.shape[0])]

            elif self.type_num==2:
                for num in range(self.All_data_num):  # 0 ~ data_end
                    width = num % self.RF_rW_H
                    height = int(num / self.RF_rW_H) % self.RF_rW_H
                    channel = int(num / (self.RF_rW_H ** 2)) % self.RF_rC
                    tensor = int(num / (self.RF_rW_H * self.RF_rW_H * self.RF_rC))
                    target_address = self.RF_rD + self.RF_rW_H * (self.RF_rW_H * (self.RF_rC * tensor + channel) + height) + width


                    self.LocalBuffer_Filter.storage[num]=self.GlobalBuffer.storage[((target_address)%self.GlobalBuffer.storage.shape[0])]
                #self.GlobalBuffer.storage=[]


    def lower(self,mem_enable, read_write,type_num ,RF_rD, RF_rW_H, RF_rC, RF_rT):
        # type_num= 0 - LocalBuffer -> Global buffer, 1 - Global buffer->D_mem (lower의 경우classifier만 이동)
        self.type_num = type_num
        self.RF_rD = RF_rD
        self.RF_rW_H = RF_rW_H
        self.RF_rC = RF_rC
        self.RF_rT = RF_rT
        self.All_data_num = self.RF_rW_H ** 2 * self.RF_rC * self.RF_rT
        # self.data = np.zeros(self.RF_rW_H ** 2 * self.RF_rC * self.RF_rT)
        if self.type_num == 0:


            for num in range(self.All_data_num):  # 0 ~ data_end
                width = num % self.RF_rW_H
                height = int(num / self.RF_rW_H) % self.RF_rW_H
                channel= int(num / (self.RF_rW_H ** 2)) % self.RF_rC
                tensor = int(num / (self.RF_rW_H * self.RF_rW_H * self.RF_rC))

                target_address = self.RF_rD + self.RF_rW_H * (self.RF_rW_H * (self.RF_rC * tensor + channel) + height) + width

                self.GlobalBuffer.storage[RF_rD+num]=self.LocalBuffer_Input.storage[target_address]

        elif self.type_num == 1:

            for num in range(self.All_data_num):  # 0 ~ data_end
                width = num % self.RF_rW_H
                height = int(num / self.RF_rW_H) % self.RF_rW_H
                channel = int(num / (self.RF_rW_H ** 2)) % self.RF_rC
                tensor = int(num / (self.RF_rW_H * self.RF_rW_H * self.RF_rC))

                target_address = self.RF_rD + self.RF_rW_H * (self.RF_rW_H * (self.RF_rC * tensor + channel) + height) + width
                #print(target_address)
                self.D_memory.storage=np.append(self.D_memory.storage,self.GlobalBuffer.storage[target_address])

                






