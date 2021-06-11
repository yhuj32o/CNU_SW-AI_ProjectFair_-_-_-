import numpy as np
from os.path import join
import sys
'''
fixed-point
1 : 부호 비트  4:정수비트 3:소수비트
'''


'''
A = a * 2^L         B = b * 2^L
A + B = (a + b)*2^L             
A * B = (a * b) * 2^(2L) 
'''
class PE:
    def __init__(self):

        self.test_input = 0
        self.test_shape = 0
        self.test_filter_1 = 0
        self.test_filter_shape = 0

        self.test_weight = 0
        self.test_bias = 0

        #self.weight[0:8]=0;
        #self.activation[0:8] = 0;
        #self.output = 0;

    def CAL(self,rI_W_H,rI_C,rF_W,rF_H,rF_C,rF_T,rB_W,rB_H,STRIDE,DW,PW,POOL,FC,input_data,filter_data,bias_data):
        self.rI_W=int(rI_W_H)
        self.rI_H=int(rI_W_H)
        self.rI_C=int(rI_C)

        self.rF_W=int(rF_W)
        self.rF_H=int(rF_H)
        self.rF_C=int(rF_C)
        self.rF_T=int(rF_T)

        self.rB_W = int(rB_W)
        self.rB_H = int(rB_H)
        # rB_W,H, 사용처 ?

        self.All_data_num=self.rI_H  * self.rI_W * self.rI_C

        self.stride=0
        self.dw=DW
        self.pw=PW
        self.pool=POOL
        self.fc=FC


        self.test_shape= input_data
        self.test_filter_shape=filter_data
        self.test_bias = bias_data

        self.input=[]
        self.filter=[]

        self.sum = []
        self.out = []

        if STRIDE == 1 :
            self.stride = 2
        else:
            self.stride = 1



        #general
        #224,224,3
        #3,3,3,32


        if self.dw==0 and self.pw==0 and self.pool==0 and self.fc==0:
            print("general Convolution")
            print(self.rI_W, self.rI_H, self.rI_C)
            print(self.rF_W,self.rF_H,self.rF_C,self.rF_T)
            print(self.All_data_num)


            for h in range(int(self.rI_H/self.stride)):
                for w in range(int(self.rI_W/self.stride)):

                    for c in range(self.rF_T):
                        for f_size in range(self.rF_W*self.rF_H*self.rF_C):
                            self.filter.append(self.test_filter_shape[self.rF_T*f_size+c])

                            if c == 0:
                                if f_size <9:
                                    if w==111 and f_size>=6:
                                        self.input.append(0)

                                    else:
                                        self.input.append(self.test_shape[((224 * (6 * h)) + f_size + 6 * w)])
                                elif f_size <18:
                                        if w==111 and f_size>=15:
                                            self.input.append(0)
                                        else:
                                            self.input.append(self.test_shape[((224*(6*h)+224*3)+(f_size%9)+6*w)])

                                else:
                                        if h == 111 or (w==111 and f_size>=24):
                                            self.input.append(0)


                                        else :
                                            self.input.append(self.test_shape[((224*(6*h)+224*6)+(f_size%9)+6*w)])

                                #if h==111 and w==111:
                                    #print(self.input)

                        self.sum.append(np.sum(np.array(self.filter)*np.array(self.input)))

                        self.filter = []


                    self.out=np.concatenate((self.out,self.sum))
                    self.input = []

                    self.sum=[]

                self.filter = []

            self.out = self.out.reshape(int(self.rI_H/self.stride), int(self.rI_W/self.stride), self.rF_T)

            print(np.array(self.out).shape)
            #print(self.out)
            self.test_shape=self.out.reshape(-1,)

            """
                        width = h % self.rI_W
                        height = int(h / self.rI_H) % self.rI_W
                        channel = int(h / (self.rI_W ** 2))
                        tensor = 0
                        target_address = 0 + self.rI_W * (
                                self.rI_H * (self.rI_C * tensor + channel) + height) + width
            """


        #112,112,32
        #3,3,32
        # dw
        elif self.dw==1 and self.pw==0 and self.pool==0 and self.fc==0:

            print("dw Convolution")
            print("stride :", self.stride)
            print(self.rI_W, self.rI_H, self.rI_C)
            print(self.rF_W, self.rF_H, self.rF_C, self.rF_T)
            print(self.All_data_num)

            if self.stride==1:
                for h in range(int(self.rI_H / self.stride)):
                    for w in range(int(self.rI_W / self.stride)):

                        for c in range(self.rF_C):
                            for f_size in range(self.rF_W * self.rF_H):
                                self.filter.append(self.test_filter_shape[self.rF_C * f_size+c])

                                if f_size <3:
                                        if h==0:
                                            self.input.append(0)
                                        elif (w==0 and f_size==0) or (w==self.rI_W-1 and f_size==2):
                                            self.input.append(0)

                                        else:
                                            #self.input.append(self.test_shape[(112*32*(h-1))+((32*f_size)+32*(w-1))+c])
                                            self.input.append(
                                                self.test_shape[(self.rI_W * self.rI_C * (h - 1)) + ((self.rI_C * f_size) + self.rI_C * (w - 1)) + c])

                                elif f_size <6:
                                            if (w==0 and f_size==3) or (w==self.rI_W-1 and f_size ==5):
                                                self.input.append(0)
                                            elif w==0:
                                                #self.input.append(self.test_shape[(112 * 32 * (h)) + (32 * ((f_size-1) % 3)+32*(w)) + c])
                                                self.input.append(self.test_shape[(self.rI_W * self.rI_C * (h)) + (
                                                                        self.rI_C * ((f_size - 1) % 3) + self.rI_C * (w)) + c])
                                            else:

                                                #self.input.append(self.test_shape[(112*32*(h)) + (32*(f_size%3)+32*(w-1))+c])
                                                self.input.append(self.test_shape[(self.rI_W * self.rI_C * (h)) + (
                                                           self.rI_C * (f_size % 3) + self.rI_C * (w - 1)) + c])
                                else:
                                            if h==self.rI_H-1:
                                                self.input.append(0)

                                            elif (w==0 and f_size==6) or (w==self.rI_W-1 and f_size==8):
                                                self.input.append(0)

                                            elif w==0:
                                                self.input.append(
                                                    #self.test_shape[(112 * 32 * (h + 1)) + (32 * ((f_size-1)%3)+32*(w)) + c])
                                                    self.test_shape[
                                                        (self.rI_W * self.rI_C * (h + 1)) + (self.rI_C * ((f_size - 1) % 3) + self.rI_C * (w)) + c])

                                            else :

                                                #self.input.append(self.test_shape[(112*32*(h+1)) + (32*((f_size)%3)+32*(w-1))+c])
                                                self.input.append(self.test_shape[(self.rI_W * self.rI_C * (h + 1)) + (
                                                            self.rI_C * ((f_size) % 3) + self.rI_C * (w - 1)) + c])


                            self.sum.append(np.sum(np.array(self.filter) * np.array(self.input)))
                            self.input = []
                            self.filter = []

                        self.out = np.concatenate((self.out, self.sum))
                        self.sum = []

                self.out = self.out.reshape(int(self.rI_H/self.stride), int(self.rI_W/self.stride), self.rF_C)

                print(np.array(self.out).shape)
                #print(self.out)
                self.test_shape=self.out.reshape(-1,)

            elif self.stride==2:
                for h in range(int(self.rI_H / self.stride)):
                    for w in range(int(self.rI_W / self.stride)):
                        for c in range(self.rF_C):
                            for f_size in range(self.rF_W * self.rF_H):
                                self.filter.append(self.test_filter_shape[self.rF_C * f_size + c])
                                if f_size<3:
                                    if w== int(self.rI_W / self.stride)-1 and f_size ==2:
                                        self.input.append(0)
                                    else:
                                        #self.input.append(self.test_shape[112*64*(2 * h)+( 64 * (2 * w))+(64 * f_size)+c])
                                        self.input.append(
                                            self.test_shape[self.rI_H * self.rI_C * (2 * h) + (self.rI_C * (2 * w)) + (self.rI_C * f_size) + c])
                                elif f_size < 6:
                                    if w== int(self.rI_W / self.stride)-1 and f_size ==5:
                                        self.input.append(0)
                                    else:
                                        #self.input.append(self.test_shape[112*64*(2 * h + 1) + (64 * (2 * w)) + (64 * (f_size%3))+c])
                                        self.input.append(self.test_shape[self.rI_H * self.rI_C * (2 * h + 1) + (self.rI_C * (2 * w)) + (
                                                    self.rI_C * (f_size % 3)) + c])
                                else:
                                    if w == int(self.rI_W / self.stride) - 1 and f_size == 8:
                                        self.input.append(0)
                                    elif h == int(self.rI_W / self.stride) -1:
                                        self.input.append(0)
                                    else:
                                        #self.input.append(self.test_shape[112*64*(2 * h + 2) + (64 * (2 * w)) + (64 * (f_size%3))+c])
                                        self.input.append(self.test_shape[self.rI_H * self.rI_C * (2 * h + 2) + (self.rI_C * (2 * w)) + (
                                                    self.rI_C * (f_size % 3)) + c])

                            self.sum.append(np.sum(np.array(self.filter) * np.array(self.input)))
                            self.input = []
                            self.filter = []

                        self.out = np.concatenate((self.out, self.sum))
                        self.sum = []

                self.out = self.out.reshape(int(self.rI_H/self.stride), int(self.rI_W/self.stride), self.rF_C)

                print(np.array(self.out).shape)
                #print(self.out)
                self.test_shape=self.out.reshape(-1,)


        #112,112,32
        #1,1,32,64
        #pw

        elif self.dw==0 and self.pw==1 and self.pool==0 and self.fc==0:
            print("pw Convolution")
            print("stride :", self.stride)
            print(self.rI_W, self.rI_H, self.rI_C)
            print(self.rF_W, self.rF_H, self.rF_C, self.rF_T)
            print(self.All_data_num)
            for h in range(int(self.rI_H / self.stride)):

                for w in range(int(self.rI_W / self.stride)):
                    for t in range(self.rF_T):
                        for c in range(self.rF_C):
                            self.filter.append(self.test_filter_shape[self.rF_T*c+t])
                            self.input.append(self.test_shape[(self.rI_W*self.rI_C*h)+self.rF_C*w+c])


                        self.sum.append(np.sum(np.array(self.filter) * np.array(self.input)))
                        self.input = []
                        self.filter = []

                    self.out = np.concatenate((self.out, self.sum))
                    self.sum = []
            self.out=self.out.reshape(self.rI_H, self.rI_W, self.rF_T)


            print(np.array(self.out).shape)
            #print(self.out)
            self.test_shape=self.out.reshape(-1,)


        #global average pooling
        # pooling의 경우 data를 1-D로 바꿀경우, 한자리당 1024개 순으로 저장 - 0 -0 1024 -1 2048-2 ....
        elif self.pool==1 and self.fc==0:

            print("pooling")
            print(self.rI_W, self.rI_H, self.rI_C)
            print(self.pool)
            print(self.All_data_num)
            for num in range(self.rI_C):
                for idx in range(self.rI_W * self.rI_H):
                    target_address = idx * 1024 + num

                    self.sum.append(self.test_shape[target_address])

                    if idx == 48:
                        cal = np.sum(self.sum) / 49
                        self.out.append(cal)
                        self.sum = []

            x = np.array(self.out)
            x = x.reshape(-1, 1, 1024)

            #print(x)
            print(x.shape)
            self.out = x
        #fully-connected
        # FC의경우 weight는 1000단위로 0,1,2,3 순으로 저장
        elif self.pool == 0 and self.fc == 1:


            print("Fully-connected")
            print(self.rI_W, self.rI_H, self.rI_C)
            print(self.rF_W, self.rF_H)
            print(self.rB_W, self.rB_H)
            self.test_shape = np.array(self.test_shape).reshape(-1,)
            self.test_filter_shape = np.array(self.test_filter_shape).reshape(-1,)
            for num in range(self.rF_H):
                for idx in range(self.rF_W):
                    target_address = 1000 * idx + num

                    CAL = self.test_shape[idx] * self.test_filter_shape[target_address]
                    self.sum.append(CAL)
                sum = np.sum(self.sum) + self.test_bias[num]
                self.out.append(sum)
                self.sum = []

            self.out=np.array(self.out)
            print(self.out)
            self.test_shape = self.out
        return self.out.reshape(-1,)

#convolution - general, Depth-wise, Point-wise Multiplication Accumulation
#pooling - GlobalaveragePooling                divide
#fully_connected - input * weight + bias       multiplication add
#np.set_printoptions(threshold=sys.maxsize)



'''
test =PE()

test.test_input = np.load(join('traffic_data.npy'))
test.test_shape = test.test_input.reshape(-1, )
test.test_filter_1 = np.load(join('mobilenet_conv1_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )


#general-Conv
test.CAL(224,3, 3,3,3,32,   0,0,    1,0,0,0,0)


"""---------------------------------------------------------"""
#depth,point conv -1
test.test_filter_1 = np.load(join('mobilenet_depth_conv_1_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(112,32,    3,3,32,1,   0,0,    0,1,0,0,0)
test.test_filter_1 = np.load(join('mobilenet_point_conv_1_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(112,32,    1,1,32,64,  0,0,    0,0,1,0,0)
"""---------------------------------------------------------"""


#depth,point conv -2
test.test_filter_1 = np.load(join('mobilenet_depth_conv_2_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(112,64,    3,3,64,1,   0,0,    1,1,0,0,0)
test.test_filter_1 = np.load(join('mobilenet_point_conv_2_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(56,64,    1,1,64,128,  0,0,    0,0,1,0,0)
#np.save('depth_3_input.npy',test.test_shape)




"""---------------------------------------------------------"""
#depth,point conv -3
test.test_input = np.load(join('depth_3_input.npy'))
test.test_shape = test.test_input.reshape(-1, )
test.test_filter_1 = np.load(join('mobilenet_depth_conv_3_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(56,128,    3,3,128,1,   0,0,    0,1,0,0,0)


test.test_filter_1 = np.load(join('mobilenet_point_conv_3_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(56,128,    1,1,128,128,  0,0,    0,0,1,0,0)

"""---------------------------------------------------------"""



#depth,point conv -4
test.test_filter_1 = np.load(join('mobilenet_depth_conv_4_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(56,128,    3,3,128,1,   0,0,    1,1,0,0,0)


test.test_filter_1 = np.load(join('mobilenet_point_conv_4_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(28,128,    1,1,128,256,  0,0,    0,0,1,0,0)
"""---------------------------------------------------------"""


#depth,point conv -5
test.test_filter_1 = np.load(join('mobilenet_depth_conv_5_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(28,256,    3,3,256,1,   0,0,    0,1,0,0,0)
test.test_filter_1 = np.load(join('mobilenet_point_conv_5_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(28,256,    1,1,256,256,  0,0,    0,0,1,0,0)
"""---------------------------------------------------------"""


#depth,point conv -6
test.test_filter_1 = np.load(join('mobilenet_depth_conv_6_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(28,256,    3,3,256,1,   0,0,    1,1,0,0,0)
test.test_filter_1 = np.load(join('mobilenet_point_conv_6_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(14,256,    1,1,256,512,  0,0,    0,0,1,0,0)
"""---------------------------------------------------------"""
#depth,point conv -7
test.test_filter_1 = np.load(join('mobilenet_depth_conv_7_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(14,512,    3,3,512,1,   0,0,    0,1,0,0,0)
test.test_filter_1 = np.load(join('mobilenet_point_conv_7_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(14,512,    1,1,512,512,  0,0,    0,0,1,0,0)
"""---------------------------------------------------------"""
#depth,point conv -8
test.test_filter_1 = np.load(join('mobilenet_depth_conv_8_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(14,512,    3,3,512,1,   0,0,    0,1,0,0,0)
test.test_filter_1 = np.load(join('mobilenet_point_conv_8_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(14,512,    1,1,512,512,  0,0,    0,0,1,0,0)
"""---------------------------------------------------------"""
#depth,point conv -9
test.test_filter_1 = np.load(join('mobilenet_depth_conv_9_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(14,512,    3,3,512,1,   0,0,    0,1,0,0,0)
test.test_filter_1 = np.load(join('mobilenet_point_conv_9_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(14,512,    1,1,512,512,  0,0,    0,0,1,0,0)
"""---------------------------------------------------------"""
#depth,point conv -10
test.test_filter_1 = np.load(join('mobilenet_depth_conv_10_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(14,512,    3,3,512,1,   0,0,    0,1,0,0,0)
test.test_filter_1 = np.load(join('mobilenet_point_conv_10_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(14,512,    1,1,512,512,  0,0,    0,0,1,0,0)
"""---------------------------------------------------------"""
#depth,point conv -11
test.test_filter_1 = np.load(join('mobilenet_depth_conv_11_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(14,512,    3,3,512,1,   0,0,    0,1,0,0,0)
test.test_filter_1 = np.load(join('mobilenet_point_conv_11_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(14,512,    1,1,512,512,  0,0,    0,0,1,0,0)
"""---------------------------------------------------------"""
#depth,point conv -12
test.test_filter_1 = np.load(join('mobilenet_depth_conv_12_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(14,512,    3,3,512,1,   0,0,    1,1,0,0,0)
test.test_filter_1 = np.load(join('mobilenet_point_conv_12_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(7,512,    1,1,512,1024,  0,0,    0,0,1,0,0)
"""---------------------------------------------------------"""
#depth,point conv -13
test.test_filter_1 = np.load(join('mobilenet_depth_conv_13_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(7,1024,    3,3,1024,1,   0,0,    0,1,0,0,0)
test.test_filter_1 = np.load(join('mobilenet_point_conv_13_weight.npy'))
test.test_filter_shape = test.test_filter_1.reshape(-1, )
test.CAL(7,1024,    1,1,1024,1024,  0,0,    0,0,1,0,0)
"""---------------------------------------------------------"""
#pooling
test.CAL(7,1024,    0,0,0,0,    0,0,        0,0,0,1,0)
"""---------------------------------------------------------"""
#fc
test.test_weight = np.load(join('mobilenet_predict_weight.npy'))
test.test_weight = np.array(test.test_weight).reshape(-1,)
test.test_bias = np.load(join('mobilenet_predict_bias.npy'))

test.CAL(1,1024,    1024,1000,0,0,  1,1000,     0,0,0,0,1)
"""---------------------------------------------------------"""




"""
    def start(self):
        for i in range(0,7):



    def insert(self, input, weight):
        self.activaiton = input
        self.weight = weight
"""
'''

