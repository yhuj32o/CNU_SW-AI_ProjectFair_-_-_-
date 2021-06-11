
import numpy as np
from os.path import join

class D_memory:
    # D_memory 초기, weight, input 불러오기
    def __init__(self):

        self.input_data = 'Memory/input_folder/traffic_data.npy'
        self.input_shape = np.load(join(self.input_data))
        self.input_shape_reshape = self.input_shape.reshape(-1,)

        self.conv1_weight = np.load(join('Memory/weight_folder/mobilenet_conv1_weight.npy'))
        self.conv1_weight_reshape = self.conv1_weight.reshape(-1,)

        self.depth_conv_1_weight = np.load(join('Memory/weight_folder/mobilenet_depth_conv_1_weight.npy'))
        self.point_conv_1_weight = np.load(join('Memory/weight_folder/mobilenet_point_conv_1_weight.npy'))
        self.depth_conv_1_weight_reshape = self.depth_conv_1_weight.reshape(-1, )
        self.point_conv_1_weight_reshape = self.point_conv_1_weight.reshape(-1, )

        self.depth_conv_2_weight = np.load(join('Memory/weight_folder/mobilenet_depth_conv_2_weight.npy'))
        self.point_conv_2_weight = np.load(join('Memory/weight_folder/mobilenet_point_conv_2_weight.npy'))
        self.depth_conv_2_weight_reshape = self.depth_conv_2_weight.reshape(-1,)
        self.point_conv_2_weight_reshape = self.point_conv_2_weight.reshape(-1,)

        self.depth_conv_3_weight = np.load(join('Memory/weight_folder/mobilenet_depth_conv_3_weight.npy'))
        self.point_conv_3_weight = np.load(join('Memory/weight_folder/mobilenet_point_conv_3_weight.npy'))
        self.depth_conv_3_weight_reshape = self.depth_conv_3_weight.reshape(-1, )
        self.point_conv_3_weight_reshape = self.point_conv_3_weight.reshape(-1, )

        self.depth_conv_4_weight = np.load(join('Memory/weight_folder/mobilenet_depth_conv_4_weight.npy'))
        self.point_conv_4_weight = np.load(join('Memory/weight_folder/mobilenet_point_conv_4_weight.npy'))
        self.depth_conv_4_weight_reshape = self.depth_conv_4_weight.reshape(-1, )
        self.point_conv_4_weight_reshape = self.point_conv_4_weight.reshape(-1, )


        self.depth_conv_5_weight = np.load(join('Memory/weight_folder/mobilenet_depth_conv_5_weight.npy'))
        self.point_conv_5_weight = np.load(join('Memory/weight_folder/mobilenet_point_conv_5_weight.npy'))
        self.depth_conv_5_weight_reshape = self.depth_conv_5_weight.reshape(-1, )
        self.point_conv_5_weight_reshape = self.point_conv_5_weight.reshape(-1, )

        self.depth_conv_6_weight = np.load(join('Memory/weight_folder/mobilenet_depth_conv_6_weight.npy'))
        self.point_conv_6_weight = np.load(join('Memory/weight_folder/mobilenet_point_conv_6_weight.npy'))
        self.depth_conv_6_weight_reshape = self.depth_conv_6_weight.reshape(-1, )
        self.point_conv_6_weight_reshape = self.point_conv_6_weight.reshape(-1, )

        self.depth_conv_7_weight = np.load(join('Memory/weight_folder/mobilenet_depth_conv_7_weight.npy'))
        self.point_conv_7_weight = np.load(join('Memory/weight_folder/mobilenet_point_conv_7_weight.npy'))
        self.depth_conv_7_weight_reshape = self.depth_conv_7_weight.reshape(-1, )
        self.point_conv_7_weight_reshape = self.point_conv_7_weight.reshape(-1, )

        self.depth_conv_8_weight = np.load(join('Memory/weight_folder/mobilenet_depth_conv_8_weight.npy'))
        self.point_conv_8_weight = np.load(join('Memory/weight_folder/mobilenet_point_conv_8_weight.npy'))
        self.depth_conv_8_weight_reshape = self.depth_conv_8_weight.reshape(-1, )
        self.point_conv_8_weight_reshape = self.point_conv_8_weight.reshape(-1, )

        self.depth_conv_9_weight = np.load(join('Memory/weight_folder/mobilenet_depth_conv_9_weight.npy'))
        self.point_conv_9_weight = np.load(join('Memory/weight_folder/mobilenet_point_conv_9_weight.npy'))
        self.depth_conv_9_weight_reshape = self.depth_conv_9_weight.reshape(-1, )
        self.point_conv_9_weight_reshape = self.point_conv_9_weight.reshape(-1, )

        self.depth_conv_10_weight = np.load(join('Memory/weight_folder/mobilenet_depth_conv_10_weight.npy'))
        self.point_conv_10_weight = np.load(join('Memory/weight_folder/mobilenet_point_conv_10_weight.npy'))
        self.depth_conv_10_weight_reshape = self.depth_conv_10_weight.reshape(-1, )
        self.point_conv_10_weight_reshape = self.point_conv_10_weight.reshape(-1, )

        self.depth_conv_11_weight = np.load(join('Memory/weight_folder/mobilenet_depth_conv_11_weight.npy'))
        self.point_conv_11_weight = np.load(join('Memory/weight_folder/mobilenet_point_conv_11_weight.npy'))
        self.depth_conv_11_weight_reshape = self.depth_conv_11_weight.reshape(-1, )
        self.point_conv_11_weight_reshape = self.point_conv_11_weight.reshape(-1, )

        self.depth_conv_12_weight = np.load(join('Memory/weight_folder/mobilenet_depth_conv_12_weight.npy'))
        self.point_conv_12_weight = np.load(join('Memory/weight_folder/mobilenet_point_conv_12_weight.npy'))
        self.depth_conv_12_weight_reshape = self.depth_conv_12_weight.reshape(-1, )
        self.point_conv_12_weight_reshape = self.point_conv_12_weight.reshape(-1, )
        self.depth_conv_13_weight = np.load(join('Memory/weight_folder/mobilenet_depth_conv_13_weight.npy'))
        self.point_conv_13_weight = np.load(join('Memory/weight_folder/mobilenet_point_conv_13_weight.npy'))
        self.depth_conv_13_weight_reshape = self.depth_conv_13_weight.reshape(-1, )
        self.point_conv_13_weight_reshape = self.point_conv_13_weight.reshape(-1, )

        self.predict_weight = np.load(join('Memory/weight_folder/mobilenet_predict_weight.npy'))
        self.predict_weight_reshape = self.predict_weight.reshape(-1,)
        self.predict_bias = np.load(join('Memory/weight_folder/mobilenet_predict_bias.npy'))
        self.predict_bias_reshape=self.predict_bias.reshape(-1,)

        # storage -> 메모리 할당 필요
        # 메모리 1D로 변환
        #self.storage=np.zeros(shape=(8388608,))
        self.storage=np.concatenate([self.input_shape_reshape,self.conv1_weight_reshape,self.depth_conv_1_weight_reshape,self.point_conv_1_weight_reshape,
                                     self.depth_conv_2_weight_reshape,self.point_conv_2_weight_reshape,self.depth_conv_3_weight_reshape,self.point_conv_3_weight_reshape,
                                     self.depth_conv_4_weight_reshape,self.point_conv_4_weight_reshape,self.depth_conv_5_weight_reshape,self.point_conv_5_weight_reshape,
                                     self.depth_conv_6_weight_reshape,self.point_conv_6_weight_reshape,self.depth_conv_7_weight_reshape,self.point_conv_7_weight_reshape,
                                     self.depth_conv_8_weight_reshape,self.point_conv_8_weight_reshape,self.depth_conv_9_weight_reshape,self.point_conv_9_weight_reshape,
                                     self.depth_conv_10_weight_reshape,self.point_conv_10_weight_reshape,self.depth_conv_11_weight_reshape,self.point_conv_11_weight_reshape,
                                     self.depth_conv_12_weight_reshape,self.point_conv_12_weight_reshape,self.depth_conv_13_weight_reshape,self.point_conv_13_weight_reshape,
                                     self.predict_weight_reshape,self.predict_bias_reshape])

        #filters_length = [864, 288, 2048, 576, 8192, 1152, 16384, 1152, 32768, 2304, 65536, 2304, 131072,
        #                 4608, 262144, 4608, 262144, 4608, 262144, 4608, 262144, 4608, 262144,
        #                 4608, 524288, 9216, 1048576, 1024000, 1000]



    # D_mem -> Global buffer 이동  type= 0 - input , type=1 - filter
    # number -> type=1 일 경우,
    # 0 - general
    # 1,3,5,7,9,11,13,15,17,19,21,23,25 - dw_1,2,3,4,5,6,7,8,9,10,11,12,13
    # 2,4,6,8,10,12,14,16,18,20,22,24,26   - pw_1,2,3,4,5,6,7,8,9,10,11,12,13
    # 27 - predict_weight , 28 - predict_bias

    def upper(self,x,y,RF_rD,RF_rW_H,RF_rC,RF_rT):
        self.RF_rD = int(RF_rD)
        self.RF_rW_H = int(RF_rW_H)
        self.RF_rC = int(RF_rC)
        self.RF_rT = int(RF_rT)
        self.All_data_num = self.RF_rW_H ** 2 * self.RF_rC * self.RF_rT

        if self.All_data_num == 1048576 :
            print(self.RF_rD)
            print(self.All_data_num)
        return self.storage[self.RF_rD:self.RF_rD+self.All_data_num]


