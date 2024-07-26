import os
import pickle
import socket
import cv2
import struct
import numpy as np
import random
from PIL import Image
import select
import sys
import os
import random
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


import torch
from torchvision import datasets
import torchvision.transforms as transforms



def infer_checkpoint(img,img_path):
    # attribute_colour_model = torch.load(r"F:\study\3_5programtest\8_bolt_change_v3\data_plastic\plastic_data_set_v4_Industrial_camera_6plastic\6_1080p_9PM_coverlight1large1smalllittle_top4cmtolenns_f3.1_strip4L3S\colour_data_aug_200transet\checkpoint\model_aug_fourcolour_1.0_4neg.pkl", map_location=torch.device(device))
    # attribute_shape_model = torch.load(r"F:\study\3_5programtest\8_bolt_change_v3\data_plastic\plastic_data_set_v4_Industrial_camera_6plastic\6_1080p_9PM_coverlight1large1smalllittle_top4cmtolenns_f3.1_strip4L3S\shape_data_aug_200transet\checkpoint\model_shape_0.99444_4neg.pkl", map_location=torch.device(device))
    attribute_colour_dic = {0: 'transwhite', 1: 'red', 2: 'black', 3: 'white'}
    attribute_shape_dic = {0: 'irregularity', 1: 'square', 2: 'other1', 3: 'other2'}
    attribute_colour2index = {'transwhite': 0, 'red': 1, 'black': 2, 'white': 3}
    attribute_shape2index = {'irregularity': 0, 'square': 1, 'other1': 2, 'other2': 3}
    attribute_colour = np.zeros((4))
    attribute_shape = np.zeros((4))

    for i in range(4):
        colour_op_list = [
            {'op': 'objects', 'param': ''},
            {'op': 'filter_nearest_obj', 'param': ''},
            {'op': 'obj_attibute', 'param': [0,attribute_colour2index[attribute_colour_dic[i]]]}
        ]
        shape_op_list = [
            {'op': 'objects', 'param': ''},
            {'op': 'filter_nearest_obj', 'param': ''},
            {'op': 'obj_attibute', 'param': [1,attribute_shape2index[attribute_shape_dic[i]]]}
        ]
        img_file_path = img_path
        y_pred_in = attribute_colour_model(colour_op_list, img, img_file_path, mode='test')
        if y_pred_in[0].data == 1:
            attribute_colour[i] = 1
        y_pred_out = attribute_shape_model(shape_op_list, img, img_file_path, mode='test')
        if y_pred_out[0].data == 1:
            attribute_shape[i] = 1
    attribute_colour_model.concept_matrix2zero()  # 将模型的 concept_matrix 置零。
    attribute_shape_model.concept_matrix2zero()  # 将模型的 concept_matrix 置零。
    print("attribute_in:", attribute_colour, "attribute_out:", attribute_shape,str(img_id))
    return attribute_colour, attribute_shape

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


file_path = r'F:\study\3_5programtest\8_bolt_change_v3\data_plastic\plastic_data_set_v4_Industrial_camera_6plastic\4_1080p_9PM_arealight1largellittle_lowst_f3.1_striplight4l_4s\ns_compar_yolo_usein_paper\abswhite_buffer/'
img_numb = len(os.listdir(file_path))
img_list = list(range(0,img_numb))

PVC_PU_num = 0
PAred_num = 0
PPblack_num = 0
ABSblack_num = 0
PE_num = 0
PPwhite_ABSwhite_num = 0

attribute_colour_model = torch.load(r"F:\study\3_5programtest\8_bolt_change_v3\data_plastic\plastic_data_set_v4_Industrial_camera_6plastic\4_1080p_9PM_arealight1largellittle_lowst_f3.1_striplight4l_4s\colour_data_aug_200transet\checkpoint\model_aug_fourcolour_1.0_4neg.pkl", map_location=torch.device(device))
attribute_shape_model = torch.load(r"F:\study\3_5programtest\8_bolt_change_v3\data_plastic\plastic_data_set_v4_Industrial_camera_6plastic\4_1080p_9PM_arealight1largellittle_lowst_f3.1_striplight4l_4s\shape_data_aug_200transet\checkpoint\model_shape_0.99444_4neg.pkl", map_location=torch.device(device))

for img_id in img_list:
    img_path = file_path + str(img_id).zfill(3) + '.jpg'
    reason_img = Image.open(img_path)
    # frame_im = Image.fromarray(np.array(frame)) #将解压缩后的图像数据转换成PIL Image对象。
    # frame_im.show()
    # print(reason_img.mode) #打印图像模式，包括灰度L，红绿蓝彩色图像RGB，带透明通道的红绿蓝RGBA，青品红黄黑四色图像CMYK
    # img = transform(frame_im)
    # img = torch.unsqueeze(img, 0)
    # img = img.to(device)
    attribute_colour,attribute_shape=infer_checkpoint(reason_img,img_path) #调用infer_checkpoint函数对图像进行属性推断，得到图像中螺栓的属性。
    # img_file="/home/xps/Desktop/ur10e_sim/src/fmauch_universal_robot/ur_real_robot/VAE_detect/true_mul_bolt_crops/cross_hex_bolt/0.jpg"
    # frame_im=Image.open(img_file)
    # print(frame_im.mode)
    # attribute_in,attribute_out=infer_checkpoint(frame_im)
    bolt_type=""
    if attribute_colour[0]==1 and attribute_shape[0]==1: #透白&不规则
        bolt_type="PVC or PU"
        PVC_PU_num = PVC_PU_num+1
        # print("plastic_type:",bolt_type)
    elif attribute_colour[1]==1 and attribute_shape[1]==1: #红&方形
        bolt_type="PAred"
        PAred_num = PAred_num+1
        # print("plastic_type:",bolt_type)
    elif attribute_colour[2]==1 and attribute_shape[0]==1: #黑&不规则
        bolt_type="PPblack"
        PPblack_num = PPblack_num+1
        # print("plastic_type:",bolt_type)
    elif attribute_colour[2]==1 and attribute_shape[1]==1: #黑&方
        bolt_type="ABSblack"
        ABSblack_num = ABSblack_num+1
        # print("plastic_type:",bolt_type)
    elif attribute_colour[3]==1 and attribute_shape[0]==1: #白&不规则
        bolt_type="PE"
        PE_num = PE_num+1
        # print("plastic_type:",bolt_type)
    elif attribute_colour[3]==1 and attribute_shape[1]==1: #白&方
        bolt_type="PPwhite or ABSwhite"
        PPwhite_ABSwhite_num = PPwhite_ABSwhite_num+1
        # print("plastic_type:",bolt_type)
    else:
        print("No matching plastic_type"+str(img_id))
    print(bolt_type)

print('PVC_PU_num='+str(PVC_PU_num),'PAred_num='+str(PAred_num),'PPblack_num='+str(PPblack_num),'ABSblack_num='+str(ABSblack_num),'PE_num='+str(PE_num),'PPwhite_ABSwhite_num='+str(PPwhite_ABSwhite_num))
# g1_PE_purity = (PE_num*0.02735)/(PE_num*0.02735+PPwhite_ABSwhite_num*0.0306+PAred_num*0.027435)
# g1_PE_re = PE_num/230
# print('组1的pe纯度为：'+str(g1_PE_purity) +  '\n' + '组1的pe回收率为：'+ str(g1_PE_re))
# g1_ppwhite_purty = (PPwhite_ABSwhite_num*0.0306)/(PPwhite_ABSwhite_num*0.0306+PE_num*0.02735+PAred_num*0.027435)
# g1_ppwhite_re= PPwhite_ABSwhite_num/200
# print('组1的ppwhite纯度为：'+str(g1_ppwhite_purty) + '\n' +'组1的ppwhite回收率为：'+ str(g1_ppwhite_re))
# g1_pared_purty = (PAred_num*0.027435)/(PAred_num*0.027435+PE_num*0.02735+PPwhite_ABSwhite_num*0.0306)
# g1_pared_re = PAred_num/230
# print('组1的pared纯度为：'+str(g1_pared_purty) + '\n' +'组1的pared回收率为：'+ str(g1_pared_re))
# g2_absblack_purty = (ABSblack_num*0.01239)/(ABSblack_num*0.01239+PPblack_num*0.0203509+PAred_num*0.027435)
# g2_absblack_re = ABSblack_num/285
# print('组2的absblack纯度为：'+str(g2_absblack_purty) + '\n' +'组2的absblack回收率为：'+ str(g2_absblack_re))
# g2_ppblack_purty = (PPblack_num*0.0203509)/(PPblack_num*0.0203509+ABSblack_num*0.01239+PAred_num*0.027435)
# g2_ppblack_re = PPblack_num/285
# print('组2的ppblack纯度为：'+str(g2_ppblack_purty) + '\n' +'组2的ppblack回收率为：'+ str(g2_ppblack_re))
# g2_pared_purty = (PAred_num*0.027435)/(PAred_num*0.027435+ABSblack_num*0.01239+PPblack_num*0.0203509)
# g2_pared_re = PAred_num/230
# print('组2的pared纯度为：'+str(g2_pared_purty) + '\n' +'组2的pared回收率为：'+ str(g2_pared_re))
g3_abswhite_purty = (PPwhite_ABSwhite_num*0.015498)/(PPwhite_ABSwhite_num*0.015498+PVC_PU_num*0.015259+PAred_num*0.027435)
g3_abswhite_re = PPwhite_ABSwhite_num/231
print('组3的abswhite纯度为：'+str(g3_abswhite_purty) + '\n' +'组3的abswhite回收率为：'+ str(g3_abswhite_re))
# g3_pu_purty = (PVC_PU_num*0.01526)/(PVC_PU_num*0.01526+PPwhite_ABSwhite_num*0.015498+PAred_num*0.027435)
# g3_pu_re = PVC_PU_num/232
# print('组3的pu纯度为：'+str(g3_pu_purty) + '\n' + '组3的pu回收率为：'+str(g3_pu_re))
# g3_pared_purty = (PAred_num*0.027435)/(PAred_num*0.027435+PVC_PU_num*0.01526+PPwhite_ABSwhite_num*0.015498)
# g3_pared_re = PAred_num/230
# print('组3的pared纯度为：'+str(g3_pared_purty) + '\n' + '组3的pared回收率为：'+str(g3_pared_re))
# g4_pe_purty = (PE_num*0.02735)/(PE_num*0.02735+PPwhite_ABSwhite_num*0.0306+PVC_PU_num*0.023296)
# g4_pe_re = PE_num/230
# print('组4的pe纯度为：'+str(g4_pe_purty) + '\n' + '组4的pe回收率为：'+str(g4_pe_re))
# g4_ppwhite_purty = (PPwhite_ABSwhite_num*0.0306)/(PPwhite_ABSwhite_num*0.0306+PE_num*0.02735+PVC_PU_num*0.023296)
# g4_ppwhite_re = PPwhite_ABSwhite_num/200
# print('组4的ppwhite纯度为：'+str(g4_ppwhite_purty) + '\n' +'组4的ppwhite回收率为：'+ str(g4_ppwhite_re))
# g4_pvc_purty = (PVC_PU_num*0.023296)/(PVC_PU_num*0.023296+PE_num*0.02735+PPwhite_ABSwhite_num*0.0306)
# g4_pvc_re = PVC_PU_num/267
# print('组4的pvc纯度为：'+str(g4_pvc_purty) + '\n' + '组4的pvc回收率为：'+str(g4_pvc_re))