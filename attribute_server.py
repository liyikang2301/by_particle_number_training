

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


#这个函数unpack_image用于从连接conn中解包图像数据。它首先接收数据，并根据图像数据的大小来确定接收完整个图像数据的步骤。
# 然后，它使用pickle模块来解压缩数据，并返回图像数据。
def unpack_image(conn):
    recv_data = b""
    data = b""
    print("unpack_image")
    payload_size = struct.calcsize(">l") #计算一个长整型数据的大小，用于确定每个数据包的大小。这个数据大小是固定的，但具体大小会因为机器的不同而不同。
    while len(data) < payload_size: #数据包的大小小于预期大小时，循环接收数据。
        # print ('payload_size')
        recv_data += conn.recv(4096) #从连接conn中接收数据，每次最多接收4096字节，并将其添加到接收到的数据中。
        # print (recv_data)
        if not recv_data: #检查接收到的数据是否为空，如果为空则返回None，表示接收失败。
            return None
        data += recv_data #将接收到的数据添加到已经接收到的数据中。
    packed_msg_size = data[:payload_size] #从接收到的数据中截取一个长整型数据，作为消息的大小。
    data = data[payload_size:] #将已经处理的数据从接收到的数据中删除，以便下次处理新的数据。
    msg_size = struct.unpack(">l", packed_msg_size)[0] #解析长整型数据，得到消息的大小。这里使用struct.unpack函数将字节串解析为长整型数据。
    if msg_size < 0: #检查消息大小是否小于零，如果是，则返回None，表示接收失败。
        return None
    print('unpack_image len(data): %d, msg_size %d' % (len(data), msg_size))
    while len(data) < msg_size: #在接收到的数据小于消息大小时，继续接收数据。
        data += conn.recv(4096) #继续从连接中接收数据，直到接收到足够的数据。

    frame_data = data[:msg_size] #从接收到的数据中截取出一段数据，这段数据就是图像数据。
    frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes") #使用pickle模块加载并解压缩图像数据，得到图像帧。
    # frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    # print('cv2')
    return frame #返回解压缩后的图像帧。
def infer_checkpoint(img):
    attribute_in_model = torch.load('./checkpoint/net_in_img_97.pkl',map_location=torch.device(device))
    attribute_out_model = torch.load('./checkpoint/net_out_img_95.pkl',map_location=torch.device(device))
    attribute_in_dic={0:'cross_groove',1:'hex_groove',2:'plane',3:'star_groove'}
    attribute_out_dic={0:'hex',1:'round',2:'other1',3:'other2'}
    attribute_in2index={'cross_groove':0,'hex_groove':1,'plane':2,'star_groove':3}
    attribute_out2index={'hex':0,'round':1,'other1':2,'other2':3}
    attribute_in=np.zeros((4))
    attribute_out=np.zeros((4))
    
    for i in range (4):
        in_op_list = [
            {'op': 'objects', 'param': ''},
            {'op':'filter_nearest_obj', 'param': ''},
            {'op':'obj_attibute', 'param':attribute_in2index[attribute_in_dic[i]]}
        ]
        out_op_list = [
            {'op': 'objects', 'param': ''},
            {'op':'filter_nearest_obj', 'param': ''},
            {'op':'obj_attibute', 'param':attribute_out2index[attribute_out_dic[i]]}
        ]
        img_file_path=""
        y_pred_in = attribute_in_model(in_op_list, img,img_file_path, mode='test')
        if y_pred_in[0].data==1:
            attribute_in[i]=1
        y_pred_out = attribute_out_model(out_op_list, img,img_file_path, mode='test')
        if y_pred_out[0].data==1:
            attribute_out[i]=1
    print("attribute_in:",attribute_in,"attribute_out:",attribute_out)
    return attribute_in,attribute_out

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ip_port = ('127.0.0.1', 5052)
server.bind(ip_port)
server.listen(5)
transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


while True:
    conn, addr = server.accept() #等待并接受客户端的连接请求，一旦有客户端连接到服务器，accept()方法就会返回一个新的socket对象conn，
    # 以及客户端的地址addr。这个socket对象用于与客户端进行通信。
    print(conn, addr)
    while True: #无限循环，用于持续接收客户端发送的图像数据，并返回处理结果。
        try: #尝试执行以下操作，如果出现异常则会捕获异常并执行对应的处理代码。
            frame = unpack_image(conn) #调用unpack_image函数从连接conn中解压缩图像数据，得到图像帧。
            if frame is None: #检查接收到的图像帧是否为空。如果为空，说明客户端请求停止发送图像数据，此时服务器会打印
                # "client request stop"并跳出内部循环，等待下一个客户端连接。
                print("client request stop")
                break
            
            frame_im = Image.fromarray(np.array(frame)) #将解压缩后的图像数据转换成PIL Image对象。
            # frame_im.show()
            print(frame_im.mode) #打印图像模式，包括灰度L，红绿蓝彩色图像RGB，带透明通道的红绿蓝RGBA，青品红黄黑四色图像CMYK
            # img = transform(frame_im)
            # img = torch.unsqueeze(img, 0)
            # img = img.to(device)
            attribute_in,attribute_out=infer_checkpoint(frame_im) #调用infer_checkpoint函数对图像进行属性推断，得到图像中螺栓的属性。
            # img_file="/home/xps/Desktop/ur10e_sim/src/fmauch_universal_robot/ur_real_robot/VAE_detect/true_mul_bolt_crops/cross_hex_bolt/0.jpg"
            # frame_im=Image.open(img_file)
            # print(frame_im.mode)
            # attribute_in,attribute_out=infer_checkpoint(frame_im)
            bolt_type=""
            if attribute_in[1]==1 and attribute_out[1]==1:
                bolt_type="in_hex_bolt"
                print("bolt_type:",bolt_type)
            elif attribute_in[3]==1 and attribute_out[1]==1:
                bolt_type="star_bolt"
                print("bolt_type:",bolt_type)
            elif attribute_in[0]==1 and attribute_out[0]==1:
                bolt_type="cross_hex_bolt"
                print("bolt_type:",bolt_type)
            elif attribute_in[2]==1 and attribute_out[0]==1:
                bolt_type="out_hex_bolt"
                print("bolt_type:",bolt_type)
            else:
                print("No matching bolt type")
            array_str = pickle.dumps(bolt_type, protocol=2) #使用pickle模块将螺栓类型bolt_type序列化为字节串。
            conn.sendall(array_str) #将序列化后的字节串通过连接conn发送给客户端。

        except ConnectionResetError as e:
            print('the connection is lost')
            break
    conn.close()
