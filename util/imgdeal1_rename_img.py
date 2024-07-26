# -*- coding:utf-8 -*-

import os
import random


class ImageRename():
    def __init__(self):
        self.path = r"F:\study\3_5programtest\8_bolt_change_v3\data_plastic\plastic_data_set_v4_Industrial_camera_6plastic\4_1080p_9PM_arealight1largellittle_lowst_f3.1_striplight4l_4s\ns_compar_yolo_usein_paper\buffer"  # 图片所在文件夹

    def rename(self):
        filelist = os.listdir(self.path)
        random.shuffle(filelist)
        total_num = len(filelist)

        i = 0

        for item in filelist:
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), format(str(i), '0>3s') + '.jpg')
                os.rename(src, dst)
                print('converting %s to %s ...' % (src, dst))
                i = i + 1
        print('total %d to rename & converted %d jpgs' % (total_num, i))


if __name__ == '__main__':
    newname = ImageRename()
    newname.rename()