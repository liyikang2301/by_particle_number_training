# from PIL import Image, ImageChops

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageEnhance,ImageChops, ImageDraw
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import os
import random


#像素值反转
def invert(img_path):
    img = Image.open(img_path)
    inv_img = ImageChops.invert(img)
    # inv_img.show()
    inv_img = np.array(inv_img)
    return inv_img

#色彩抖动
def randomColor(img_path):
    # 随机生成0，1来随机确定调整哪个参数，可能会调整饱和度，也可能会调整图像的饱和度和亮度
    image = Image.open(img_path)
    saturation = random.randint(0, 1)
    brightness = random.randint(0, 1)
    contrast = random.randint(0, 1)
    sharpness = random.randint(0, 1)

    # 当三个参数中一个参数为1，就可执行相应的操作
    if random.random() < saturation:
        random_factor = np.random.randint(0, 40) / 10.  # 随机因子
        image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    if random.random() < brightness:
        random_factor = np.random.randint(0, 40) / 10.  # 随机因子
        image = ImageEnhance.Brightness(image).enhance(random_factor)  # 调整图像的亮度
    if random.random() < contrast:
        random_factor = np.random.randint(0, 40) / 10.  # 随机因1子
        image = ImageEnhance.Contrast(image).enhance(random_factor)  # 调整图像对比度
    if random.random() < sharpness:
        random_factor = np.random.randint(0, 41) / 10.  # 随机因子
        image=ImageEnhance.Sharpness(image).enhance(random_factor)  # 调整图像锐度
    # image.show()
    image = np.array(image)
    return image


# 限制对比度自适应直方图均衡
def clahe(img_path):
    image = cv2.imread(img_path)
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    image_clahe = cv2.merge([b, g, r])
    # image_clahe = Image.fromarray(image_clahe.astype('uint8')).convert('RGB')
    # image_clahe.show()
    return image_clahe


# 伽马变换
def gamma(img_path):
    image = cv2.imread(img_path)
    fgamma = random.randint(0, 2)
    image_gamma = np.uint8(np.power((np.array(image) / 255.0), fgamma) * 255.0)
    cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
    cv2.convertScaleAbs(image_gamma, image_gamma)
    # image_gamma = Image.fromarray(image_gamma.astype('uint8')).convert('RGB')
    # image_gamma.show()
    return image_gamma


# 直方图均衡
def hist(img_path):
    image = cv2.imread(img_path)
    r, g, b = cv2.split(image)
    r1 = cv2.equalizeHist(r)
    g1 = cv2.equalizeHist(g)
    b1 = cv2.equalizeHist(b)
    image_equal_clo = cv2.merge([r1, g1, b1])
    # image_equal_clo = Image.fromarray(image_equal_clo.astype('uint8')).convert('RGB')
    # image_equal_clo.show()
    return image_equal_clo

#向外缩放然后裁剪
def scale_crop(img_path):
    image = cv2.imread(img_path)
    original_height, original_width = image.shape[:2]
    min_scale = 1.3
    max_scale = 1.5
    # 随机生成一个缩放比例
    scale_factor = np.random.uniform(min_scale, max_scale)
    # 缩放图像
    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    start_row = (scaled_image.shape[0] - original_height) // 2
    start_col = (scaled_image.shape[1] - original_width) // 2

    # 裁剪图像
    scale_crop_image = scaled_image[start_row:start_row + original_height, start_col:start_col + original_width]

    return scale_crop_image

#随机裁剪
def random_crop(img_path):
    image = cv2.imread(img_path)
    # 获取原始图片的大小
    original_height, original_width = image.shape[:2]
    min_crop_size = (20,20)
    max_crop_size = image.shape

    # 随机生成裁剪区域的大小
    crop_height = np.random.randint(min_crop_size[0], max_crop_size[0] -10 + 1)
    crop_width = np.random.randint(min_crop_size[1], max_crop_size[1] -10 + 1)

    # 随机生成裁剪区域的起始坐标
    start_row = np.random.randint(0, original_height - crop_height + 1)
    start_col = np.random.randint(0, original_width - crop_width + 1)

    # 执行裁剪操作
    cropped_image = image[start_row:start_row + crop_height, start_col:start_col + crop_width]
    #start_row:start_row + crop_height：这个部分表示沿着行的切片操作，start_row 是裁剪区域的起始行坐标，
    # start_row + crop_height 是裁剪区域的结束行坐标（不包含该行）。因此，这个切片操作实际上选取了从 start_row 开始，
    # 到 start_row + crop_height - 1 结束的行。
    #start_col:start_col + crop_width：这个部分表示沿着列的切片操作

    return cropped_image


#随机拉伸长宽
def random_distort(img_path, distortion_range=(0.5, 2.0)):
    # 获取图像的高度和宽度
    image = cv2.imread(img_path)
    height, width = image.shape[:2]

    # 随机生成拉伸因子
    stretch_factor_x = np.random.uniform(*distortion_range)
    stretch_factor_y = np.random.uniform(*distortion_range)

    # 对图像进行拉伸
    distorted_image = cv2.resize(image, (0, 0), fx=stretch_factor_x, fy=stretch_factor_y,
                                 interpolation=cv2.INTER_LINEAR)
    #表示将图像尺寸在宽与高方向上根据随机缩放因子进行缩放
    return distorted_image

def mirror_image(input_image_path):
    # 使用 OpenCV 读取图像
    image = cv2.imread(input_image_path)

    # 水平镜像
    mirrored_image = cv2.flip(image, 1)

    return mirrored_image




if __name__ == '__main__':
    in_path = r"F:\study\3_5programtest\8_bolt_change_v3\data_plastic\plastic_data_set_v4_Industrial_camera_6plastic\4_1080p_9PM_arealight1largellittle_lowst_f3.1_striplight4l_4s\morelight_g2_colouraug200\ppblack_black_and_irregularity/"
    out_path = r"F:\study\3_5programtest\8_bolt_change_v3\data_plastic\plastic_data_set_v4_Industrial_camera_6plastic\4_1080p_9PM_arealight1largellittle_lowst_f3.1_striplight4l_4s\morelight_g2_shapeaug200\buffer/"
    plastic_type = 'ppblack'
    chg_list = list(range(0,200))

    #以下部分为形状属性训练的增强变换
    # for chg_id in chg_list:
    #     img_path = in_path +  str(chg_id).zfill(3) + '.jpg'
    #     # img_path = r"F:\study\3_5programtest\8_bolt_change_v3\data_plastic\data_augmentation_test\ppwhite19_invert.jpg"
    #
    #     invert_img = invert(img_path)
    #     cv2.imwrite(out_path + plastic_type + str(chg_id) + '_invert' + '.jpg', invert_img)
    #
    #     randomColor_img = randomColor(img_path)
    #     cv2.imwrite(out_path + plastic_type + str(chg_id) + '_randomColor' + '.jpg', randomColor_img)
    #
    #     clahe_img = clahe(img_path)
    #     cv2.imwrite(out_path + plastic_type + str(chg_id) + '_clahe' + '.jpg', clahe_img)
    #
    #     gamma_img = gamma(img_path)
    #     cv2.imwrite(out_path + plastic_type + str(chg_id) + '_gamma' + '.jpg', gamma_img)
    #
    #     hist_img = hist(img_path)
    #     cv2.imwrite(out_path + plastic_type + str(chg_id) + '_hist' + '.jpg', hist_img)


    #以下部分为颜色属性训练的增强变换
    for chg_id in chg_list:
        img_path = in_path +  str(chg_id).zfill(3) + '.jpg'

        scale_crop_image = scale_crop(img_path)
        cv2.imwrite(out_path + plastic_type + str(chg_id) + '_scale_crop' + '.jpg', scale_crop_image)

        # mirror_image_image = mirror_image(img_path)
        # cv2.imwrite(out_path + plastic_type + str(chg_id) + '_mirror' + '.jpg', mirror_image_image)
    #
    #     random_crop_image = random_crop(img_path)
    #     cv2.imwrite(out_path + plastic_type + str(chg_id) + '_random_crop' + '.jpg', random_crop_image)
    #
    #     random_distort_image = random_distort(img_path)
    #     cv2.imwrite(out_path + plastic_type + str(chg_id) + '_random_distort' + '.jpg', random_distort_image)