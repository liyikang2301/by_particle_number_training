import cv2
import numpy as np
from PIL import Image ,ImageEnhance

def convert_white_to_black(image_path):
    # 打开图片
    white_img = Image.open(image_path)
    # 获取图片的宽度和高度
    img_np = np.asarray(white_img)
    width, height = white_img.size
    # 遍历每个像素点
    for x in range(width):
        for y in range(height):
            # 获取像素点的RGB值
            r, g, b = white_img.getpixel((x, y))
            # 判断是否为白色像素点（RGB值为255, 255, 255）
            if r == 255 and g == 255 and b == 255:
                # 将白色像素点变为黑色（RGB值为0, 0, 0）
                white_img.putpixel((x, y), (0, 0, 0))
    # 保存修改后的图片
    white_img = np.array(white_img)
    # img.save("output_image.png")
    return white_img

def decrease_brightness(image_path):
    factor = 0.25 #取值范围为0到1，0表示完全黑暗，1表示原始亮度
    # 打开图片
    img = Image.open(image_path)
    # 创建图像增强对象
    enhancer = ImageEnhance.Brightness(img)
    # 降低亮度
    img_darkened = enhancer.enhance(factor)
    # 保存处理后的图片
    # img_darkened.save("darkened_image.png")
    img_darkened = np.array(img_darkened)
    return img_darkened


if __name__ == '__main__':
    in_path = r"F:\study\3_5programtest\8_bolt_change_v3\data_plastic\plastic_data_set_v2\ppblack_black_and_irregularity/"
    out_path = r"F:\study\3_5programtest\8_bolt_change_v3\data_plastic\data_augmentation_test/"
    plastic_type = 'ppblack'
    chg_list = list(range(0,280))

    for chg_id in chg_list:
        img_path = in_path +  str(chg_id).zfill(3) + '.jpg'

        # scale_crop_image = scale_crop(img_path)
        # cv2.imwrite(out_path + plastic_type + str(chg_id) + '_scale_crop' + '.jpg', scale_crop_image)

        white_image = decrease_brightness(img_path)
        cv2.imwrite(out_path + plastic_type + str(chg_id) + '_random_crop' + '.jpg', white_image)