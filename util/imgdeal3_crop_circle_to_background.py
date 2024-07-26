import cv2
import numpy as np
from PIL import Image, ImageDraw

def crop_to_circle(image_path, output_path):
    # 打开图像
    image = Image.open(image_path)

    # 获取图像尺寸
    width, height = image.size

    # 计算圆形的半径
    radius = min(width, height) // 2

    # 创建一个与图像相同大小的透明图像
    circle_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # 将图像裁剪为圆形
    mask = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(mask)
    degree = 5
    mask_draw.ellipse((degree, degree, width-degree, height-degree), fill=255)
    image.putalpha(mask)

    # 将裁剪后的图像粘贴到透明图像中
    circle_image.paste(image, (0, 0), image)

    # 保存结果
    # circle_image.save(output_path)

    background = Image.open(background_path)

    # 获取图像尺寸
    bg_width, bg_height = background.size
    fg_width, fg_height = circle_image.size

    # 计算前景图像的中心坐标
    fg_center_x = bg_width // 2
    fg_center_y = bg_height // 2

    # 计算前景图像的左上角坐标，使其中心与背景图像中心重合
    fg_left = fg_center_x - fg_width // 2
    fg_upper = fg_center_y - fg_height // 2

    # 将前景图像叠加到背景图像上
    background.paste(circle_image, (fg_left, fg_upper), circle_image)

    crop_width = (bg_width-width) // 2
    crop_hight = (bg_height-height) //2
    background = background.crop((crop_width+degree-1, crop_hight+degree-1, crop_width+width-degree+2, crop_hight+height-degree+2))
    background = np.array(background)
    # 保存结果
    # background.save(output_path)
    # print(f"已保存合并后的图像到 {output_path}")
    return background

# 示例用法
if __name__ == '__main__':
    in_path = r"F:\study\3_5programtest\8_bolt_change_v3\data_plastic\plastic_data_set_v3\crop_circle_to_background_test\org_img/"
    out_path = r"F:\study\3_5programtest\8_bolt_change_v3\data_plastic\plastic_data_set_v3\crop_circle_to_background_test\changed_img/"
    plastic_type = 'ppblack'
    chg_list = list(range(0,257))

    #以下部分为形状属性训练的增强变换
    for chg_id in chg_list:
        img_path = in_path +  str(chg_id).zfill(3) + '.jpg'
        background_path = r"F:\study\3_5programtest\8_bolt_change_v3\data_plastic\plastic_data_set_v3\crop_circle_to_background_test\background.jpg"
        background_img = crop_to_circle(img_path, out_path)
        cv2.imwrite(out_path + plastic_type + str(chg_id) + '_invert' + '.jpg', background_img)
        # img_path = r"F:\study\3_5programtest\8_bolt_change_v3\data_plastic\data_augmentation_test\ppwhite19_invert.jpg"
    #
    #     invert_img = invert(img_path)
    #     cv2.imwrite(out_path + plastic_type + str(chg_id) + '_invert' + '.jpg', invert_img)


# input_image_path = r"F:\study\3_5programtest\8_bolt_change_v3\data_plastic\plastic_data_set_v3\crop_circle_to_background_test\234.jpg"  # 替换为您的图像路径
# output_image_path = r"F:\study\3_5programtest\8_bolt_change_v3\data_plastic\plastic_data_set_v3\crop_circle_to_background_test/output_circle6.png"  # 替换为输出图像的路径
# background_path = r"F:\study\3_5programtest\8_bolt_change_v3\data_plastic\plastic_data_set_v3\crop_circle_to_background_test\background.jpg"
# crop_to_circle(input_image_path, output_image_path)