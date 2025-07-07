import cv2
import numpy as np
import os

# 输入 & 输出文件夹
input_folder = "H:/Feng/duibi/all/tmp"
output_folder = "H:/Feng/duibi/all/xihua2"

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def skeletonize_image(image_path, output_path):
    """ 处理单个图像并保存骨架化结果 """
    # 读取灰度图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 反转颜色（确保线条是白色，背景是黑色）
    inverted = cv2.bitwise_not(image)

    # 二值化处理
    # _, binary = cv2.threshold(inverted, 50, 255, cv2.THRESH_BINARY)
    binary = cv2.adaptiveThreshold(255-inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 10)  # 细节更多
    
    cv2.imwrite(output_path, binary)

# 遍历输入文件夹中的所有图片
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        skeletonize_image(input_path, output_path)

print(f"/n✅ 处理完成！所有图像已保存至 {output_folder}")
