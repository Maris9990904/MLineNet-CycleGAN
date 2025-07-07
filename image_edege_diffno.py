import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# 输入和输出文件夹路径
input_folder = '/mnt/sdb1/fenghaixia/DexiNed-master/opt/dataset/BIPED_baojidian/inputs'  # 替换为壁画图像文件夹路径
output_folder = '/mnt/sdb1/fenghaixia/DexiNed-master/opt/dataset/BIPED_baojidian/inputs_mask'  # 替换为输出文件夹路径

# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".tif"):  # 支持JPG和PNG格式
        # 构建图像路径
        image_path = os.path.join(input_folder, filename)
        
        # 读取图像
        img = cv2.imread(image_path)
        
        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        
        # 应用双边滤波，增强边缘
        bilateral_img = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        
        # 转换回彩色图像（如果需要保存为彩色图像）
        output_img = cv2.cvtColor(bilateral_img, cv2.COLOR_GRAY2BGR)
        
        # 保存增强后的图像
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, output_img)
        
        print(f"已处理并保存图像: {filename}")

print("所有图像已处理完成。")
