import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# 输入和输出文件夹路径
input_folder = 'H:/Feng/duibi/all/01inputs'  # 替换为壁画图像文件夹路径
output_folder = 'opt/dataset/DunHuang/test_inputs_mask'  # 替换为输出文件夹路径

# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".tif"):  # 支持JPG和TIF格式
        # 构建图像路径
        image_path = os.path.join(input_folder, filename)
        
        # 读取原始图像
        img = cv2.imread(image_path)
        
        # **2. 在彩色图像上应用CLAHE（对比度受限自适应直方图均衡）**
        # 分离通道
        b, g, r = cv2.split(img)
        
        # 分别对每个通道应用CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # 调整参数以适应图像
        b_clahe = clahe.apply(b)
        g_clahe = clahe.apply(g)
        r_clahe = clahe.apply(r)
        
        # 合并处理后的通道
        clahe_img = cv2.merge([b_clahe, g_clahe, r_clahe])


        # **4. 在CLAHE处理后的图像上应用双边滤波**
        bilateral_clahe_img = cv2.bilateralFilter(clahe_img, d=7, sigmaColor=70, sigmaSpace=70)
        
        # 输出CLAHE + 双边滤波处理后的图像
        output_path_bilateral_clahe = os.path.join(output_folder, filename)
        cv2.imwrite(output_path_bilateral_clahe, bilateral_clahe_img)


        print(f"已处理并保存图像: {filename}")

print("所有图像已处理完成。")


##GRAY
# # 遍历输入文件夹中的所有图像文件
# for filename in os.listdir(input_folder):
#     if filename.endswith(".jpg") or filename.endswith(".tif"):  # 支持JPG和TIF格式
#         # 构建图像路径
#         image_path = os.path.join(input_folder, filename)
        
#         # 读取原始图像
#         img = cv2.imread(image_path)
        
#         # 转换为灰度图
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        

#         # **2. 应用 CLAHE（对比度受限自适应直方图均衡）**
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 调整参数以适应图像
#         clahe_img = clahe.apply(gray)


#         # **4. 再次应用双边滤波到 CLAHE 处理后的图像**
#         bilateral_clahe_img = cv2.bilateralFilter(clahe_img, d=6, sigmaColor=55, sigmaSpace=55)

#         # 生成 CLAHE + 双边滤波
#         output_path_bil_clahe = os.path.join(output_folder,   filename)
#         cv2.imwrite(output_path_bil_clahe, bilateral_clahe_img)

#         print(f"已处理并保存图像: {filename}")

# print("所有图像已处理完成。")

