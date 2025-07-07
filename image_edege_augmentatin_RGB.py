import cv2
import numpy as np
import os

# 输入和输出文件夹路径
input_folder = 'H:/Feng/duibi/all/01inputs'  # 替换为壁画图像文件夹路径
output_folder = 'H:/Feng/duibi/all/zeqiang'  # 替换为输出文件夹路径

# 创建三个独立的输出子文件夹
original_heatmap_folder = os.path.join(output_folder, "original_heatmaps")
clahe_heatmap_folder = os.path.join(output_folder, "clahe_heatmaps")
clahe_bf_heatmap_folder = os.path.join(output_folder, "clahe_bf_heatmaps")

os.makedirs(original_heatmap_folder, exist_ok=True)
os.makedirs(clahe_heatmap_folder, exist_ok=True)
os.makedirs(clahe_bf_heatmap_folder, exist_ok=True)

# 遍历输入文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".tif"):  # 仅处理JPG和TIF格式
        image_path = os.path.join(input_folder, filename)
        img = cv2.imread(image_path)

        # 定义一个函数来生成反转的热力图
        def generate_reversed_heatmap(image, colormap=cv2.COLORMAP_JET):
            heatmap = cv2.applyColorMap(image, colormap)
            reversed_heatmap = cv2.bitwise_not(heatmap)  # 反转颜色条
            return reversed_heatmap

        # 1. 生成 **原始影像热力图**（颜色条反转）
        heatmap_original_reversed = generate_reversed_heatmap(img)
        output_path_original_heatmap = os.path.join(original_heatmap_folder, filename)
        cv2.imwrite(output_path_original_heatmap, heatmap_original_reversed)

        # 2. **CLAHE 处理**
        b, g, r = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        b_clahe, g_clahe, r_clahe = clahe.apply(b), clahe.apply(g), clahe.apply(r)
        clahe_img = cv2.merge([b_clahe, g_clahe, r_clahe])

        # CLAHE **热力图**（颜色条反转）
        heatmap_clahe_reversed = generate_reversed_heatmap(clahe_img)
        output_path_clahe_heatmap = os.path.join(clahe_heatmap_folder, filename)
        cv2.imwrite(output_path_clahe_heatmap, heatmap_clahe_reversed)

        # 3. **CLAHE + 双边滤波处理**
        bilateral_clahe_img = cv2.bilateralFilter(clahe_img, d=5, sigmaColor=50, sigmaSpace=50)
        heatmap_bilateral_clahe_reversed = generate_reversed_heatmap(bilateral_clahe_img)
        output_path_bilateral_clahe_heatmap = os.path.join(clahe_bf_heatmap_folder, filename)
        cv2.imwrite(output_path_bilateral_clahe_heatmap, heatmap_bilateral_clahe_reversed)

        print(f"已处理并保存影像: {filename}")

print("所有影像已处理完成。")
