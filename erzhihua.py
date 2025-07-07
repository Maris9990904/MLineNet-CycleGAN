import os
import cv2
import numpy as np

def process_images_in_folder(input_folder, output_folder, threshold_factor=0.99):
    """读取文件夹中的图像，并根据阈值将图像小于某个值的像素设为黑色，其余设为白色"""
    
    # 计算阈值
    threshold = int(255 *  threshold_factor)
    print(f"Threshold set to: {threshold}")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有图像
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        
        # 只处理图像文件
        if os.path.isfile(file_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            print(f"Processing file: {file_name}")
            
            # 读取图像
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
            
            if image is None:
                print(f"Failed to read {file_name}. Skipping.")
                continue
            
            # 创建二值化图像
            binary_image = np.where(image < threshold, 0, 255).astype(np.uint8)
            
            # 保存处理后的图像
            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, binary_image)
            print(f"Processed image saved to: {output_path}")

# 示例用法Dunhuang\10TEED+aug+Gan
input_folder = 'Dadingge/teed+agu+gan'  # 输入文件夹路径
output_folder = 'Dadingge/tmp'  # 输出文件夹路径

process_images_in_folder(input_folder, output_folder)
