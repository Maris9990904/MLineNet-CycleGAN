



import cv2
import numpy as np
import os
from skimage.morphology import skeletonize
import imageio

# 输入和输出文件夹路径import cv2
import numpy as np
import os
from skimage.morphology import skeletonize
import imageio

# 输入和输出文件夹路径
input_folder = "H:/Feng/duibi/all_4/07TEED+aug+Gan"  # 修改为你的输入文件夹路径
output_folder = "H:/Feng/duibi/houchuli"  # 修改为你的输出文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 处理所有图片
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  
        image_path = os.path.join(input_folder, filename)

        # 读取灰度图（修正 as_gray=True 的错误）
        image = imageio.imread(image_path, mode='L')  

        # 归一化到 [0,1] 并骨架化（细化线条）
        binary = image / 255.0
        thinned = skeletonize(binary)

        # 还原到 0-255 并保存
        thinned = (thinned * 255).astype(np.uint8)
        output_path = os.path.join(output_folder, filename)
        imageio.imwrite(output_path, thinned)

        print(f"已处理: {filename} -> {output_path}")

print("所有图片处理完成！")

input_folder = "H:/Feng/duibi/all_4/07TEED+aug+Gan"  # 修改为你的输入文件夹路径
output_folder = "H:/Feng/duibi/houchuli"  # 修改为你的输出文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 处理所有图片
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  
        image_path = os.path.join(input_folder, filename)
        image = imageio.imread(image_path, as_gray=True)  # 读取灰度图

        # 归一化到 [0,1] 并骨架化（细化线条）
        binary = image / 255.0
        thinned = skeletonize(binary)

        # 还原到 0-255 并保存
        thinned = (thinned * 255).astype(np.uint8)
        output_path = os.path.join(output_folder, filename)
        imageio.imwrite(output_path, thinned)

        print(f"已处理: {filename} -> {output_path}")

print("所有图片处理完成！")

