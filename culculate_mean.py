import os
import json
from PIL import Image
import numpy as np

# 影像列表txt文件路径（存放影像的相对路径）
image_list_txt = 'opt/dataset/BIPED/train_pair.lst'
# 影像保存的文件夹
image_folder = 'opt/dataset/BIPED'

# 初始化通道均值
mean_r, mean_g, mean_b = 0.0, 0.0, 0.0
count = 0

# 读取txt文件中的影像相对路径
with open(image_list_txt, 'r') as f:
    image_data = json.load(f)

# 遍历影像并计算均值
for image_pair in image_data:
    if image_pair:
        image_path = os.path.join(image_folder, image_pair[0])
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image, dtype=np.float32)
            
            # 计算每个通道的均值
            mean_r += np.mean(image_np[:, :, 0])
            mean_g += np.mean(image_np[:, :, 1])
            mean_b += np.mean(image_np[:, :, 2])
            
            count += 1
        else:
            print(f"Warning: {image_path} not found.")

# 计算最终均值
if count > 0:
    mean_r /= count
    mean_g /= count
    mean_b /= count

# 输出均值
mean_values = (round(mean_r, 3), round(mean_g, 3), round(mean_b, 3))
print(f"Mean (R, G, B): {mean_values}")

