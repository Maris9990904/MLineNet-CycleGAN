import os
from PIL import Image
import numpy as np

# 定义读取和处理文件夹中的图像的函数
def invert_images_in_folder(folder_path):
    # 遍历文件夹中的每个文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 只处理图像文件（可以根据需要调整条件）
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                # 打开图像
                img = Image.open(file_path)

                # 转换为灰度图像（如果是彩色图像）
                img_gray = img.convert("L")

                # 将图像转换为 numpy 数组
                img_array = np.array(img_gray)

                # 反转颜色：黑色变白，白色变黑
                img_inverted = 255 - img_array

                # 将反转后的数组转换回图像
                img_inverted = Image.fromarray(img_inverted)

                # 保存图像，名称保持不变
                img_inverted.save(file_path)
                print(f"Processed and saved: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# 设置你的文件夹路径
folder_path = "results_Dunhuang_clahe_gus"  # 替换为你的文件夹路径
invert_images_in_folder(folder_path)
