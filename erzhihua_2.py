import cv2
import os
import numpy as np
# 输入 & 输出文件夹
input_folder = "H:/Feng/duibi/all/xihua"  # 替换为你的输入文件夹路径
output_folder = "H:/Feng/duibi/all/xihua_binary"  # 输出文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有图片
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 读取灰度图像
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # 处理图像：像素值 > 128 的设为 255，≤ 128 的设为 0
        binary_image = np.where(image>128,255,0)
        # 保存处理后的图像
        cv2.imwrite(output_path, binary_image)
        print(f"✅ 处理完成: {input_path} → {output_path}")

print(f"\n🎯 所有图像已处理完毕，保存至: {output_folder}")
