import os
import cv2
import numpy as np
def generate_heatmaps(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有图像文件
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            
            # 读取原始图像
            img = cv2.imread(image_path)
            
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray=255-gray

            # 生成原始影像的热力图
            heatmap_original = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

            # # **1. 创建自定义的颜色映射从白色到红色**
            # # 这里我们定义一个简单的颜色映射函数：灰度值（0到255）映射到从白色到红色的过渡
            # # 白色：[255, 255, 255], 红色：[255, 0, 0]
            # colored_map = np.zeros((256, 1, 3), dtype=np.uint8)
            # for i in range(256):
            #     colored_map[i] = [255 , 255 - i, 255-i]  # 从白色到红色的渐变
            
            # # 使用自定义颜色映射将灰度图转换为热力图
            # heatmap_original = cv2.applyColorMap(gray, cv2.COLORMAP_JET)  # 先创建一个Jet色图（用于测试），后面修改

            # # 使用自定义的颜色映射替换热力图颜色
            # for i in range(256):
            #     heatmap_original[gray == i] = colored_map[i]


            # 保存热力图到输出文件夹
            output_path_original = os.path.join(output_folder, filename)
            cv2.imwrite(output_path_original, heatmap_original)
            print(f"Saved heatmap: {output_path_original}")

# 示例使用
input_folder = 'Dunhuang/DunHuang2DunHuang_mask_1/o1'  # 替换为你的输入文件夹路径
output_folder = 'Dunhuang/DunHuang2DunHuang_mask_1/all_edges/01heatmaps_o1'  # 替换为你想保存热力图的文件夹路径
generate_heatmaps(input_folder, output_folder)
