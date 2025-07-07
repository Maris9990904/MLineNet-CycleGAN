import os
import cv2
import numpy as np
from scipy.ndimage import label
from PIL import Image

# 设置路径
input_folder = "H:/Feng/tmp"
output_folder = "H:/Feng/tmp/tmp_LT"
os.makedirs(output_folder, exist_ok=True)

def compute_connectivity_index(binary_image, min_region_ratio=0.01):
    """
    改进版 LCI，支持单个区域：统计多个足够大的连通区域占比
    """
    total_pixels = np.sum(binary_image)
    if total_pixels == 0:
        return 0.0

    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_features = label(binary_image, structure=structure)
    counts = np.bincount(labeled_array.ravel())
    counts[0] = 0

    threshold = total_pixels * min_region_ratio
    large_regions = counts[counts >= threshold]
    connected_pixels = np.sum(large_regions)

    return connected_pixels / total_pixels

def compute_lci_gridwise(binary_image, grid_size=9):
    """
    将图像切分为 grid_size × grid_size 区域，分别计算 LCI 并求平均
    """
    h, w = binary_image.shape
    h_splits = np.array_split(binary_image, grid_size, axis=0)
    lci_list = []

    for row in h_splits:
        row_splits = np.array_split(row, grid_size, axis=1)
        for cell in row_splits:
            lci = compute_connectivity_index(cell)
            lci_list.append(lci)

    return np.mean(lci_list)

# 遍历图像
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
        image_path = os.path.join(input_folder, filename)
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY_INV)

        # 使用 6x6 划分方式计算 LCI
        ci = compute_lci_gridwise(binary)
        ci_percent = round(ci * 100, 2)

        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_CI{ci_percent:.2f}{ext}"
        new_path = os.path.join(output_folder, new_filename)

        Image.fromarray(image).save(new_path)

        print(f"{filename} → 连通指数 (6x6 平均): {ci_percent:.2f}%，保存为：{new_filename}")
