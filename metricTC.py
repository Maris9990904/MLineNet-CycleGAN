import os
import math
import cv2
import numpy as np
from pathlib import Path

def compute_binary_tc(image):
    """专用于二值图像的 TC 计算：levels=2"""
    levels = 2
    glcm = np.zeros((levels, levels), dtype=np.float64)

    # 将图像从 [0, 255] 转为 [0, 1]
    image = (image < 127).astype(int)

    # 水平邻接像素构建 GLCM（可扩展为多个方向）
    for i in range(image.shape[0]):
        for j in range(image.shape[1] - 1):
            row = image[i, j]
            col = image[i, j + 1]
            glcm[row, col] += 1
            glcm[col, row] += 1  # 保持对称

    glcm /= (glcm.sum() + 1e-6)  # 归一化

    # TC公式：1 / (1 + log(1 + sum((i-j)^2 * P(i,j)))) * 100
    contrast = sum((i - j) ** 2 * glcm[i, j] for i in range(levels) for j in range(levels))
    tc = 1 / (1 + math.log(1 + contrast)) * 100
    return round(tc, 2)

def process_binary_folder(input_folder, output_folder):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for file in input_path.glob("*.*"):
        if file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            tc = compute_binary_tc(img)
            new_filename = f"{file.stem}_TC{tc:.2f}{file.suffix}"
            cv2.imwrite(str(output_path / new_filename), img)

    print(f"处理完成，结果保存到：{output_path.resolve()}")
process_binary_folder("H:/Feng/tmp", "H:/Feng/tmp/tmp_TC")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 构造两个4x4的GLCM矩阵
# 高TC：对角线主导，表示结构简单、对比度低
glcm_high_tc_4x4 = np.array([
    [30, 2, 1, 0],
    [2, 25, 2, 1],
    [1, 2, 20, 2],
    [0, 1, 2, 15]
], dtype=float)
glcm_high_tc_4x4 /= glcm_high_tc_4x4.sum()

# 低TC：远离对角线的差值更大，结构混杂、对比度高
glcm_low_tc_4x4 = np.array([
    [1, 2, 8, 15],
    [2, 5, 10, 8],
    [8, 10, 5, 2],
    [15, 8, 2, 1]
], dtype=float)
glcm_low_tc_4x4 /= glcm_low_tc_4x4.sum()

# TC计算函数
def compute_tc(glcm):
    return 1 / (1 + np.log(1 + sum((i - j)**2 * glcm[i, j] for i in range(4) for j in range(4)))) * 100

tc_high_4x4 = compute_tc(glcm_high_tc_4x4)
tc_low_4x4 = compute_tc(glcm_low_tc_4x4)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(glcm_high_tc_4x4, annot=True,annot_kws={"size": 20}, cmap="Blues", square=True, ax=axes[0], cbar=False, fmt=".2f")
axes[0].set_title(f"高TC矩阵（TC = {tc_high_4x4:.2f}%）")

sns.heatmap(glcm_low_tc_4x4, annot=True, cmap="Blues",annot_kws={"size": 20}, square=True, ax=axes[1], cbar=False, fmt=".2f")
axes[1].set_title(f"低TC矩阵（TC = {tc_low_4x4:.2f}%）")

plt.tight_layout()
plt.show()
