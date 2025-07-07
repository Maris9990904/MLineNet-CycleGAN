import os
import cv2
import numpy as np

def save_colored_comparison(real_labels_folder, pred_labels_folder, output_folder):
    """
    读取真实标签和推理结果，并生成对比图：
    - **绿色（TP）**: 预测边缘在真实边缘1个像素范围内
    - **蓝色（FP）**: 预测为边缘，但不在真实边缘1个像素范围内（误检）
    - **黑色（FN）**: 真实边缘未被正确预测（但1像素范围内有预测则不算）
    - **白色（TN）**: 背景正确预测

    处理后图像保存到 output_folder。
    """
    os.makedirs(output_folder, exist_ok=True)
    
    kernel = np.ones((3, 3), np.uint8)  # 3x3核膨胀真实边缘，允许1像素误差范围

    for filename in os.listdir(pred_labels_folder):
        real_file_path = os.path.join(real_labels_folder, filename[:-4] + '.png')
        pred_file_path = os.path.join(pred_labels_folder, filename)
        output_file_path = os.path.join(output_folder, filename)

        # 读取灰度图像
        real_img = cv2.imread(real_file_path, cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.imread(pred_file_path, cv2.IMREAD_GRAYSCALE)

        if real_img is None or pred_img is None:
            print(f"Error: Could not read {real_file_path} or {pred_file_path}")
            continue

        # **二值化真实标签（黑色=1, 白色=0）**
        real_bin = (real_img < 128).astype(np.uint8)
        # **膨胀真实边缘，使其覆盖1个像素的范围**
        real_dilated = cv2.dilate(real_bin, kernel, iterations=1)

        # **归一化预测图（转换到 [0,1]）并二值化（黑色=1, 白色=0）**
        pred_bin = (1 - (pred_img.astype(np.float32) / 255.0)) > 0.5
        pred_bin = pred_bin.astype(np.uint8)
        pre_dilated = cv2.dilate(pred_bin, kernel, iterations=1)



        # **创建彩色图像（默认白色）**
        colored_img = np.ones((*real_bin.shape, 3), dtype=np.uint8) * 255

        # **正确映射颜色**
        tp_mask = (real_dilated == 1) & (pred_bin == 1)  # 真实边缘1像素内有预测
        fp_mask = (real_dilated == 0) & (pred_bin == 1)  # 误检
        fn_mask = (real_bin == 1) & (pred_bin == 0) & (pre_dilated == 0)  # 真实边缘未被预测，且1像素内无预测
        tn_mask = (real_bin == 0) & (pred_bin == 0)  # 背景正确预测
        true_positives=(np.sum(tp_mask))
        false_positives=(np.sum(fp_mask))
        false_negatives=(np.sum(fn_mask))
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1=2 * (precision * recall) / (precision + recall + 1e-8)
        print(f1)
        colored_img[tp_mask] = [0, 255, 0]  # **绿色 (TP)**
        colored_img[fp_mask] = [0, 0, 255]  # **蓝色 (FP)**
        colored_img[fn_mask] = [0, 0, 0]    # **黑色 (FN)**
        colored_img[tn_mask] = [255, 255, 255]  # **白色 (TN)**

        # **保存结果**
        cv2.imwrite(output_file_path, colored_img)
    
    print(f"Processed {len(os.listdir(pred_labels_folder))} images. Saved to {output_folder}.")

# 示例路径（请替换为你的实际路径）
real_labels_folder = "opt/dataset/BIPED/edgev1_01"  # 真实标签
pred_labels_folder = "result/BIPED2BIPED/fused"  # 预测结果
output_folder = "result/BIPED2BIPED/colored_comparison"  # 结果保存路径

# 执行可视化比较
save_colored_comparison(real_labels_folder, pred_labels_folder, output_folder)
