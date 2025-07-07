import concurrent.futures
import os
import cv2
import numpy as np
import glob
import math
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import label
from skimage import img_as_float

# SSIM
def compute_ssim(image1, image2):
    return ssim(image1, image2, data_range=1.0)

# 替换原 TC 的 Gabor 方法为 GLCM 对比度法
def compute_binary_tc(image):
    image = (image < 127).astype(np.uint8)

    # 水平方向相邻像素对
    row_vals = image[:, :-1].ravel()
    col_vals = image[:, 1:].ravel()

    # GLCM 统计：使用 bincount 实现计数，位置 = row * N + col
    levels = 2
    indices = row_vals * levels + col_vals
    counts = np.bincount(indices, minlength=levels * levels)
    glcm = counts.reshape((levels, levels)).astype(np.float64)

    # 保持对称（加上转置）
    glcm = (glcm + glcm.T) / 2.0
    glcm /= (glcm.sum() + 1e-6)

    # 计算对比度
    contrast = sum((i - j) ** 2 * glcm[i, j] for i in range(levels) for j in range(levels))
    tc = 1 / (1 + math.log(1 + contrast)) * 100
    return round(tc, 2)


# 替换原 LS 为线条连通性指数（LCI）
def compute_connectivity_index(image):
    binary_image = (image < 127).astype(int)  # 反转：线条为1
    total_pixels = np.sum(binary_image)
    if total_pixels == 0:
        return 0.0
    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_features = label(binary_image, structure=structure)
    counts = np.bincount(labeled_array.ravel())
    counts[0] = 0
    if len(counts) == 0:
        return 0.0
    largest_connected = np.max(counts)
    return round((largest_connected / total_pixels) * 100, 2)

# 主处理函数
def process_model_images(real_folder, model_folder):
    real_images = {os.path.basename(f): f for f in glob.glob(os.path.join(real_folder, "*.jpg"))}
    result_files = glob.glob(os.path.join(model_folder, "*.jpg"))

    ssim_scores = []
    tc_scores = []
    lci_scores = []

    for result_path in result_files:
        filename = os.path.basename(result_path)
        if filename not in real_images:
            continue

        real_img = cv2.imread(real_images[filename], cv2.IMREAD_GRAYSCALE)
        result_img = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
        if real_img is None or result_img is None:
            continue

        real_img = img_as_float(real_img)
        result_img = img_as_float(result_img)

        ssim_scores.append(compute_ssim(real_img, result_img))
        tc_scores.append(compute_binary_tc((result_img * 255).astype(np.uint8)))
        lci_scores.append(compute_connectivity_index((result_img * 255).astype(np.uint8)))

    return {
        "model_name": os.path.basename(model_folder),
        "ssim": np.mean(ssim_scores),
        "std_ssim": np.std(ssim_scores),
        "tc": np.mean(tc_scores),
        "std_tc": np.std(tc_scores),
        "lci": np.mean(lci_scores),
        "std_lci": np.std(lci_scores)
    }

# 处理所有模型
def process_all_models(real_folder, results_folder, output_file):
    model_folders = glob.glob(os.path.join(results_folder, "*"))
    results = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_model_images, real_folder, model_folder) for model_folder in model_folders]

        for future in concurrent.futures.as_completed(futures):
            model_results = future.result()
            result_lines = [
                f"Model: {model_results['model_name']}",
                f"SSIM: {model_results['ssim']:.4f} ± {model_results['std_ssim']:.4f}",
                f"TC: {model_results['tc']:.2f} ± {model_results['std_tc']:.2f}",
                f"LCI: {model_results['lci']:.2f} ± {model_results['std_lci']:.2f}",
                ""
            ]
            results.extend(result_lines)

            # ✅ 实时打印每个模型的结果
            print("\n".join(result_lines))


    with open(output_file, "w") as f:
        f.write("\n".join(results))

    print(f"Results saved to {output_file}")

# 路径设置
real_folder = "H:/Feng/duibi/dunhuang_fangfa/01inputs"
results_folder = "H:/Feng/duibi/dunhuang_fangfa"
output_file = "H:/Feng/duibi/results_dunhuang_fangfa.txt"

process_all_models(real_folder, results_folder, output_file)
