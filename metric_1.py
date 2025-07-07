import os
import cv2
import numpy as np
import glob
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import sobel, gaussian_filter
from skimage.filters import gabor
from skimage import img_as_float

# 计算结构相似性 SSIM
def compute_ssim(image1, image2):
    return ssim(image1, image2, data_range=255)

# 计算梯度相似性 GS
def compute_gradient_similarity(image1, image2):
    grad1_x = sobel(image1, axis=0)
    grad1_y = sobel(image1, axis=1)
    grad2_x = sobel(image2, axis=0)
    grad2_y = sobel(image2, axis=1)
    
    grad1 = np.hypot(grad1_x, grad1_y)
    grad2 = np.hypot(grad2_x, grad2_y)
    
    gs = np.sum((grad1 * grad2)) / (np.sqrt(np.sum(grad1 ** 2)) * np.sqrt(np.sum(grad2 ** 2)) + 1e-10)
    return gs

# 计算纹理复杂度 TC（基于 Gabor 滤波）
def compute_texture_complexity(image):
    _, gabor_response = gabor(image, frequency=0.2)
    return np.mean(np.abs(gabor_response))

# 计算线条流畅性 LS（基于曲率变化）
# 计算线条流畅性 LS（基于曲率变化）
def compute_line_smoothness(image):
    edges = (image == 0).astype(np.uint8)  # 壁画边缘像素值为 0
    smoothed = gaussian_filter(edges.astype(np.float32), sigma=1)
    
    # 确保输入类型为适合的整数类型，避免类型不兼容
    smoothed = np.uint8(smoothed * 255)  # 转换为 uint8 类型并放缩回 0-255 范围
    laplacian = cv2.Laplacian(smoothed, cv2.CV_64F)
    return np.mean(np.abs(laplacian))


# 处理所有图片并计算指标
def process_images(real_folder, results_folder, output_file):
    # 获取真实影像的路径
    real_images = {os.path.basename(f): f for f in glob.glob(os.path.join(real_folder, "*.jpg"))}
    
    # 获取模型文件夹（如a、b、c等）
    model_folders = glob.glob(os.path.join(results_folder, "*"))
    
    results = []

    # 遍历每个模型文件夹
    for model_folder in model_folders:
        model_name = os.path.basename(model_folder)
        result_files = glob.glob(os.path.join(model_folder, "*.jpg"))  # 获取模型推理结果文件
        
        # 遍历每个模型的推理结果
        for result_path in result_files:
            filename = os.path.basename(result_path)
            
            if filename not in real_images:
                print(f"Warning: No matching real image for {filename}")
                continue

            # 读取真实图像和模型结果
            real_img = cv2.imread(real_images[filename], cv2.IMREAD_GRAYSCALE)
            result_img = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)

            if real_img is None or result_img is None:
                print(f"Error reading image: {filename}")
                continue

            # 归一化图像
            real_img = img_as_float(real_img)
            result_img = img_as_float(result_img)

            # 计算各项指标
            ssim_score = compute_ssim(real_img, result_img)
            gs_score = compute_gradient_similarity(real_img, result_img)
            tc_score = compute_texture_complexity(result_img)
            ls_score = compute_line_smoothness(result_img)

            # 将结果保存为一行
            results.append(f"Model: {model_name}, {filename} SSIM: {ssim_score:.4f}, GS: {gs_score:.4f}, TC: {tc_score:.4f}, LS: {ls_score:.4f}")

    # 保存结果到txt文件
    with open(output_file, "w") as f:
        f.write("\n".join(results))
    
    print(f"Results saved to {output_file}")

# 运行
real_folder = "Dunhuang/denoise/01inputs"  # 真实影像文件夹路径
results_folder = "Dunhuang/denoise/all"  # 模型结果文件夹路径
output_file = "Dunhuang/denoise/results.txt"

process_images(real_folder, results_folder, output_file)
