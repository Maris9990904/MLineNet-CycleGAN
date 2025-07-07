import concurrent.futures
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
def compute_line_smoothness(image):
    edges = (image == 0).astype(np.uint8)  # 壁画边缘像素值为 0
    smoothed = gaussian_filter(edges.astype(np.float32), sigma=1)
    
    # 确保输入类型为适合的整数类型，避免类型不兼容
    smoothed = np.uint8(smoothed * 255)  # 转换为 uint8 类型并放缩回 0-255 范围
    laplacian = cv2.Laplacian(smoothed, cv2.CV_64F)
    return np.mean(np.abs(laplacian))

# 处理每个模型的所有图像并计算综合指标
def process_model_images(real_folder, model_folder):
    real_images = {os.path.basename(f): f for f in glob.glob(os.path.join(real_folder, "*.jpg"))}
    
    # 获取该模型的所有推理结果文件
    result_files = glob.glob(os.path.join(model_folder, "*.jpg"))
    
    # 记录每个指标的所有值
    ssim_scores = []
    gs_scores = []
    tc_scores = []
    ls_scores = []

    for result_path in result_files:
        filename = os.path.basename(result_path)
        
        if filename not in real_images:
            continue

        # 读取真实图像和模型结果
        real_img = cv2.imread(real_images[filename], cv2.IMREAD_GRAYSCALE)
        result_img = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)

        # 归一化图像
        real_img = img_as_float(real_img)
        result_img = img_as_float(result_img)

        # 计算各项指标
        ssim_scores.append(compute_ssim(real_img, result_img))
        gs_scores.append(compute_gradient_similarity(real_img, result_img))
        tc_scores.append(compute_texture_complexity(result_img))
        ls_scores.append(compute_line_smoothness(result_img))

    # 计算平均值
    avg_ssim = np.mean(ssim_scores)
    avg_gs = np.mean(gs_scores)
    avg_tc = np.mean(tc_scores)
    avg_ls = np.mean(ls_scores)

    # 计算标准差（可选）
    std_ssim = np.std(ssim_scores)
    std_gs = np.std(gs_scores)
    std_tc = np.std(tc_scores)
    std_ls = np.std(ls_scores)

    return {
        "model_name": os.path.basename(model_folder),
        "ssim": avg_ssim,
        "gs": avg_gs,
        "tc": avg_tc,
        "ls": avg_ls,
        "std_ssim": std_ssim,
        "std_gs": std_gs,
        "std_tc": std_tc,
        "std_ls": std_ls
    }

# 处理所有模型
def process_all_models(real_folder, results_folder, output_file):
    model_folders = glob.glob(os.path.join(results_folder, "*"))
    
    # 清空输出文件
    with open(output_file, "w") as f:
        f.write("Model Evaluation Results/n")
    
    results = []
    
    # 使用并行处理模型
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_model_images, real_folder, model_folder) for model_folder in model_folders]
        
        for future in concurrent.futures.as_completed(futures):
            model_results = future.result()
            results.append(f"Model: {model_results['model_name']}")
            results.append(f"SSIM: {model_results['ssim']:.4f} ± {model_results['std_ssim']:.4f}")
            results.append(f"GS: {model_results['gs']:.4f} ± {model_results['std_gs']:.4f}")
            results.append(f"TC: {model_results['tc']:.4f} ± {model_results['std_tc']:.4f}")
            results.append(f"LS: {model_results['ls']:.4f} ± {model_results['std_ls']:.4f}")
            results.append("/n")
    
    # 将结果保存到txt文件
    with open(output_file, "w") as f:
        f.write("/n".join(results))
    
    print(f"Results saved to {output_file}")

# 运行
real_folder = "H:/Feng/duibi/dadingge_all/01inputs"  # 真实影像文件夹路径
results_folder = "H:/Feng/duibi/dadingge_all/all"  # 模型结果文件夹路径
output_file = "H:/Feng/duibi/dadingge_all/results.txt"

process_all_models(real_folder, results_folder, output_file)
