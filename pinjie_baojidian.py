import os
import cv2
import numpy as np
from osgeo import gdal

def get_image_dimensions(image_path):
    """获取图像的宽度和高度"""
    dataset = gdal.Open(image_path)
    if dataset is None:
        raise FileNotFoundError(f"Unable to open image: {image_path}")
    original_width = dataset.RasterXSize
    original_height = dataset.RasterYSize
    return original_width, original_height

def stitch_tiles(tiles_folder, original_image_path, tile_size=512, stride=256, file_format='jpg', output_folder='output/'):
    """将多个模型的裁剪图块拼接回原始大图，并根据模型名称保存结果"""
    
    # 获取原始图像尺寸
    original_width, original_height = 4568,3332
    # 计算需要多少图块
    x_tiles = (original_width + stride - 1) // stride
    y_tiles = (original_height + stride - 1) // stride
    
    # 初始化字典，用于存储每个模型的拼接结果
    models = {}

    for model_name in os.listdir(tiles_folder):
        model_folder = os.path.join(tiles_folder, model_name)
        
        if not os.path.isdir(model_folder):
            continue
        
        print(f"Processing model: {model_name}")
        
        # 初始化大图
        stitched_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)
        
        count = 0
        for i in range(y_tiles):
            for j in range(x_tiles):
                # 计算当前图块的起始坐标
                x_start = j * stride
                y_start = i * stride
                x_end = min(x_start + stride, original_width)
                y_end = min(y_start + stride, original_height)
                
                # 读取图块
                tile_path = os.path.join(model_folder, f"{count:04d}.{file_format}")
                tile = cv2.imread(tile_path)
                
                # 如果没有读取到图像，跳过该图块
                if tile is None:
                    print(f"Tile {tile_path} not found. Skipping.")
                    count += 1
                    continue
                
                # 只保留512x512图块的中间部分（256x256）
                mid_tile = tile[128:384, 128:384]  # 从512x512裁剪中间256x256部分
                
                # 如果图块在边缘，可能需要调整拼接位置
                if mid_tile.shape[0] != (y_end - y_start) or mid_tile.shape[1] != (x_end - x_start):
                    mid_tile = mid_tile[:y_end - y_start, :x_end - x_start]
                
                # 将中间256x256的部分放回原始图像中
                stitched_image[y_start:y_end, x_start:x_end, :] = mid_tile
                count += 1

        # 裁剪到原始尺寸
        if stitched_image.shape[1] > original_width or stitched_image.shape[0] > original_height:
            stitched_image = stitched_image[:original_height, :original_width]
            print(f"Image cropped to original size: {original_width}x{original_height}")

        # 保存拼接后的图像
        output_path = os.path.join(output_folder, f"{model_name}_edges_fuse.jpg")
        cv2.imwrite(output_path, stitched_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"Stitched image for {model_name} saved to {output_path}")
        
        # 存储每个模型的拼接结果
        models[model_name] = stitched_image

    return models

# 示例用法
tiles_folder = 'Baojidian'  # 包含多个模型文件夹的路径
original_image_path = 'dataset_baisha/input.jpg'  # 原始大图路径
output_folder = 'Baojidian_stitched'  # 输出文件夹路径

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 调用函数，拼接图块并保存结果
models_results = stitch_tiles(tiles_folder, original_image_path, 512, 256, 'jpg', output_folder)
