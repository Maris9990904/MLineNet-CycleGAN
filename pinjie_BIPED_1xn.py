import os
from PIL import Image

def add_border(img, border_size=2):
    """给图像添加黑色边框，默认边框为 2 像素"""
    width, height = img.size
    # 创建一个比原图大，且背景为黑色的图像
    bordered_img = Image.new("RGB", (width + 2 * border_size, height + 2 * border_size), (0, 0, 0))
    # 将原图粘贴到黑色背景上，位置偏移边框的大小
    bordered_img.paste(img, (border_size, border_size))
    return bordered_img

def concatenate_images(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有子文件夹
    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]

    # 用于存储同名但不同后缀的图片
    image_dict = {}

    # 遍历所有子文件夹
    for subfolder in subfolders:
        for filename in os.listdir(subfolder):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_path = os.path.join(subfolder, filename)

                # 提取主文件名（去掉后缀）
                base_name = os.path.splitext(filename)[0]

                # 读取图像
                img = Image.open(file_path)

                # 给图像添加黑色边框
                img_with_border = add_border(img)

                # 归类到字典中
                if base_name not in image_dict:
                    image_dict[base_name] = []
                image_dict[base_name].append(img_with_border)

    # 遍历字典，拼接相同主文件名的图片
    for base_name, images in image_dict.items():
        # 获取每个图像的尺寸（现在有边框）
        width, height = images[0].size
        total_width = width * len(images) + 10 * (len(images) - 1)  # 10 像素间隔
        max_height = height

        # 创建白色背景的空图像
        concatenated_image = Image.new("RGB", (total_width, max_height), (255, 255, 255))  # 设为白色背景

        # 逐个拼接图像
        x_offset = 0
        for img in images:
            concatenated_image.paste(img, (x_offset, 0))
            x_offset += width + 5  # 添加10像素间隔

        # 统一保存格式（例如 PNG）
        output_path = os.path.join(output_folder, f"{base_name}.png")
        concatenated_image.save(output_path)
        print(f"Processed and saved: {output_path}")

# 设置输入和输出文件夹
input_folder = "Dunhuang/q/all_jubu"  # 替换为你的输入文件夹
output_folder = "Dunhuang/q/combined_all_jubu"  # 输出文件夹
concatenate_images(input_folder, output_folder)
