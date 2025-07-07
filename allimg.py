import os
from PIL import Image

# 输入文件夹路径（包含多个模型的结果）
input_folder = "Baojidian_stitched"
# 输出文件夹路径
output_folder = "combined_Baojidian_stitched"
os.makedirs(output_folder, exist_ok=True)

# 获取所有模型的文件夹名称
model_folders = [os.path.join(input_folder, folder) for folder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, folder))]

# 确保文件夹非空
if not model_folders:
    print("输入文件夹中没有模型结果文件夹！")
    exit()

# 获取所有模型文件夹中图像的交集（确保图像名称一致）
common_images = set(os.listdir(model_folders[0]))
for model_folder in model_folders[1:]:
    common_images.intersection_update(os.listdir(model_folder))

if not common_images:
    print("模型文件夹之间没有共同的图像！")
    exit()

# 定义每行最多放5个图像
max_images_per_row = 5

# 遍历共同的图像名称
for image_name in common_images:
    # 读取每个模型对应的图像
    images = []
    for model_folder in model_folders:
        image_path = os.path.join(model_folder, image_name)
        if os.path.exists(image_path):
            images.append(Image.open(image_path))
        else:
            print(f"图像 {image_name} 在 {model_folder} 中不存在，跳过！")
    
    if images:
        # 获取每个图像的宽度和高度
        widths, heights = zip(*(img.size for img in images))
        
        # 计算每行的宽度（最多5个图像）
        row_width = sum(widths[:max_images_per_row])
        row_height = max(heights[:max_images_per_row])

        # 计算总的宽度和高度
        total_width = 0
        total_height = 0
        for i in range(0, len(images), max_images_per_row):
            row_images = images[i:i + max_images_per_row]
            row_width = sum(img.size[0] for img in row_images)
            row_height = max(img.size[1] for img in row_images)
            total_width = max(total_width, row_width)
            total_height += row_height
        
        # 创建拼接后的新图像
        combined_image = Image.new("RGB", (total_width, total_height))

        # 拼接图像
        y_offset = 0
        for i in range(0, len(images), max_images_per_row):
            row_images = images[i:i + max_images_per_row]
            x_offset = 0
            for img in row_images:
                combined_image.paste(img, (x_offset, y_offset))
                x_offset += img.size[0]
            y_offset += row_height

        # 保存拼接图像
        combined_image.save(os.path.join(output_folder, image_name))

print("所有图像拼接完成！拼接结果保存在:", output_folder)
