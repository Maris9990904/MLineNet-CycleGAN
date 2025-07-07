import os
import shutil

def copy_matching_images(image_folder, model_results_folder, output_folder):
    # 获取文件夹1中的影像名称
    image_names = {os.path.splitext(f)[0] for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))}
    
    # 遍历文件夹2（多个模型结果文件夹）
    for model_folder in os.listdir(model_results_folder):
        model_path = os.path.join(model_results_folder, model_folder)
        if not os.path.isdir(model_path):
            continue
        
        # 创建对应的输出子文件夹
        output_model_path = os.path.join(output_folder, model_folder)
        os.makedirs(output_model_path, exist_ok=True)
        
        # 遍历模型文件夹中的文件
        for file in os.listdir(model_path):
            file_name, file_ext = os.path.splitext(file)
            if file_name in image_names:  # 如果文件名在影像名称列表中
                src_file = os.path.join(model_path, file)
                dst_file = os.path.join(output_model_path, file)
                shutil.copy2(src_file, dst_file)  # 复制文件并保留元数据
                print(f"Copied: {src_file} -> {dst_file}")
    
    print("All matching images have been copied.")

# 示例调用
image_folder = "H:/Feng/duibi/bianyuanzhanshi_all/01input"  # 影像所在文件夹
model_results_folder = "Dunhuang/edge detection display/all_edges"  # 存放多个模型结果的文件夹
output_folder = "H:/Feng/duibi/bianyuanzhanshi_all"  # 目标输出文件夹

copy_matching_images(image_folder, model_results_folder, output_folder)