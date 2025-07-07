import os
import shutil
import tkinter as tk
from tkinter import filedialog

def copy_matching_images():
    # 选择基准图像文件夹
    root = tk.Tk()
    root.withdraw()
    
    base_folder = filedialog.askdirectory(title="选择基准图像文件夹")
    if not base_folder:
        print("未选择基准图像文件夹，程序退出。")
        return

    # 选择待搜索的文件夹（包含多个子文件夹）
    search_folder = filedialog.askdirectory(title="选择待搜索的文件夹")
    if not search_folder:
        print("未选择待搜索的文件夹，程序退出。")
        return

    # 选择目标文件夹（存放匹配的图像）
    save_folder = filedialog.askdirectory(title="选择保存匹配图像的文件夹")
    if not save_folder:
        print("未选择保存文件夹，程序退出。")
        return

    # 获取基准图像文件名（不含后缀）
    base_image_names = {os.path.splitext(f)[0] for f in os.listdir(base_folder)}

    # 遍历待搜索的文件夹中的所有子文件夹
    for sub_folder in os.listdir(search_folder):
        sub_folder_path = os.path.join(search_folder, sub_folder)
        
        if os.path.isdir(sub_folder_path):  # 确保是子文件夹
            for image_name in os.listdir(sub_folder_path):
                image_base_name, ext = os.path.splitext(image_name)

                # 如果文件名匹配，则复制到目标文件夹
                if image_base_name in base_image_names:
                    src_path = os.path.join(sub_folder_path, image_name)
                    dest_path = os.path.join(save_folder, sub_folder)  # 目标子文件夹
                    os.makedirs(dest_path, exist_ok=True)  # 确保目标文件夹存在

                    shutil.copy2(src_path, os.path.join(dest_path, image_name))
                    print(f"已复制：{src_path} -> {dest_path}")

    print("所有匹配的图片已复制完成！")

if __name__ == "__main__":
    copy_matching_images()
