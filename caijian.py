import os
import cv2
import tkinter as tk
from tkinter import filedialog

# 选择ROI的回调
def select_roi(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图像 {image_path}")
        return None

    roi = cv2.selectROI("Select ROI", image, False, False)
    cv2.destroyAllWindows()
    
    if roi == (0, 0, 0, 0):
        print("用户未选择ROI，跳过该图片")
        return None
    
    return roi  # (x, y, w, h)

# 主函数
def main():
    # 选择主文件夹
    root = tk.Tk()
    root.withdraw()
    main_folder = filedialog.askdirectory(title="选择包含模型结果的主文件夹")
    if not main_folder:
        print("未选择文件夹，程序退出。")
        return
    
    # 获取所有模型文件夹
    model_folders = sorted([os.path.join(main_folder, d) for d in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, d))])
    if not model_folders:
        print("未找到模型文件夹，程序退出。")
        return

    # 选择保存路径
    save_folder = filedialog.askdirectory(title="选择保存裁剪结果的文件夹")
    if not save_folder:
        print("未选择保存文件夹，程序退出。")
        return

    # 读取第一个模型文件夹
    first_model_folder = model_folders[0]
    
    # 获取基准文件夹中所有图像的"文件名（不含后缀）"
    image_names_without_ext = {os.path.splitext(f)[0]: f for f in os.listdir(first_model_folder)}

    if not image_names_without_ext:
        print("未找到图片文件，程序退出。")
        return

    crop_regions = {}

    # 依次处理每张图
    for image_base_name, image_full_name in image_names_without_ext.items():
        first_image_path = os.path.join(first_model_folder, image_full_name)

        # 让用户选择裁剪区域
        print(f"请在窗口中选择 {image_full_name} 的裁剪区域...")
        roi = select_roi(first_image_path)
        if roi is None:
            continue
        
        crop_regions[image_base_name] = roi

        # 遍历所有模型文件夹，裁剪匹配的图片（不考虑后缀）
        for model_folder in model_folders:
            matched_files = [f for f in os.listdir(model_folder) if os.path.splitext(f)[0] == image_base_name]
            
            for matched_file in matched_files:
                image_path = os.path.join(model_folder, matched_file)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"警告：无法读取 {image_path}，跳过。")
                    continue

                x, y, w, h = roi
                cropped = image[y:y+h, x:x+w]

                # 生成保存路径
                model_name = os.path.basename(model_folder)
                save_path = os.path.join(save_folder, model_name)
                os.makedirs(save_path, exist_ok=True)
                save_image_path = os.path.join(save_path, matched_file)

                # 保存裁剪后的图像
                cv2.imwrite(save_image_path, cropped)
                print(f"已保存：{save_image_path}")

    print("所有图片裁剪完成！")

if __name__ == "__main__":
    main()
