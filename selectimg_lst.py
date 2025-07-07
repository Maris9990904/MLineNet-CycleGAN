import os

# 读取test.lst文件内容

lst = []
# 从 test.lst 文件读取路径列表
lst_file = "BIPED/test_pair.lst"  # test.lst 文件路径

# 输入和输出文件夹路径
inputs_folder = "BIPED/inputs"  # 输入文件夹路径
edges_folder = "BIPED/edges"    # 边缘文件夹路径

with open(lst_file, 'r') as f:
    lst = eval(f.read())  # 读取文件并转化为Python对象

# 获取lst中的图像文件名
valid_inputs = {os.path.basename(item[0]) for item in lst}  # 从inputs文件夹的路径中提取文件名
valid_edges = {os.path.basename(item[1]) for item in lst}   # 从edges文件夹的路径中提取文件名



# 遍历inputs文件夹，删除不在valid_inputs中的文件
for filename in os.listdir(inputs_folder):
    if filename not in valid_inputs:
        file_path = os.path.join(inputs_folder, filename)
        os.remove(file_path)
        print(f"删除文件: {file_path}")

# 遍历edges文件夹，删除不在valid_edges中的文件
for filename in os.listdir(edges_folder):
    if filename not in valid_edges:
        file_path = os.path.join(edges_folder, filename)
        os.remove(file_path)
        print(f"删除文件: {file_path}")

print("清理完成！")




