'结构'
import matplotlib.pyplot as plt
import numpy as np

# 方法与指标
methods = ["MLineNet", "CLAHE+BF+MLineNet", "CycleGAN+MLineNet", "Ours"]
metrics = ["LDP", "TC", "LS", "Q"]

# 原始数据（不归一化）
data = [
    [86.58, 92.84, 85.37, 88.26],  # MLineNet
    [88.19, 92.42, 86.97, 89.19],  # CLAHE+BF+MLineNet
    [86.89, 94.21, 85.39, 88.83],  # CycleGAN+MLineNet
    [89.54, 93.77, 88.14, 90.48]   # Ours
]

# 雷达图角度设置
num_vars = len(metrics)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# 数据闭合处理
data_closed = [row + [row[0]] for row in data]

# 使用淡雅色
colors = ['#8ecae6', '#a1c298', '#fcbf49', '#f4978e']

# 创建雷达图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.spines['polar'].set_visible(False)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 设置坐标标签
plt.xticks(angles[:-1], metrics, fontsize=15, fontweight='bold')

# 设置 y 轴刻度（真实数值）
ax.set_ylim(70,95)
ax.set_yticks([70, 75, 80, 85, 90, 95])
ax.set_yticklabels(["70", "75", "80", "85", "90", "95"], fontsize=12)

# 绘制曲线与填充
for i, method in enumerate(methods):
    ax.plot(angles, data_closed[i], color=colors[i], linewidth=2, label=method)
    ax.fill(angles, data_closed[i], color=colors[i], alpha=0.15)

# 添加图例和标题
plt.legend(loc='upper right', bbox_to_anchor=(1, 1.5), fontsize=15, frameon=False)
# plt.title("Radar Chart of Different Methods", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# '敦煌'
# import matplotlib.pyplot as plt
# import numpy as np

# # 方法与指标
# methods = ["Canny", "Sobel", "DiffusionEdge", "Teed", "Ours"]
# metrics = ["SSIM", "TC", "LCI", "Q"]

# # 新数据（未归一化）
# data = [
#     [81.14, 94.56, 82.19, 85.96],  # Canny
#     [84.31, 89.43, 84.49, 86.08],  # Sobel
#     [82.79, 96.55, 83.25, 87.53],  # DiffusionEdge
#     [86.13, 93.37, 87.14, 88.88],  # Teed
#     [89.54, 93.77, 88.14, 90.48]   # Ours
# ]

# # 雷达图角度设置
# num_vars = len(metrics)
# angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
# angles += angles[:1]

# # 闭合数据
# data_closed = [row + [row[0]] for row in data]

# # 淡雅颜色
# colors = ['#cab2d6', '#a6cee3', '#b2df8a', '#fdbf6f', '#fbb4ae']

# # 创建图
# fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
# ax.spines['polar'].set_visible(False)
# ax.set_theta_offset(np.pi / 2)
# ax.set_theta_direction(-1)

# # 设置坐标轴
# plt.xticks(angles[:-1], metrics, fontsize=15, fontweight='bold')
# ax.set_ylim(70, 100)
# ax.set_yticks([70, 75, 80, 85, 90, 95, 100])
# ax.set_yticklabels(["70", "75", "80", "85", "90", "95", "100"], fontsize=12)

# # 绘制曲线与填充
# for i, method in enumerate(methods):
#     ax.plot(angles, data_closed[i], color=colors[i], linewidth=2, label=method)
#     ax.fill(angles, data_closed[i], color=colors[i], alpha=0.15)

# # 图例设置
# plt.legend(loc='upper right', bbox_to_anchor=(1, 1.5), fontsize=13, frameon=False)

# plt.tight_layout()
# plt.show()
# '白沙'

# import matplotlib.pyplot as plt
# import numpy as np

# # 方法与指标
# methods = ["Canny", "Sobel", "DiffusionEdge", "Teed", "Ours"]
# metrics = ["SSIM", "TC", "LCI", "Q"]

# # 新数据（未归一化）
# data = [
#     [80.66, 88.81, 84.77, 84.75],  # Sobel
#     [81.78, 89.01, 85.72, 85.50],  # Canny
    
#     [78.92, 95.58, 82.10, 85.53],  # DiffusionEdge
#     [82.62, 93.52, 87.99, 88.04],  # Teed
#     [85.77, 93.93, 88.18, 89.29]   # Ours
# ]

# # 雷达图角度设置
# num_vars = len(metrics)
# angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
# angles += angles[:1]

# # 闭合数据
# data_closed = [row + [row[0]] for row in data]

# # 淡雅颜色
# colors = ['#cab2d6', '#a6cee3', '#b2df8a', '#fdbf6f', '#fbb4ae']

# # 创建图
# fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
# ax.spines['polar'].set_visible(False)
# ax.set_theta_offset(np.pi / 2)
# ax.set_theta_direction(-1)

# # 设置坐标轴
# plt.xticks(angles[:-1], metrics, fontsize=15, fontweight='bold')
# ax.set_ylim(70, 100)
# ax.set_yticks([70, 75, 80, 85, 90, 95, 100])
# ax.set_yticklabels(["70", "75", "80", "85", "90", "95", "100"], fontsize=12)

# # 绘制曲线与填充
# for i, method in enumerate(methods):
#     ax.plot(angles, data_closed[i], color=colors[i], linewidth=2, label=method)
#     ax.fill(angles, data_closed[i], color=colors[i], alpha=0.15)

# # 图例设置
# plt.legend(loc='upper right', bbox_to_anchor=(1, 1.5), fontsize=13, frameon=False)

# plt.tight_layout()
# plt.show()