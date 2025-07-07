import matplotlib.pyplot as plt
import numpy as np

# 方法与指标
methods = ["MLineNet", "CLAHE+BF+MLineNet", "CycleGAN+MLineNet", "Ours"]
metrics = ["SSIM", "TC", "LCI", "Q"]

# 数据
data = [
    [86.58, 94.00, 88.37, 89.65],  # MLineNet
    [88.19, 92.42, 86.97, 89.19],  # CLAHE+BF+MLineNet
    [86.89, 94.21, 85.39, 88.83],  # CycleGAN+MLineNet
    [89.54, 93.77, 88.14, 90.48]   # Ours
]

# 转置数据：每行是一个指标，每列是对应的四个方法
data = np.array(data).T  # shape: (4 metrics, 4 methods)

# 横轴位置和柱宽
x = np.arange(len(metrics))
width = 0.2

# 颜色
colors = ['#8ecae6', '#a1c298', '#fcbf49', '#f4978e']

# 画图
fig, ax = plt.subplots(figsize=(10, 6))

# 存储每个指标下所有模型柱形中心点及高度
center_xs_by_metric = [[] for _ in metrics]
heights_by_metric = [[] for _ in metrics]

for i in range(len(methods)):
    offset = x + (i - 1.5) * width
    bars = ax.bar(offset, data[:, i], width=width, label=f"{i+1} ({methods[i]})", color=colors[i])

    for j, bar in enumerate(bars):
        height = bar.get_height()
        center = bar.get_x() + bar.get_width() / 2
        center_xs_by_metric[j].append(center)
        heights_by_metric[j].append(height)

        # 添加顶部标注
        ax.text(center, height + 0.3, f"{height:.1f}", ha='center', va='bottom', fontsize=14)

# 添加每个指标的连线
for i in range(len(metrics)):
    ax.plot(center_xs_by_metric[i], heights_by_metric[i], color='gray', linestyle='--', linewidth=1)

# 坐标设置
ax.set_ylabel("Score", fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_ylim(80, 100)
ax.legend(title="Structure", fontsize=12)

plt.tight_layout()
plt.show()
