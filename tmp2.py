import cv2
import numpy as np

# 输入 & 输出文件路径
input_image_path = "H:/Feng/duibi/all/xihua/9-0-0.jpg"  # 你的轮廓图
output_image_path = "H:/Feng/duibi/all/xihua2/straightened_contour.jpg"  # 处理后的图像

# 读取灰度图
image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# 确保图像正确加载
if image is None:
    raise FileNotFoundError(f"无法加载图像: {input_image_path}")

# **二值化，确保只有黑白**
_, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# **提取轮廓**
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# **创建白色背景**
straightened_image = np.ones_like(image) * 255

# **让轮廓变直**
for contour in contours:
    if len(contour) > 2:  # 过滤掉小的噪点
        straight_contour = cv2.approxPolyDP(contour, epsilon=0.5, closed=False)  # 让线条更直
        cv2.drawContours(straightened_image, [straight_contour], -1, (0, 0, 0), 1)  # 重新绘制直线

# **保存处理后的图像**
cv2.imwrite(output_image_path, straightened_image)
print(f"✅ 线条已平滑（锯齿已拉直），结果已保存至: {output_image_path}")
