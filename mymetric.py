import os
import numpy as np
import cv2
from sklearn.metrics import precision_recall_curve, auc
from concurrent.futures import ThreadPoolExecutor

# 计算 F-measure
def calculate_f_measure(precision, recall):
    return 2 * (precision * recall) / (precision + recall + 1e-8)

# 计算 Precision 和 Recall
def compute_precision_recall(pred_edges, gt_edges):
    true_positives = np.sum((pred_edges == 1) & (gt_edges == 1))
    false_positives = np.sum((pred_edges == 1) & (gt_edges == 0))
    false_negatives = np.sum((pred_edges == 0) & (gt_edges == 1))

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)

    return precision, recall

# 并行计算 ODS
def compute_ods(pred_edges, gt_edges, thresholds):
    best_threshold = 0
    best_f_measure = 0

    def process_threshold(threshold):
        binary_pred_edges = (pred_edges >= threshold).astype(int)
        precision, recall = compute_precision_recall(binary_pred_edges, gt_edges)
        return calculate_f_measure(precision, recall), threshold

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_threshold, thresholds)
        for f_measure, threshold in results:
            if f_measure > best_f_measure:
                best_f_measure = f_measure
                best_threshold = threshold

    print('best_threshold:', best_threshold)
    return best_f_measure

# 并行计算 OIS
def compute_ois(pred_edges, gt_edges, thresholds):
    f_measures = np.zeros(len(pred_edges))

    def process_image(i):
        best_f_measure = 0
        for threshold in thresholds:
            binary_pred_edges = (pred_edges[i] >= threshold).astype(int)
            precision, recall = compute_precision_recall(binary_pred_edges, gt_edges[i])
            best_f_measure = max(best_f_measure, calculate_f_measure(precision, recall))
        return best_f_measure

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_image, range(len(pred_edges)))
        f_measures[:] = list(results)

    return np.mean(f_measures)

# 计算 AP
def compute_ap(pred_edges, gt_edges):
    all_preds = pred_edges.flatten()
    all_gts = gt_edges.flatten()

    precision, recall, _ = precision_recall_curve(all_gts, all_preds)
    return auc(recall, precision)

# 计算所有的边缘检测指标
def compute_edge_detection_metrics(pred_edges, gt_edges):
    thresholds = np.linspace(0, 1, num=10)
    ods = compute_ods(pred_edges, gt_edges, thresholds)
    ois = compute_ois(pred_edges, gt_edges, thresholds)
    ap = compute_ap(pred_edges, gt_edges)
    return ods, ois, ap

# 并行加载数据
def load_image(file_paths):
    real_file_path, pred_file_path = file_paths
    real_img = cv2.imread(real_file_path, cv2.IMREAD_GRAYSCALE)
    pred_img = cv2.imread(pred_file_path, cv2.IMREAD_GRAYSCALE)

    if real_img is None or pred_img is None:
        print(f"Error: Could not read {real_file_path} or {pred_file_path}")
        return None, None

    real_img =  np.where(real_img < 128, 1, 0).astype(np.uint8)
    pred_img = 1 - (pred_img.astype(np.float32) / 255.0)
    return real_img, pred_img

# 并行加载数据集
def load_data(real_labels_path, pred_labels_path):
    real_labels, pred_labels = [], []
    file_pairs = [
        (os.path.join(real_labels_path, filename[:-4] + '.png'), os.path.join(pred_labels_path, filename))
        for filename in os.listdir(pred_labels_path)
    ]

    with ThreadPoolExecutor() as executor:
        results = executor.map(load_image, file_pairs)

        for real_img, pred_img in results:
            if real_img is not None and pred_img is not None:
                real_labels.append(real_img)
                pred_labels.append(pred_img)

    return np.array(real_labels), np.array(pred_labels)

# 计算模型指标的主函数
def evaluate_edge_detection_metrics(real_labels_folder, pred_labels_folder):
    real_labels, pred_labels = load_data(real_labels_folder, pred_labels_folder)
    ods, ois, ap = compute_edge_detection_metrics(pred_labels, real_labels)
    print(f"ODS: {ods:.4f}, OIS: {ois:.4f}, AP: {ap:.4f}")

# 示例调用
real_labels_folder = 'opt/dataset/BIPED/edgev1_01'
pred_labels_folder = 'result/BIPED2BIPED/fused'
evaluate_edge_detection_metrics(real_labels_folder, pred_labels_folder)
