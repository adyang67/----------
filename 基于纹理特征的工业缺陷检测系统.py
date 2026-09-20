# ============ 一键：自动下载数据集 → 训练 → 评估 → 输出结果 ============
import os, sys, time
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

ROOT = r"D:\人工智能综合实践课程"          # 项目根目录，按需改
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import config
from src import data
from src.traditional import TraditionalDetector
from src.padim import PaDiM
from src.evaluate import evaluate

MAX_TEST = None      # None=跑完整测试集；想快速验证改成 20


# ---------- 1) 数据集：不存在则自动下载+解压（约 5.26GB，仅首次，需联网） ----------
config.ensure_dirs()
import download_data
if not os.path.isdir(os.path.join(config.MVTEC_DIR, config.CATEGORY)):
    print("未检测到数据集，开始自动下载 + 解压（约 5.26 GB，仅首次需要）...")
    download_data.main()
else:
    print("数据集已存在，跳过下载。")


# ---------- 2) 加载数据 ----------
CATEGORY = config.CATEGORY
train_paths = data.list_train_images(CATEGORY)
test_items  = data.list_test_images(CATEGORY)
n_defect = sum(1 for it in test_items if it["label"] == 1)
print(f"类别 = {CATEGORY} | 训练集(合格品) = {len(train_paths)} 张 | "
    f"测试集 = {len(test_items)} 张（缺陷 {n_defect} 张）")


# ---------- 3) 方法 A：LBP+HOG+Gabor + One-Class SVM ----------
print("\n===== 方法 A：传统纹理特征 + One-Class SVM =====")
a = TraditionalDetector()
t0 = time.time(); a.fit(train_paths)
print(f"方法 A 训练耗时 {time.time()-t0:.1f} s")
ma = evaluate(a, test_items, max_images=MAX_TEST, verbose=True)


# ---------- 4) 方法 B：PaDiM（ResNet18 预训练 + 多元高斯） ----------
print("\n===== 方法 B：PaDiM（预训练特征 + 多元高斯）=====")
b = PaDiM()
t0 = time.time(); b.fit(train_paths)
print(f"方法 B 训练耗时 {time.time()-t0:.1f} s")
mb = evaluate(b, test_items, max_images=MAX_TEST, verbose=True)


# ---------- 5) 结果汇总 ----------
df = pd.DataFrame([
    {"方法": "方法 A（传统）", "图像级 AUROC": round(ma["image_auroc"], 4),
     "像素级 AUROC": round(ma["pixel_auroc"], 4),
     "推理耗时(ms/张)": round(ma["avg_inference_ms"], 2)},
    {"方法": "方法 B（PaDiM）", "图像级 AUROC": round(mb["image_auroc"], 4),
     "像素级 AUROC": round(mb["pixel_auroc"], 4),
     "推理耗时(ms/张)": round(mb["avg_inference_ms"], 2)},
])
print("\n===== 结果汇总 =====")
print(df.to_string(index=False))

