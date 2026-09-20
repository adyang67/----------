# -*- coding: utf-8 -*-
"""可视化：输入图像、真实掩码、预测热力图。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文标题需要中文字体（Windows 自带）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from src import data


def normalize(amap):
    mn, mx = amap.min(), amap.max()
    if mx - mn < 1e-8:
        return np.zeros_like(amap)
    return (amap - mn) / (mx - mn)


def plot_sample(img, mask, amap, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img); axes[0].set_title("输入图像"); axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("真实缺陷"); axes[1].axis("off")
    axes[2].imshow(normalize(amap), cmap="jet"); axes[2].set_title("预测热力图"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_samples(detector, test_items, out_dir, n=4, method="method"):
    """对前 n 张缺陷样本生成可视化。"""
    os.makedirs(out_dir, exist_ok=True)
    anomalies = [it for it in test_items if it["label"] == 1]
    count = 0
    for it in anomalies:
        if count >= n:
            break
        img = data.load_image(it["path"])
        amap, score = detector.predict(img)
        mask = data.load_mask(it["mask"]) if it["mask"] else np.zeros_like(amap)
        name = f"{method}_{it['type']}_{count}.png"
        plot_sample(img, mask, amap, os.path.join(out_dir, name))
        count += 1
    print(f"  已保存 {count} 张可视化样本 -> {out_dir}")
