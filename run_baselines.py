# -*- coding: utf-8 -*-
"""一键复现四条基线：两种方法（传统 / 深度）各两个变体。

  A1  传统  LBP + One-Class SVM（单一特征）
  A2  传统  LBP + HOG + Gabor + One-Class SVM（特征融合）
  B1  深度  PaDiM + ResNet18
  B2  深度  PaDiM + WideResNet50-2（backbone 消融）

用法：
  python run_baselines.py                          # 跑全量测试集
  python run_baselines.py --category screw         # 指定类别
  python run_baselines.py --max-test 20            # 每基线只用 20 张测试图快速验证
  python run_baselines.py --skip b2                # 跳过较重的 wide_resnet50_2

结果：每个基线训练+评估一次，缓存到 results/<key>_<category>.json，
最后输出对比表格、CSV 与柱状图。
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src import data, visualize
from src.traditional import TraditionalDetector
from src.padim import PaDiM
from src.evaluate import evaluate

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------- 四条基线定义 ----------
BASELINES = [
    {
        "key": "A1_lbp_svm",
        "method": "传统",
        "name": "A1: LBP + One-Class SVM",
        "build": lambda: TraditionalDetector(features=("lbp",)),
    },
    {
        "key": "A2_fusion_svm",
        "method": "传统",
        "name": "A2: LBP+HOG+Gabor + One-Class SVM",
        "build": lambda: TraditionalDetector(features=("lbp", "hog", "gabor")),
    },
    {
        "key": "B1_padim_resnet18",
        "method": "深度",
        "name": "B1: PaDiM + ResNet18",
        "build": lambda: PaDiM(backbone="resnet18"),
    },
    {
        "key": "B2_padim_wrn50",
        "method": "深度",
        "name": "B2: PaDiM + WideResNet50-2",
        "build": lambda: PaDiM(backbone="wide_resnet50_2"),
    },
]


def run_baseline(spec, category, max_test):
    """训练+评估单条基线，返回 metrics dict。"""
    key = spec["key"]
    cache_path = os.path.join(config.RESULTS_DIR, f"{key}_{category}.json")
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            metrics = json.load(f)
        print(f"[缓存命中] {spec['name']} -> {cache_path}")
        return metrics

    train_paths = data.list_train_images(category)
    test_items = data.list_test_images(category)
    print(f"\n{'=' * 60}\n{spec['name']}  类别={category} "
          f"训练(合格)={len(train_paths)} 测试={len(test_items)}")

    detector = spec["build"]()
    t0 = time.time()
    detector.fit(train_paths)
    train_time = time.time() - t0

    metrics = evaluate(detector, test_items, max_images=max_test)
    metrics.update({"key": key, "name": spec["name"], "method": spec["method"],
                    "category": category, "train_time_s": train_time})

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 可视化前 4 张缺陷样本
    out_dir = os.path.join(config.RESULTS_DIR, f"{key}_{category}")
    visualize.save_samples(detector, test_items, out_dir, n=4, method=key)

    print(f"  图像级 AUROC : {metrics['image_auroc']:.4f}")
    print(f"  像素级 AUROC : {metrics['pixel_auroc']:.4f}")
    print(f"  平均推理耗时 : {metrics['avg_inference_ms']:.2f} ms/张")
    print(f"  训练耗时     : {train_time:.1f} s")
    return metrics


def main():
    p = argparse.ArgumentParser(description="一键复现四条基线")
    p.add_argument("--category", default=config.CATEGORY)
    p.add_argument("--max-test", type=int, default=None, help="限制测试图像数量（调试用）")
    p.add_argument("--skip", default="", help="逗号分隔要跳过的基线 key，如 b2 或 a1,b2")
    args = p.parse_args()

    config.ensure_dirs()
    cat = args.category
    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}

    # 用 key 里的短名匹配 --skip（a1/a2/b1/b2 或完整 key 均可）
    def should_skip(spec):
        short = spec["key"].split("_")[0].lower()
        return short in skip or spec["key"].lower() in skip

    results = []
    for spec in BASELINES:
        if should_skip(spec):
            print(f"[跳过] {spec['name']}")
            continue
        results.append(run_baseline(spec, cat, args.max_test))

    if not results:
        print("没有可运行的基线。")
        return

    # ---------- 汇总对比 ----------
    print("\n" + "=" * 72)
    print(f"{'基线':<34}{'图像级AUROC':<14}{'像素级AUROC':<14}{'推理ms':<12}{'训练s':<10}")
    print("-" * 72)
    for m in results:
        print(f"{m['name']:<34}{m['image_auroc']:<14.4f}{m['pixel_auroc']:<14.4f}"
              f"{m['avg_inference_ms']:<12.2f}{m['train_time_s']:<10.1f}")

    # CSV
    csv_path = os.path.join(config.RESULTS_DIR, f"baselines_{cat}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("baseline,method,image_auroc,pixel_auroc,avg_inference_ms,train_time_s\n")
        for m in results:
            f.write(f"{m['key']},{m['method']},{m['image_auroc']:.4f},"
                    f"{m['pixel_auroc']:.4f},{m['avg_inference_ms']:.2f},{m['train_time_s']:.1f}\n")
    print(f"\nCSV 已保存 -> {csv_path}")

    # 柱状图
    names = [m["name"].replace(": ", "\n") for m in results]
    img_auc = [m["image_auroc"] for m in results]
    px_auc = [m["pixel_auroc"] for m in results]
    infer = [m["avg_inference_ms"] for m in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(results))
    w = 0.36
    axes[0].bar(x - w / 2, img_auc, w, label="图像级 AUROC", color="tab:blue")
    axes[0].bar(x + w / 2, px_auc, w, label="像素级 AUROC", color="tab:orange")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=9)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("AUROC 对比（越高越好）")
    axes[0].legend()

    axes[1].bar(x, infer, 0.5, color="tab:green")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=9)
    axes[1].set_title("平均推理耗时 ms/张（越低越好）")

    plt.tight_layout()
    fig_path = os.path.join(config.RESULTS_DIR, f"baselines_{cat}.png")
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"对比图已保存 -> {fig_path}")


if __name__ == "__main__":
    main()
