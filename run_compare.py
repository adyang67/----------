# -*- coding: utf-8 -*-
"""对比两种方法的性能与推理速度，并输出表格 / CSV / 对比图。

若已存在某方法的 JSON 结果则直接读取，否则自动运行该方法。
"""
import os
import sys
import json
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
from run_traditional import run as run_trad
from run_padim import run as run_padim


def load_or_run(run_fn, method, category):
    path = os.path.join(config.RESULTS_DIR, f"{method}_{category}.json")
    if os.path.exists(path):
        print(f"读取已缓存结果: {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    metrics = run_fn(category)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def main():
    p = argparse.ArgumentParser(description="对比方法 A 与方法 B")
    p.add_argument("--category", default=config.CATEGORY)
    args = p.parse_args()
    cat = args.category

    config.ensure_dirs()
    trad = load_or_run(run_trad, "traditional", cat)
    padim = load_or_run(run_padim, "padim", cat)

    print("\n========== 两种方法对比 ==========")
    print(f"{'指标':<16}{'方法A 传统':<16}{'方法B PaDiM':<16}")
    rows = [
        ("图像级 AUROC", "image_auroc"),
        ("像素级 AUROC", "pixel_auroc"),
        ("平均推理(ms)", "avg_inference_ms"),
        ("训练耗时(s)", "train_time_s"),
    ]
    for name, key in rows:
        print(f"{name:<16}{trad[key]:<16.4f}{padim[key]:<16.4f}")

    # 保存 CSV
    csv_path = os.path.join(config.RESULTS_DIR, f"compare_{cat}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("metric,traditional,padim\n")
        for name, key in rows:
            f.write(f"{name},{trad[key]:.4f},{padim[key]:.4f}\n")
    print(f"\nCSV 已保存 -> {csv_path}")

    # 画对比图
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(2)
    w = 0.35
    axes[0].bar(x - w / 2, [trad["image_auroc"], trad["pixel_auroc"]], w,
                label="方法A 传统", color="tab:blue")
    axes[0].bar(x + w / 2, [padim["image_auroc"], padim["pixel_auroc"]], w,
                label="方法B PaDiM", color="tab:orange")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["图像级 AUROC", "像素级 AUROC"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("AUROC 对比（越高越好）")
    axes[0].legend()

    axes[1].bar(["方法A 传统", "方法B PaDiM"],
                [trad["avg_inference_ms"], padim["avg_inference_ms"]],
                color=["tab:blue", "tab:orange"])
    axes[1].set_title("平均推理耗时 ms/张（越低越好）")

    plt.tight_layout()
    fig_path = os.path.join(config.RESULTS_DIR, f"compare_{cat}.png")
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"对比图已保存 -> {fig_path}")


if __name__ == "__main__":
    main()
