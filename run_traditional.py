# -*- coding: utf-8 -*-
"""运行方法 A：LBP + HOG + Gabor 纹理特征 + One-Class SVM。"""
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

import config
from src import data, visualize
from src.traditional import TraditionalDetector
from src.evaluate import evaluate


def run(category=None, max_test=None, verbose=True):
    config.ensure_dirs()
    category = category or config.CATEGORY
    train_paths = data.list_train_images(category)
    test_items = data.list_test_images(category)
    print(f"[方法A·传统] 类别={category}  训练(合格)={len(train_paths)}  测试={len(test_items)}")

    detector = TraditionalDetector()
    t0 = time.time()
    detector.fit(train_paths)
    train_time = time.time() - t0

    metrics = evaluate(detector, test_items, max_images=max_test, verbose=verbose)
    metrics.update({"method": "traditional", "category": category,
                    "train_time_s": train_time})

    print("\n===== 方法 A（LBP + HOG + Gabor + One-Class SVM）=====")
    print(f"  图像级 AUROC : {metrics['image_auroc']:.4f}")
    print(f"  像素级 AUROC : {metrics['pixel_auroc']:.4f}")
    print(f"  平均推理耗时 : {metrics['avg_inference_ms']:.2f} ms/张")
    print(f"  训练耗时     : {train_time:.1f} s")

    out_dir = os.path.join(config.RESULTS_DIR, f"traditional_{category}")
    visualize.save_samples(detector, test_items, out_dir, n=4, method="traditional")
    return metrics


def main():
    p = argparse.ArgumentParser(description="运行方法 A：传统纹理特征 + One-Class SVM")
    p.add_argument("--category", default=config.CATEGORY)
    p.add_argument("--max-test", type=int, default=None, help="限制测试图像数量（调试用）")
    args = p.parse_args()

    metrics = run(category=args.category, max_test=args.max_test)
    out = os.path.join(config.RESULTS_DIR, f"traditional_{args.category}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 -> {out}")


if __name__ == "__main__":
    main()
