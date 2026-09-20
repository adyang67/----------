# -*- coding: utf-8 -*-
"""评估：图像级 / 像素级 AUROC 与推理速度。"""
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src import data


def evaluate(detector, test_items, max_images=None, verbose=True):
    """对测试集评估，返回 dict(image_auroc, pixel_auroc, avg_inference_ms, n_test)。"""
    test_items = test_items if max_images is None else test_items[:max_images]

    image_scores, image_labels = [], []
    pixel_scores, pixel_labels = [], []
    times = []

    it = tqdm(test_items, desc="评估", disable=not verbose)
    for item in it:
        img = data.load_image(item["path"])
        t0 = time.perf_counter()
        anomaly_map, score = detector.predict(img)
        times.append(time.perf_counter() - t0)

        image_scores.append(score)
        image_labels.append(item["label"])

        if item["mask"] is not None and item["label"] == 1:
            mask = data.load_mask(item["mask"])
        else:
            mask = np.zeros_like(anomaly_map)
        pixel_scores.append(anomaly_map.ravel())
        pixel_labels.append(mask.ravel())

    image_auroc = roc_auc_score(image_labels, image_scores)
    pixel_scores = np.concatenate(pixel_scores)
    pixel_labels = np.concatenate(pixel_labels)
    pixel_auroc = roc_auc_score(pixel_labels, pixel_scores)

    return {
        "image_auroc": float(image_auroc),
        "pixel_auroc": float(pixel_auroc),
        "avg_inference_ms": float(np.mean(times) * 1000),
        "n_test": len(test_items),
    }
