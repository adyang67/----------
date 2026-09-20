# -*- coding: utf-8 -*-
"""MVTec AD 数据集加载工具。"""
import os
import numpy as np
from PIL import Image

import config


def category_root(category=None):
    category = category or config.CATEGORY
    return os.path.join(config.MVTEC_DIR, category)


def list_train_images(category=None):
    """返回训练集（全部为合格品）图像路径列表。"""
    train_dir = os.path.join(category_root(category), "train", "good")
    files = [os.path.join(train_dir, f) for f in sorted(os.listdir(train_dir))
             if f.lower().endswith((".png", ".jpg"))]
    return files


def list_test_images(category=None):
    """返回测试集条目：每个为 dict(path, label, type, mask)。

    label: 0=合格, 1=缺陷。mask: 缺陷掩码路径（合格品为 None）。
    """
    root = category_root(category)
    test_root = os.path.join(root, "test")
    gt_root = os.path.join(root, "ground_truth")
    items = []
    for sub in sorted(os.listdir(test_root)):
        subdir = os.path.join(test_root, sub)
        if not os.path.isdir(subdir):
            continue
        label = 0 if sub == "good" else 1
        mask_dir = os.path.join(gt_root, sub) if label else None
        for f in sorted(os.listdir(subdir)):
            if not f.lower().endswith((".png", ".jpg")):
                continue
            item = {"path": os.path.join(subdir, f), "label": label, "type": sub,
                    "mask": _find_mask(mask_dir, f) if mask_dir else None}
            items.append(item)
    return items


def _find_mask(mask_dir, filename):
    base = os.path.splitext(filename)[0]
    for cand in (filename, base + "_mask.png", base + ".png"):
        p = os.path.join(mask_dir, cand)
        if os.path.exists(p):
            return p
    return None


def load_image(path, size=None, mode="RGB"):
    """加载并 resize 图像，返回 uint8 numpy 数组。"""
    size = size or config.IMAGE_SIZE
    img = Image.open(path).convert(mode)
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img)


def load_mask(path, size=None):
    """加载并 resize 二值掩码，返回 0/1 float 数组。"""
    size = size or config.IMAGE_SIZE
    mask = Image.open(path).convert("L")
    mask = mask.resize((size, size), Image.NEAREST)
    return (np.array(mask) > 127).astype(np.float32)
