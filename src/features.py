# -*- coding: utf-8 -*-
"""手工纹理特征：LBP、HOG、Gabor（支持特征集合选择，用于消融实验）。

性能关键设计：LBP 与 Gabor 都在**整张图**上预计算一次，再按 patch 切片统计，
避免对每个 patch 重复调用滤波器（否则每张图会跑 5000+ 次 Gabor，极慢）。

特征集合（features 参数）：
  ("lbp",)                   -> 仅 LBP（方法 A1 基础基线）
  ("lbp", "hog", "gabor")    -> 三类特征融合（方法 A2 完整基线）
"""
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern, hog
from skimage.filters import gabor

import config


def to_gray(img):
    """转灰度图，返回 float32，值域 [0,1]。"""
    if img.ndim == 3:
        return rgb2gray(img).astype(np.float32)
    return img.astype(np.float32)


def lbp_map(gray, P=None, R=None):
    """整图 LBP 编码图（返回 uint8，均匀模式）。"""
    P = P or config.LBP_P
    R = R or config.LBP_R
    img_u8 = (np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8)
    return local_binary_pattern(img_u8, P=P, R=R, method="uniform")


def lbp_histogram_from_codes(codes_patch, P=None):
    """从 LBP 编码图的一个 patch 计算直方图。"""
    P = P or config.LBP_P
    n_bins = int(P + 2)  # 均匀模式共有 P+2 种
    hist, _ = np.histogram(codes_patch.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def hog_descriptor(gray_patch, pixels_per_cell=8, orientations=9):
    kwargs = dict(orientations=orientations,
                  pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                  cells_per_block=(2, 2),
                  feature_vector=True)
    try:
        fd = hog(gray_patch, channel_axis=None, **kwargs)
    except TypeError:
        fd = hog(gray_patch, **kwargs)
    return fd.astype(np.float32)


def gabor_magnitude_maps(gray, thetas=None, freqs=None):
    """整图多方向多频率 Gabor 幅度响应图，返回 list。"""
    thetas = thetas or config.GABOR_THETAS
    freqs = freqs or config.GABOR_FREQS
    maps = []
    for freq in freqs:
        for theta in thetas:
            real, imag = gabor(gray, frequency=freq, theta=np.deg2rad(theta))
            maps.append(np.sqrt(real ** 2 + imag ** 2).astype(np.float32))
    return maps


def extract_patch_features(gray_patch, lbp_codes_patch, mag_maps, r, c, ps,
                           features=("lbp", "hog", "gabor")):
    """从预计算好的整图特征里，按指定特征集合提取单个 patch 的拼接特征。

    参数：
      gray_patch      当前 patch 的灰度图
      lbp_codes_patch 当前 patch 的 LBP 编码（未用 LBP 时可为 None）
      mag_maps        整图 Gabor 幅度响应图列表（未用 Gabor 时可为 None）
      r, c, ps        patch 左上角坐标与边长
      features        特征集合元组
    """
    parts = []
    if "lbp" in features:
        parts.append(lbp_histogram_from_codes(lbp_codes_patch))
    if "hog" in features:
        parts.append(hog_descriptor(gray_patch))
    if "gabor" in features:
        g = []
        for m in mag_maps:
            s = m[r:r + ps, c:c + ps]
            g.append(s.mean())
            g.append(s.std())
        parts.append(np.array(g, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


# 兼容旧调用名
def extract_patch_fast(gray_patch, lbp_codes_patch, mag_maps, r, c, ps):
    return extract_patch_features(gray_patch, lbp_codes_patch, mag_maps, r, c, ps)


def feature_dim(features=("lbp", "hog", "gabor")):
    """返回给定特征集合的拼接维度（供打印/诊断使用）。"""
    dim = 0
    if "lbp" in features:
        dim += config.LBP_P + 2
    if "hog" in features:
        dim += hog_descriptor(np.zeros((config.PATCH_SIZE, config.PATCH_SIZE), np.float32)).shape[0]
    if "gabor" in features:
        dim += 2 * len(config.GABOR_THETAS) * len(config.GABOR_FREQS)
    return dim
