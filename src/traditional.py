# -*- coding: utf-8 -*-
"""方法 A（传统路线）：LBP + HOG + Gabor 纹理特征 + One-Class SVM。

仅使用合格品图像训练 One-Class SVM；推理时对每个滑窗 patch 计算
特征，用 decision_function 的负值作为异常分数，反投影为像素级异常图。

性能：LBP 与 Gabor 在整图上预计算一次（见 features.py），每张图仅需
做一次滤波 + 逐 patch 切片统计，单图推理约零点几秒。
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from skimage.transform import resize
from skimage.filters import gaussian
from tqdm import tqdm

import config
from src import data
from src.features import (to_gray, lbp_map, gabor_magnitude_maps,
                          extract_patch_features, feature_dim)


class TraditionalDetector:
    def __init__(self, patch_size=None, stride=None, max_train_patches=None,
                 nu=None, seed=None, features=("lbp", "hog", "gabor")):
        self.patch_size = patch_size or config.PATCH_SIZE
        self.stride = stride or config.STRIDE
        self.max_train_patches = max_train_patches or config.MAX_TRAIN_PATCHES
        self.nu = nu if nu is not None else config.OCSVM_NU
        self.seed = seed if seed is not None else config.SEED
        self.lbp_p = config.LBP_P
        self.lbp_r = config.LBP_R
        self.thetas = config.GABOR_THETAS
        self.freqs = config.GABOR_FREQS
        self.features = tuple(features)
        self.scaler = StandardScaler()
        self.model = OneClassSVM(kernel="rbf", gamma="scale", nu=self.nu)
        self._trained = False

    # ---------- patch 提取 ----------
    def _positions(self, h, w):
        """返回覆盖整张图的滑窗起点（行、列）。"""
        ps, st = self.patch_size, self.stride
        rows = list(range(0, h - ps + 1, st))
        cols = list(range(0, w - ps + 1, st))
        if (h - ps) not in rows:   # 保证右/下边缘被覆盖
            rows.append(h - ps)
        if (w - ps) not in cols:
            cols.append(w - ps)
        return rows, cols

    def _image_features(self, gray):
        """整图滑窗提取特征，返回 (features (M,d), rows, cols)。"""
        h, w = gray.shape
        # 只在需要时预计算整图特征（Gabor 很贵，A1 基线可跳过）
        lbp = lbp_map(gray, self.lbp_p, self.lbp_r) if "lbp" in self.features else None
        mag_maps = (gabor_magnitude_maps(gray, self.thetas, self.freqs)
                    if "gabor" in self.features else None)
        rows, cols = self._positions(h, w)
        ps = self.patch_size
        feats = []
        for r in rows:
            for c in cols:
                feats.append(extract_patch_features(
                    gray[r:r + ps, c:c + ps],
                    lbp[r:r + ps, c:c + ps] if lbp is not None else None,
                    mag_maps, r, c, ps, features=self.features))
        return np.array(feats, dtype=np.float32), rows, cols

    # ---------- 训练 ----------
    def fit(self, train_paths):
        print(f"[方法A·传统] 特征集合={'+'.join(self.features)} 提取训练集纹理特征 ...")
        all_feats = []
        for p in tqdm(train_paths, desc="训练图像"):
            gray = to_gray(data.load_image(p))
            feats, _, _ = self._image_features(gray)
            all_feats.append(feats)
        all_feats = np.vstack(all_feats)
        print(f"  共 {all_feats.shape[0]} 个 patch，特征维度 {all_feats.shape[1]}")

        rng = np.random.RandomState(self.seed)
        if all_feats.shape[0] > self.max_train_patches:
            idx = rng.choice(all_feats.shape[0], self.max_train_patches, replace=False)
            all_feats = all_feats[idx]
            print(f"  随机下采样到 {self.max_train_patches} 个 patch")

        print("[方法A·传统] 标准化 + 训练 One-Class SVM ...")
        self.scaler.fit(all_feats)
        X = self.scaler.transform(all_feats)
        self.model.fit(X)
        self._trained = True
        print("  训练完成 [OK]")

    # ---------- 推理 ----------
    def predict(self, img):
        """输入 RGB uint8 数组，返回 (anomaly_map (H,W), image_score)。"""
        gray = to_gray(img)  # float [0,1]
        if gray.shape[0] != config.IMAGE_SIZE or gray.shape[1] != config.IMAGE_SIZE:
            gray = resize(gray, (config.IMAGE_SIZE, config.IMAGE_SIZE),
                          mode="reflect", anti_aliasing=True)

        h, w = gray.shape
        feats, rows, cols = self._image_features(gray)
        X = self.scaler.transform(feats)
        # decision_function 越小越异常 -> 取负号作为异常分数
        scores = -self.model.decision_function(X)

        # 反投影为像素级异常图（对重叠区域取平均）
        acc = np.zeros((h, w), dtype=np.float32)
        cnt = np.zeros((h, w), dtype=np.float32)
        k = 0
        for r in rows:
            for c in cols:
                acc[r:r + self.patch_size, c:c + self.patch_size] += scores[k]
                cnt[r:r + self.patch_size, c:c + self.patch_size] += 1.0
                k += 1
        cnt[cnt == 0] = 1.0
        anomaly_map = acc / cnt

        anomaly_map = gaussian(anomaly_map, sigma=max(config.SMOOTH_SIGMA // 2, 1))
        image_score = float(anomaly_map.max())
        return anomaly_map.astype(np.float32), image_score
