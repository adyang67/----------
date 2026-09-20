# -*- coding: utf-8 -*-
"""方法 B（深度路线）：PaDiM —— 预训练 CNN 特征 + 多元高斯拟合。

思路：
  1. 用 ImageNet 预训练的 ResNet 提取多层特征（layer1/2/3）。
  2. 将不同层特征上采样到同一分辨率并逐位置拼接成嵌入向量。
  3. 对每个空间位置用训练集嵌入拟合一个多元高斯分布（均值 + 协方差）。
  4. 推理时计算每个位置的马氏距离作为异常分数，上采样得到像素级异常图。
"""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from skimage.transform import resize
from skimage.filters import gaussian
from tqdm import tqdm

import config
from src import data


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_backbone(name):
    if name == "resnet18":
        try:
            return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            return models.resnet18(pretrained=True)
    elif name == "wide_resnet50_2":
        try:
            return models.wide_resnet50_2(
                weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        except Exception:
            return models.wide_resnet50_2(pretrained=True)
    else:
        raise ValueError(f"未知 backbone: {name}")


class PaDiM:
    def __init__(self, backbone=None, layers=None, dims=None, seed=None):
        self.backbone_name = backbone or config.BACKBONE
        self.layers = tuple(layers or config.PADIM_LAYERS)
        self.dims = dims or config.PADIM_DIMS
        self.seed = seed if seed is not None else config.SEED
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = _build_backbone(self.backbone_name)
        self.net.to(self.device).eval()

        self.layer_names = [f"layer{i}" for i in self.layers]
        self.outputs = {}
        self._register_hooks()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.mean = None
        self.eigvals = None
        self.eigvecs = None
        self.eps = config.PADIM_EPS
        self._trained = False
        self._proj = None      # 随机投影矩阵 (D, d)
        self._spatial = None   # 嵌入特征图空间尺寸 (H1, W1)

    def _register_hooks(self):
        def make_hook(name):
            def hook(module, inp, out):
                self.outputs[name] = out
            return hook
        for name in self.layer_names:
            getattr(self.net, name).register_forward_hook(make_hook(name))

    def _init_proj(self, d_in):
        rng = np.random.RandomState(self.seed)
        proj = rng.randn(d_in, self.dims).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8
        self._proj = proj

    def _embed(self, img_np):
        """img_np: RGB uint8 (H,W,3) -> 返回嵌入 (P, d) numpy float32。"""
        if isinstance(img_np, str):
            img_np = data.load_image(img_np)
        if img_np.shape[0] != config.IMAGE_SIZE or img_np.shape[1] != config.IMAGE_SIZE:
            img_np = (resize(img_np, (config.IMAGE_SIZE, config.IMAGE_SIZE),
                             mode="reflect", anti_aliasing=True) * 255).astype(np.uint8)

        t = self.transform(img_np).unsqueeze(0).to(self.device)
        self.outputs = {}
        with torch.no_grad():
            self.net(t)

        ref = self.outputs[self.layer_names[0]]
        H1, W1 = ref.shape[-2], ref.shape[-1]
        feats = []
        for name in self.layer_names:
            f = self.outputs[name]
            if f.shape[-2:] != (H1, W1):
                f = F.interpolate(f, size=(H1, W1), mode="bilinear", align_corners=False)
            feats.append(f)
        emb = torch.cat(feats, dim=1)                    # (1, D, H1, W1)
        emb = emb[0].permute(1, 2, 0).flatten(0, 1)      # (H1*W1, D)
        emb = emb.cpu().numpy()

        if self._proj is None:
            self._init_proj(emb.shape[1])
        emb = emb @ self._proj                          # (P, d)
        self._spatial = (H1, W1)
        return emb.astype(np.float32)

    # ---------- 训练 ----------
    def fit(self, train_paths):
        print(f"[方法B·PaDiM] backbone={self.backbone_name}, layers={self.layers}, dims={self.dims}")
        print("[方法B·PaDiM] 提取训练集嵌入 ...")

        # 在线累加均值与协方差，避免同时保存所有嵌入（省内存）
        first = self._embed(data.load_image(train_paths[0]))
        P, d = first.shape
        sum1 = np.zeros((P, d), dtype=np.float32)
        sum2 = np.zeros((P, d, d), dtype=np.float32)
        n = 0
        for p in tqdm(train_paths, desc="训练图像"):
            emb = self._embed(data.load_image(p))
            sum1 += emb
            sum2 += np.einsum("pd,pe->pde", emb, emb)
            n += 1

        self.mean = (sum1 / n)
        cov = (sum2 / n - np.einsum("pd,pe->pde", self.mean, self.mean))
        cov = cov * (n / (n - 1))  # 无偏估计
        del sum1, sum2

        print("[方法B·PaDiM] 逐位置协方差特征分解 ...")
        self.eigvals, self.eigvecs = np.linalg.eigh(cov)  # (P,d), (P,d,d)
        self.eigvals = np.clip(self.eigvals, 0.0, None)   # 数值稳定性
        del cov
        self._trained = True
        print("  训练完成 [OK]")

    # ---------- 推理 ----------
    def predict(self, img_np):
        """输入 RGB uint8 数组，返回 (anomaly_map (H,W), image_score)。"""
        emb = self._embed(img_np)              # (P, d)
        diff = emb - self.mean                 # (P, d)
        proj = np.einsum("pd,pde->pe", diff, self.eigvecs)  # 投影到特征向量
        dist = (proj ** 2) / (self.eigvals + self.eps)      # 马氏距离
        score = dist.sum(axis=1)               # (P,)
        H1, W1 = self._spatial
        amap = score.reshape(H1, W1)

        amap = resize(amap, (config.IMAGE_SIZE, config.IMAGE_SIZE),
                      mode="reflect", anti_aliasing=True)
        amap = gaussian(amap, sigma=config.SMOOTH_SIGMA)
        image_score = float(amap.max())
        return amap.astype(np.float32), image_score
