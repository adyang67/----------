# -*- coding: utf-8 -*-
"""下载并解压 MVTec AD 数据集（约 5.26 GB，请预留磁盘空间与时间）。

数据源：Hugging Face 国内镜像 hf-mirror.com 上的公开副本
（原 mydrive.ch 直链已 404 失效）。
下载用 curl（支持断点续传），中断后重跑本脚本会接着上次进度继续。

注意：这个压缩包是「平铺」结构（顶层直接是 bottle/cable/... 等类别目录），
所以必须解压到 config.MVTEC_DIR（data/mvtec_anomaly_detection/）才对得上
config 里的路径，而不是 data/。
"""
import os
import sys
import stat
import tarfile
import subprocess

import config

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

# MVTec AD 完整压缩包（hf-mirror 镜像）
URL = ("https://hf-mirror.com/datasets/micguida1/mvtech_anomaly_detection/"
       "resolve/main/mvtec_anomaly_detection.tar.xz")

# 完整文件字节数（用于判断是否已下完）
EXPECTED_SIZE = 5264982680

# 平铺压缩包 -> 解压到 mvtec_anomaly_detection/ 下
EXTRACT_DIR = config.MVTEC_DIR


def _download(url, dest):
    print(f"开始下载: {url}")
    print(f"保存到: {dest}")
    print("（curl 断点续传，中断后重跑本脚本可继续）")
    # -L 跟随重定向；-C - 断点续传；--retry 自动重试
    cmd = ["curl", "-L", "-C", "-", "--retry", "5", "--retry-delay", "3",
           "-o", dest, url]
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"curl 下载失败，退出码 {rc}，请检查网络后重跑本脚本")


def _make_writable(path):
    """递归去掉只读属性，避免解压出的只读文件/目录后续删不掉。"""
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            p = os.path.join(root, name)
            try:
                os.chmod(p, os.stat(p).st_mode | stat.S_IWRITE)
            except OSError:
                pass


def _extract(tar_path, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    print(f"正在解压（.tar.xz），可能需要几分钟 ... -> {dest_dir}")
    with tarfile.open(tar_path, "r:xz") as tf:
        tf.extractall(dest_dir)
    _make_writable(dest_dir)
    print(f"解压完成 -> {dest_dir}")


def main():
    config.ensure_dirs()
    tar_path = os.path.join(config.DATA_DIR, "mvtec_anomaly_detection.tar.xz")

    # 用「目标类别目录是否存在」判断是否已解压好，而不是只看 mvtec_anomaly_detection 目录
    cat_dir = os.path.join(EXTRACT_DIR, config.CATEGORY)
    if os.path.isdir(cat_dir):
        print(f"数据集已存在: {cat_dir}")
        return

    if os.path.exists(tar_path) and os.path.getsize(tar_path) >= EXPECTED_SIZE:
        print(f"发现已下载完整的压缩包: {tar_path} "
              f"({os.path.getsize(tar_path) / 1e9:.2f} GB)")
    else:
        if os.path.exists(tar_path):
            print(f"发现未下载完成的压缩包（{os.path.getsize(tar_path) / 1e9:.2f} GB），"
                  "继续下载 ...")
        _download(URL, tar_path)

    _extract(tar_path, EXTRACT_DIR)

    if os.path.isdir(cat_dir):
        print("数据集准备完成 [OK]")
    else:
        print("警告：未在预期路径找到数据集，请检查解压结果。")


if __name__ == "__main__":
    main()
