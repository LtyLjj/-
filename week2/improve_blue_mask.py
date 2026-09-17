# -*- coding: utf-8 -*-
"""
初始图: mask_image.png, final_result.png
去掉瓶盖上的黑点，结果写入 output/

运行: python improve_blue_mask.py
"""

from pathlib import Path

import cv2
import numpy as np


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def read_bgr(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def linear_gray(img, a=1.2, b=10):
    out = img.astype(np.float32) * a + b
    return np.clip(out, 0, 255).astype(np.uint8)


def gamma_gray(img, gamma=0.8):
    r = img.astype(np.float32) / 255.0
    return np.clip(np.power(r, gamma) * 255.0, 0, 255).astype(np.uint8)


def contrast_stretch(img, a=40, b=200, c=0, d=255):
    x = img.astype(np.float32)
    out = np.zeros_like(x)
    m1 = x < a
    m2 = (x >= a) & (x <= b)
    m3 = x > b
    out[m1] = (c / max(a, 1)) * x[m1]
    out[m2] = ((d - c) / max(b - a, 1)) * (x[m2] - a) + c
    out[m3] = ((255 - d) / max(255 - b, 1)) * (x[m3] - b) + d
    return np.clip(out, 0, 255).astype(np.uint8)


def threshold_gray(img, t=127):
    out = np.zeros_like(img)
    out[img >= t] = 255
    return out


def fix_mask(mask_gray):
    """灰度变换后阈值，再把最大白色连通域内部黑点填实"""
    x = linear_gray(mask_gray, a=1.2, b=5)
    x = contrast_stretch(x, a=40, b=200, c=0, d=255)
    x = gamma_gray(x, gamma=0.8)
    binary = threshold_gray(x, t=127)

    # 只保留最大白色区域（瓶盖），并填充其内部黑点
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return binary
    cap = max(cnts, key=cv2.contourArea)
    solid = np.zeros_like(binary)
    cv2.drawContours(solid, [cap], -1, 255, thickness=-1)
    return solid


def fix_final(final_bgr, mask_fixed):
    """用实心 Mask 去掉 Final Result 瓶盖黑点"""
    if mask_fixed.shape[:2] != final_bgr.shape[:2]:
        mask_fixed = cv2.resize(
            mask_fixed,
            (final_bgr.shape[1], final_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    out = np.zeros_like(final_bgr)
    fg = mask_fixed == 255
    # 正常蓝色大约 sum>160；更暗的当作瓶盖黑点填掉
    bright = fg & (final_bgr.sum(axis=2) > 160)
    if np.any(bright):
        mean_bgr = final_bgr[bright].mean(axis=0)
    else:
        mean_bgr = np.array([180.0, 90.0, 40.0])

    out[fg] = final_bgr[fg]
    dark = fg & (final_bgr.sum(axis=2) <= 160)
    out[dark] = mean_bgr
    return out


def main():
    here = Path(__file__).resolve().parent
    out_dir = here / "output"
    out_dir.mkdir(exist_ok=True)

    mask = read_gray(here / "mask_image.png")
    final = read_bgr(here / "final_result.png")
    if mask is None or final is None:
        raise SystemExit(1)

    # 统一存成普通图，方便以后直接用
    cv2.imwrite(str(here / "mask_image.png"), mask)
    cv2.imwrite(str(here / "final_result.png"), final)

    mask_after = fix_mask(mask)
    final_after = fix_final(final, mask_after)

    cv2.imwrite(str(out_dir / "mask_before.png"), mask)
    cv2.imwrite(str(out_dir / "mask_after.png"), mask_after)
    cv2.imwrite(str(out_dir / "final_before.png"), final)
    cv2.imwrite(str(out_dir / "final_after.png"), final_after)


if __name__ == "__main__":
    main()
