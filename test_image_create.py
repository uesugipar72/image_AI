import cv2
import numpy as np
import os

outdir = "dataset"
classes = ["plane", "piled", "bowl"]
os.makedirs(outdir, exist_ok=True)

for cls in classes:
    os.makedirs(os.path.join(outdir, cls), exist_ok=True)

# 平面型: 単純な円
for i in range(10):
    img = np.zeros((256, 256, 3), np.uint8) + 255
    cv2.circle(img, (128, 128), 60, (0, 0, 0), -1)
    cv2.imwrite(f"dataset/plane/plane_{i}.jpg", img)

# 重積型: 重なった円
for i in range(10):
    img = np.zeros((256, 256, 3), np.uint8) + 255
    cv2.circle(img, (100, 128), 50, (0, 0, 0), -1)
    cv2.circle(img, (150, 128), 50, (0, 0, 0), -1)
    cv2.imwrite(f"dataset/piled/piled_{i}.jpg", img)

# おわん型: 輪郭だけの円（中心は空洞）
for i in range(10):
    img = np.zeros((256, 256, 3), np.uint8) + 255
    cv2.circle(img, (128, 128), 60, (0, 0, 0), 5)
    cv2.imwrite(f"dataset/bowl/bowl_{i}.jpg", img)
