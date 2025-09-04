import cv2
import numpy as np
import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops  # ✅ 最新版対応

# === 1. 特徴量抽出関数 ===
def extract_features(image_path):
    print(f"[INFO] 処理中の画像: {image_path}")  # ★ 追加（どのファイルを処理しているか表示）

    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] 画像を読み込めませんでした: {image_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (256, 256))

    # グレースケール変換
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    # 二値化（Otsu法）
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ノイズ除去（モルフォロジー開閉）
    kernel = np.ones((3, 3), np.uint8)
    th_clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)

    # ラベリング
    labeled = label(th_clean)
    regions = regionprops(labeled)

    # 最大面積の領域を取得（細胞集団の主輪郭と仮定）
    if len(regions) == 0:
        return None
    largest_region = max(regions, key=lambda r: r.area)

    # 形状特徴
    area = largest_region.area
    perimeter = largest_region.perimeter if largest_region.perimeter > 0 else 1
    circularity = 4 * np.pi * area / (perimeter ** 2)
    aspect_ratio = (
        largest_region.major_axis_length / largest_region.minor_axis_length
        if largest_region.minor_axis_length > 0 else 1
    )
    solidity = largest_region.solidity

    # 中心部の空洞度（おわん型検出の目安）
    mask = largest_region.filled_image
    hole_area = mask.size - np.sum(mask)  # filled領域 - 実際の領域
    hole_ratio = hole_area / mask.size

    # テクスチャ特徴（GLCM）
    glcm = graycomatrix(gray, [5], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    features = [
        area, perimeter, circularity,
        aspect_ratio, solidity, hole_ratio,
        contrast, homogeneity
    ]
    return features


# === 2. データセット読み込み ===
# フォルダ構造：
# dataset/
#   plane/   -> 平面型
#   piled/   -> 重積型
#   bowl/    -> おわん型
dataset_dir = "dataset"
classes = ["plane", "piled", "bowl"]

X = []
y = []

for label_name in classes:
    folder = os.path.join(dataset_dir, label_name)
    for file in glob.glob(os.path.join(folder, "*.jpg")):
        feats = extract_features(file)
        if feats is not None:
            X.append(feats)
            y.append(label_name)

X = np.array(X)
y = np.array(y)

# === 3. 学習・評価 ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# === 4. 新規画像の分類例 ===
def predict_image(image_path):
    print(f"[PREDICT] 分類対象: {image_path}")  # ★ 追加（新規画像分類のときに表示）
    
    feats = extract_features(image_path)
    if feats is None:
        return "No cell group detected"
    pred = clf.predict([feats])
    return pred[0]

test_image = "new_sample.jpg"
print(f"{test_image} -> {predict_image(test_image)}")
