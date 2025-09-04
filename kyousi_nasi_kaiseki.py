import cv2
import numpy as np
import glob
import os
import pandas as pd
from sklearn.cluster import KMeans
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. 特徴量抽出関数 ===
def extract_features(image_path, save_mask_dir=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] 画像を読み込めませんでした: {image_path}")
        return None, None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    # 二値化
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    th_clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)

    # ラベリング
    labeled = label(th_clean)
    regions = regionprops(labeled)
    if len(regions) == 0:
        return None, None
    largest_region = max(regions, key=lambda r: r.area)

    # 特徴量計算
    area = largest_region.area
    perimeter = largest_region.perimeter if largest_region.perimeter > 0 else 1
    circularity = 4 * np.pi * area / (perimeter ** 2)
    aspect_ratio = (
        largest_region.major_axis_length / largest_region.minor_axis_length
        if largest_region.minor_axis_length > 0 else 1
    )
    solidity = largest_region.solidity

    mask = largest_region.filled_image
    hole_area = mask.size - np.sum(mask)
    hole_ratio = hole_area / mask.size

    glcm = graycomatrix(gray, [5], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    features = [
        area, perimeter, circularity,
        aspect_ratio, solidity, hole_ratio,
        contrast, homogeneity
    ]

    # マスクを保存（確認用）
    if save_mask_dir:
        os.makedirs(save_mask_dir, exist_ok=True)
        mask_img = img_as_ubyte(mask) * 255
        mask_filename = os.path.join(save_mask_dir, os.path.basename(image_path))
        cv2.imwrite(mask_filename, mask_img)

    return features, mask


# === 2. データセット読み込み ===
dataset_dir = "dataset"
mask_dir = "masks"  # 計算に使った領域マスクの保存先
X = []
files = []

for file in glob.glob(os.path.join(dataset_dir, "*/*.jpg")):
    feats, _ = extract_features(file, save_mask_dir=mask_dir)
    if feats is not None:
        X.append(feats)
        files.append(file)

X = np.array(X)

# === 2.1 特徴量をDataFrame化 ===
columns = [
    "area", "perimeter", "circularity",
    "aspect_ratio", "solidity", "hole_ratio",
    "contrast", "homogeneity"
]
df = pd.DataFrame(X, columns=columns, index=files)
df.to_csv("features.csv", encoding="utf-8-sig")
print("\n=== 抽出された特徴量 (先頭5件) ===")
print(df.head())


# === 3. KMeansクラスタリング ===
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

print("\n=== 各クラスタの中心（特徴量平均） ===")
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"Cluster {i}: Center={center}")

# === 4. クラスタ番号 → 病理ラベルの対応付け ===
cluster_to_label = {
    0: "plane",
    1: "bowl",
    2: "piled"
}

# === 5. 新規画像の予測関数 ===
def predict_image(image_path):
    feats, _ = extract_features(image_path)
    if feats is None:
        return "No cell group detected"
    cluster_id = kmeans.predict([feats])[0]
    label_name = cluster_to_label.get(cluster_id, f"Cluster{cluster_id}")
    return label_name

# === 6. 新規画像でテスト ===
test_image = "new_sample.jpg"
print(f"\n新規画像: {test_image} -> {predict_image(test_image)}")

# === 7. 全画像のクラスタリング結果表示 ===
print("\n=== 全画像のクラスタリング結果 ===")
for file, label in zip(files, labels):
    print(f"画像: {file} -> クラスタ: {label} -> ラベル: {cluster_to_label.get(label, 'Unknown')}")

# === 8. 可視化: ヒストグラム（クラスタ別） ===
df["cluster"] = labels

for col in columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=col, hue="cluster", kde=True, palette="Set1", element="step")
    plt.title(f"Feature distribution: {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()

# === 9. 可視化: 散布図（2次元プロット） ===
plt.figure(figsize=(6, 6))
sns.scatterplot(
    data=df, x="area", y="circularity",
    hue="cluster", palette="Set1", s=60, edgecolor="k"
)
plt.title("Scatter plot: area vs circularity")
plt.xlabel("Area")
plt.ylabel("Circularity")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()