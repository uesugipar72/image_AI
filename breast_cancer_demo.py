from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# データ読み込み
data = load_breast_cancer()
X, y = data.data, data.target

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# モデル作成と学習
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 精度評価
print("正解率:", accuracy_score(y_test, y_pred))
print("\n分類レポート:\n", classification_report(y_test, y_pred, target_names=data.target_names))

# 特徴量の重要度を取得
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]  # 大きい順にソート

# 上位10個の特徴量を表示
plt.figure(figsize=(10, 6))
plt.bar(range(10), importances[indices[:10]], align="center")
plt.xticks(range(10), [data.feature_names[i] for i in indices[:10]], rotation=45)
plt.title("特徴量の重要度（上位10）")
plt.tight_layout()
plt.show()