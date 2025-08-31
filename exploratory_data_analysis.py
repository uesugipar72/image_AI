from sklearn.datasets import load_breast_cancer
import pandas as pd

# データ読み込み
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target  # ラベル追加（0=悪性, 1=良性）

# データの先頭を確認
print(df.head())

# 基本情報
print(df.info())
print(df.describe())

# クラスの分布（悪性/良性の件数）
print(df["target"].value_counts())
