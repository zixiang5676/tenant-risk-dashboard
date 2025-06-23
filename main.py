# ==========================================
# 0. 套件匯入 & 中文字體設定（✅ 新增）
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- 讓 Matplotlib 正確顯示中文與負號 ---
plt.rcParams["font.family"] = "Microsoft JhengHei"   # 微軟正黑體（Win10/11 預裝）
plt.rcParams["axes.unicode_minus"] = False           # 解決負號亂碼

# ==========================================
# 1. 讀取資料
# ==========================================
df = pd.read_excel("data.xlsx", engine="openpyxl")

# ==========================================
# 2. 結構化欄位前處理
# ==========================================
df["租約起日"] = pd.to_datetime(df["租約起日"], errors="coerce")
df["租約訖日"] = pd.to_datetime(df["租約訖日"], errors="coerce")
df["租期（月）"] = (
    (df["租約訖日"].dt.year - df["租約起日"].dt.year) * 12 +
    (df["租約訖日"].dt.month - df["租約起日"].dt.month)
)
df["每坪租金"] = df["簽約租金"] / df["實際使用坪數"]

# ==========================================
# 3. 模擬「每月用水」資料 → 產生行為特徵
# ==========================================
np.random.seed(42)
months = 12                     # 取 12 個月
sim_usage = np.random.normal(loc=300, scale=50, size=(len(df), months))

# 隨機挑 10 % 租戶 → 2 個月超低用水（異常）
ab_rows = np.random.choice(len(df), size=int(0.10 * len(df)), replace=False)
for r in ab_rows:
    ab_m = np.random.choice(months, size=2, replace=False)
    sim_usage[r, ab_m] = np.random.uniform(10, 50, size=2)

df["平均每日用水量"] = sim_usage.mean(axis=1).round(2)
df["用水標準差"]    = sim_usage.std(axis=1).round(2)

# ==========================================
# 4. LOF 異常偵測（10 % 異常比例）
# ==========================================
lof_cols = ["每坪租金", "平均每日用水量"]
# 轉數值; 若有 NaN → 補中位數 (較穩健)
for c in lof_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df[lof_cols] = df[lof_cols].fillna(df[lof_cols].median())

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
df["LOF_異常標記"] = (lof.fit_predict(df[lof_cols]) == -1)

# ==========================================
# 5. 趨勢偏離分數（簡易：Z 分數 * 10）
# ==========================================
mu, sigma = df["平均每日用水量"].mean(), df["平均每日用水量"].std()
df["趨勢偏離分數"] = ((df["平均每日用水量"] - mu) / sigma * 10).abs().round(0)

# ==========================================
# 6. 高風險標籤（邏輯可再調整）
# ==========================================
df["高風險標籤"] = (
    (df["每坪租金"] > 800) |
    (df["趨勢偏離分數"] >= 40) |
    (df["LOF_異常標記"]) |
    (
        (df["是否為弱勢身分"] == "是") &        # ⬅️ 弱勢身分加權 (計畫書要求)
        (df["趨勢偏離分數"] >= 20)
    )
)

# ==========================================
# 7. 隨機森林風險分數 (0–100)
# ==========================================
features = [
    "出租人年齡", "簽約租金", "實際使用坪數", "平均每日用水量",
    "租期（月）", "用水標準差", "趨勢偏離分數"
]
X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
y = df["高風險標籤"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
rf.fit(X_tr, y_tr)
print("=== 測試集報告 ===")
print(classification_report(y_te, rf.predict(X_te)))

df["風險分數"] = (rf.predict_proba(X)[:, 1] * 100).round(1)

# === 風險分級 (低<40, 中40~70, 高>70) ===
df["風險等級"] = pd.cut(
    df["風險分數"],
    bins=[-0.1, 40, 70, 100.1],
    labels=["低", "中", "高"]
)

# ==========================================
# 8. 視覺化：分佈直條圖 + 租金 vs. 風險
# ==========================================
# -- 8-1 風險等級分佈圖 --
plt.figure(figsize=(8, 5))
level_counts = df["風險等級"].value_counts().reindex(["低", "中", "高"]).fillna(0)
plt.bar(level_counts.index, level_counts.values)
plt.title("風險等級分佈圖")
plt.xlabel("風險等級")
plt.ylabel("筆數")
plt.tight_layout()
plt.savefig("風險等級分佈圖.png")
plt.close()

# -- 8-2 租金 vs. 風險散佈 --
plt.figure(figsize=(8, 5))
plt.scatter(df["簽約租金"], df["風險分數"], alpha=0.4)
plt.title("簽約租金 vs. 風險分數")
plt.xlabel("簽約租金(元)")
plt.ylabel("風險分數(0-100)")
plt.tight_layout()
plt.savefig("租金_風險散佈圖.png")
plt.close()

print("圖表已存為 風險等級分佈圖.png、租金_風險散佈圖.png")

# ==========================================
# 9. 匯出完整結果
# ==========================================
out_cols = [
    "契約_ID", "每坪租金", "平均每日用水量", "用水標準差", "趨勢偏離分數",
    "LOF_異常標記", "是否為弱勢身分", "高風險標籤", "風險分數", "風險等級"
]
df[out_cols].to_excel("完整風險評估結果.xlsx", index=False)
print("已匯出：完整風險評估結果.xlsx")
