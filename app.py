# app.py ─────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pydeck as pdk
import seaborn as sns
from sklearn.cluster import DBSCAN

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta', 'SimHei', 'Microsoft JhengHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ------------------ 1. 資料讀取 ------------------
DATA_PATH = "完整風險評估結果.xlsx"
RAW_PATH  = "data.xlsx"

df = pd.read_excel(DATA_PATH)
raw_df = pd.read_excel(RAW_PATH)

# ------------------ 2. Streamlit 版面設定 ------------------
st.set_page_config(page_title="租客風險儀表板", layout="wide")
st.title("🏠 租客風險評分與熱區監測儀表板")

# 側邊欄篩選器 ------------------------------------------------
with st.sidebar:
    st.header("🔍 篩選條件")
    city_opts = sorted(raw_df["縣市"].dropna().unique())
    sel_city  = st.multiselect("選擇縣市", city_opts, default=city_opts)
    risk_opts = ["低", "中", "高"]
    sel_risk  = st.multiselect("風險等級", risk_opts, default=risk_opts)

mask = raw_df["縣市"].isin(sel_city) & df["風險等級"].isin(sel_risk)
view  = df[mask].reset_index(drop=True)
raw_view = raw_df.loc[mask, :]

# ------------------ 3. KPI Cards ------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("樣本數", f"{len(view):,}")
col2.metric("高風險戶", f"{(view['風險等級']=='高').sum():,}")
col3.metric("中風險戶", f"{(view['風險等級']=='中').sum():,}")
col4.metric("低風險戶", f"{(view['風險等級']=='低').sum():,}")

# ------------------ 4-1 風險等級直條圖 ------------------
st.subheader("📊 風險等級分佈")
fig_bar = px.histogram(
    view,
    x="風險等級",
    category_orders={"風險等級": ["低", "中", "高"]},
    color="風險等級",
    color_discrete_map={"低":"#1f77b4","中":"#ff7f0e","高":"#d62728"},
)
st.plotly_chart(fig_bar, use_container_width=True)

# ------------------ 4-2 租金 vs. 風險散佈 ------------------
st.subheader("💸 每坪租金 vs. 風險分數")
fig_scatter = px.scatter(
    view,
    x="每坪租金",
    y="風險分數",
    color="風險等級",
    hover_data=["契約_ID", "平均每日用水量", "是否為弱勢身分"],
    color_discrete_map={"低":"#1f77b4","中":"#ff7f0e","高":"#d62728"},
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------ 4-3 地理熱區圖 (KDE + PyDeck) ------------------
st.subheader("🗺️ 租客熱區圖 (KDE + 聚落偵測)")
geo = raw_view[["X座標", "Y座標"]].dropna()
geo.columns = ["lon", "lat"]

if not geo.empty:
    # KDE 圖
    fig_kde, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(
        data=geo,
        x="lon", y="lat",
        cmap="Reds", shade=True, alpha=0.7, ax=ax
    )
    ax.set_title("租客分佈熱區圖（KDE密度）", fontsize=14)
    ax.set_xlabel("經度（X座標）", fontsize=12)
    ax.set_ylabel("緯度（Y座標）", fontsize=12)
    ax.tick_params(labelsize=10)
    st.pyplot(fig_kde)

    # PyDeck 聚落圖（可展開）
    with st.expander("📍 查看互動地圖（聚落群）"):
        clustering = DBSCAN(eps=0.01, min_samples=5).fit(geo)
        geo["cluster"] = clustering.labels_
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=geo,
            get_position='[lon, lat]',
            get_fill_color="[200, 30, 0, 160]",
            get_radius=20,
        )
        st.pydeck_chart(
            pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=geo["lat"].mean(),
                    longitude=geo["lon"].mean(),
                    zoom=11,
                ),
                layers=[layer],
                tooltip={"text": "聚落群: {cluster}"},
            )
        )
else:
    st.warning("⚠️ 缺少有效地理座標資料，無法顯示熱區圖。")

# ------------------ 5. 明細資料表 ------------------
st.subheader("📄 完整明細 (可下載)")
st.dataframe(view)
st.download_button(
    "下載當前篩選結果 (CSV)",
    data=view.to_csv(index=False).encode("utf-8-sig"),
    file_name="篩選_租客風險結果.csv",
    mime="text/csv"
)
