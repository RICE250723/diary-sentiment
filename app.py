import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime
import json
import os
import seaborn as sns

# ---------------------------
# 設定
# ---------------------------
CSV_PATH = "diary_log.csv"
HABITS_PATH = "habits.json"
DEFAULT_HABITS = ["バーピー", "読書", "瞑想"]
MODEL_NAME = "LoneWolfgang/bert-for-japanese-twitter-sentiment"  # 新モデル

# 日本語フォント設定（Mac用）
font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc"
if os.path.exists(font_path):
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = font_name

# ---------------------------
# モデル読み込み
# ---------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# ---------------------------
# ラベル設定（3分類）
# ---------------------------
LABELS_EN = ["negative", "neutral", "positive"]
LABELS_JP = ["ネガティブ", "ニュートラル", "ポジティブ"]
LABELS_MAP = dict(zip(LABELS_EN, LABELS_JP))

# ---------------------------
# 感情予測関数
# ---------------------------
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    label_id = np.argmax(probs)
    return LABELS_EN[label_id], probs

# ---------------------------
# 習慣管理関数
# ---------------------------
def load_habits():
    if os.path.exists(HABITS_PATH):
        try:
            with open(HABITS_PATH, "r", encoding="utf-8") as f:
                habits = json.load(f)
            if isinstance(habits, list):
                return habits
        except Exception:
            pass
    return DEFAULT_HABITS.copy()

def save_habits(habits):
    with open(HABITS_PATH, "w", encoding="utf-8") as f:
        json.dump(habits, f, ensure_ascii=False, indent=2)

# ---------------------------
# CSV管理関数
# ---------------------------
def append_row_to_csv(row: dict):
    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
        df = pd.read_csv(CSV_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(CSV_PATH, index=False)

def load_df():
    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
        return pd.read_csv(CSV_PATH)
    else:
        return pd.DataFrame(columns=["date", "time", "datetime", "text", "habits", "label"] + LABELS_EN)

# ---------------------------
# サイドバー
# ---------------------------
st.sidebar.header("設定 / データ管理")
habits = load_habits()

with st.sidebar.expander("行動リストの削除"):
    to_delete = st.multiselect("削除する行動を選択", habits)
    if st.button("行動を削除", key="delete_habit"):
        if to_delete:
            habits = [h for h in habits if h not in to_delete]
            save_habits(habits)
            st.sidebar.success("選択した行動を削除しました。")
            st.rerun()

with st.sidebar.expander("保存データの確認・削除"):
    df_sidebar = load_df()
    if not df_sidebar.empty:
        st.write(f"記録数: {len(df_sidebar)}")
        st.dataframe(df_sidebar.tail(10))
        indices = st.multiselect("削除する行の index", df_sidebar.index.tolist(), key="delete_rows")
        if st.button("選択行を削除", key="delete_rows_button") and indices:
            df_sidebar = df_sidebar.drop(index=indices).reset_index(drop=True)
            df_sidebar.to_csv(CSV_PATH, index=False)
            st.sidebar.success("選択行を削除しました。")
            st.rerun()
    else:
        st.info("まだ記録がありません。")

st.sidebar.markdown("---")
if st.sidebar.button("全データ削除（注意）"):
    st.session_state["confirm_delete"] = True

if st.session_state.get("confirm_delete", False):
    st.sidebar.warning("本当に全データを削除しますか？")
    if st.sidebar.button("はい、削除"):
        st.session_state["confirm_delete_second"] = True
        st.rerun()
    if st.sidebar.button("キャンセル"):
        st.session_state["confirm_delete"] = False
        st.session_state["confirm_delete_second"] = False
        st.rerun()

if st.session_state.get("confirm_delete_second", False):
    st.sidebar.error("最終確認: この操作は元に戻せません。")
    if st.sidebar.button("最終削除実行"):
        if os.path.exists(CSV_PATH):
            os.remove(CSV_PATH)
        st.sidebar.success("全データを削除しました。")
        st.session_state["confirm_delete"] = False
        st.session_state["confirm_delete_second"] = False
        st.rerun()
    if st.sidebar.button("キャンセル（削除しない）"):
        st.session_state["confirm_delete"] = False
        st.session_state["confirm_delete_second"] = False
        st.rerun()

# ---------------------------
# メインUI
# ---------------------------
for key in ["user_text", "selected_habits", "entry_time", "reset_user_text",
            "reset_selected_habits", "new_habit_input", "reset_new_habit_input",
            "reset_entry_time"]:
    if key not in st.session_state:
        st.session_state[key] = "" if "text" in key or "new_habit_input" in key else [] if "selected" in key else False if "reset" in key else datetime.now().time()

# --- リセット処理 ---
if st.session_state["reset_user_text"]:
    st.session_state["user_text"] = ""
    st.session_state["reset_user_text"] = False

if st.session_state["reset_selected_habits"]:
    st.session_state["selected_habits"] = []
    st.session_state["reset_selected_habits"] = False

if st.session_state["reset_new_habit_input"]:
    st.session_state["new_habit_input"] = ""
    st.session_state["reset_new_habit_input"] = False

if st.session_state["reset_entry_time"]:
    st.session_state["entry_time"] = datetime.now().time()
    st.session_state["reset_entry_time"] = False

st.title("行動・感情相関ダッシュボード")
st.subheader("日々の行動と感情の関係をデータで見える化")

entry_date = st.date_input("日付", value=datetime.today())
entry_time = st.time_input("時間", key="entry_time")
user_text = st.text_area("今日の日記", height=200, key="user_text")

col1, col2 = st.columns([4,1])
with col1:
    new_habit_input = st.text_input("新しい行動を追加（例：散歩）", key="new_habit_input")
with col2:
    if st.button("➕ 追加"):
        nh = st.session_state["new_habit_input"].strip()
        if nh and nh not in habits:
            habits.append(nh)
            save_habits(habits)
            st.session_state["reset_new_habit_input"] = True
            st.rerun()
        elif not nh:
            st.warning("行動名を入力してください。")
        else:
            st.warning("すでに存在します。")

selected_habits = st.multiselect("今日行った行動", options=habits, key="selected_habits")

# --- 感情分析＆保存 ---
if st.button("感情分析＆保存"):
    label, probs = predict_sentiment(st.session_state["user_text"])

    row = {
        "date": entry_date.strftime("%Y-%m-%d"),
        "time": entry_time.strftime("%H:%M"),
        "datetime": datetime.combine(entry_date, entry_time).isoformat(),
        "text": st.session_state["user_text"],
        "habits": ",".join(st.session_state.get("selected_habits", [])),
        "label": label
    }
    for i, l in enumerate(LABELS_EN):
        row[l] = probs[i]

    append_row_to_csv(row)

    st.success("日記を保存しました！")

    # --- 入力リセットフラグを立てる ---
    st.session_state["reset_user_text"] = True
    st.session_state["reset_selected_habits"] = True
    st.session_state["reset_new_habit_input"] = True
    st.session_state["reset_entry_time"] = True

    st.rerun()
# ---------------------------
# 最新結果表示
# ---------------------------
st.subheader("2) 保存済みデータの確認")
df = load_df()
if not df.empty:
    st.dataframe(df.tail(10))
else:
    st.info("まだ保存された日記はありません。")

# ---------------------------
# 期間選択UI
# ---------------------------
st.markdown("---")
period = st.selectbox("表示期間", ["日", "週", "月", "年"], key="period_select")

def filter_df_by_period(df, period):
    df["date_dt"] = pd.to_datetime(df["date"])
    today = pd.Timestamp.today()
    if period == "日":
        return df[df["date_dt"] == today.normalize()]
    elif period == "週":
        start = today - pd.Timedelta(days=today.weekday())
        end = start + pd.Timedelta(days=6)
        return df[(df["date_dt"] >= start.normalize()) & (df["date_dt"] <= end.normalize())]
    elif period == "月":
        return df[df["date_dt"].dt.month == today.month]
    elif period == "年":
        return df[df["date_dt"].dt.year == today.year]
    else:
        return df

filtered_df = filter_df_by_period(df, period)

# ---------------------------
# 感情推移グラフ
# ---------------------------
st.subheader("3) 感情推移")

if not filtered_df.empty:
    chart_type = st.radio("グラフタイプ", ["折れ線グラフ", "棒グラフ"], horizontal=True)

    if period == "日":
        filtered_df["time_dt"] = pd.to_datetime(filtered_df["time"], format="%H:%M")
        df_sorted = filtered_df.sort_values("time_dt")
        x_labels = df_sorted["time"]
        y_dict = {l: df_sorted[l] for l in LABELS_EN}
        xlabel = "時間"
    elif period in ["週", "月"]:
        df_grouped = filtered_df.groupby(filtered_df["date_dt"].dt.date).agg({l: "mean" for l in LABELS_EN}).reset_index()
        x_labels = df_grouped["date_dt"].astype(str)
        y_dict = {l: df_grouped[l] for l in LABELS_EN}
        xlabel = "日付"
    elif period == "年":
        df_grouped = filtered_df.groupby(filtered_df["date_dt"].dt.month).agg({l: "mean" for l in LABELS_EN}).reset_index()
        x_labels = df_grouped["date_dt"].apply(lambda m: f"{int(m)}月")
        y_dict = {l: df_grouped[l] for l in LABELS_EN}
        xlabel = "月"

    fig, ax = plt.subplots(figsize=(10, 5))
    ind = np.arange(len(x_labels))
    colors = ["#228B22", "#808080", "#FF6347"]  # ポジ,ニュートラル,ネガ

    if chart_type == "折れ線グラフ":
        for i, l in enumerate(LABELS_EN):
            ax.plot(ind, y_dict[l], label=LABELS_MAP[l], marker="o", color=colors[i])
        ax.set_xticks(ind)
        ax.set_xticklabels(x_labels, rotation=45)
    else:
        width = 0.25
        for i, l in enumerate(LABELS_EN):
            ax.bar(ind + (i - 1)*width, y_dict[l], width=width, label=LABELS_MAP[l], color=colors[i])
        ax.set_xticks(ind)
        ax.set_xticklabels(x_labels, rotation=45)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("感情スコア")
    ax.set_title(f"{period}ごとの感情推移")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("データがありません。")

# ---------------------------
# 行動と感情の相関（ヒートマップ）
# ---------------------------
st.subheader("4) 行動と感情の相関")

if not filtered_df.empty:
    habit_set = set()
    for hlist in filtered_df["habits"].dropna():
        habit_set.update(hlist.split(","))
    habit_list = sorted(list(habit_set))

    if habit_list:
        habit_df = pd.DataFrame(0, index=filtered_df.index, columns=habit_list)
        for idx, hlist in filtered_df["habits"].items():
            for h in str(hlist).split(","):
                if h in habit_df.columns:
                    habit_df.at[idx, h] = 1
        label_map_num = {l: i for i, l in enumerate(LABELS_EN)}
        filtered_df["label_num"] = filtered_df["label"].map(label_map_num)
        corr_df = pd.concat([habit_df, filtered_df["label_num"]], axis=1)
        corr = corr_df.corr()["label_num"].drop("label_num")
        fig2, ax2 = plt.subplots(figsize=(8, max(4, len(habit_list)*0.5)))
        sns.heatmap(corr.to_frame().T, annot=True, cmap="coolwarm", center=0, ax=ax2)
        ax2.set_xlabel("行動")
        ax2.set_ylabel("感情との相関")
        st.pyplot(fig2)
    else:
        st.info("行動データがありません。")
else:
    st.info("データがありません。")