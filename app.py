import streamlit as st  # Webアプリ作成
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # 感情分析モデル
import torch  # 機械学習
import pandas as pd  # データ分析
import numpy as np  # 数値計算
import matplotlib.pyplot as plt  # グラフ描画
import matplotlib.font_manager as fm  # フォント管理
from datetime import datetime  # 日付・時間管理
import json  # データ保存形式
import os  # ファイル操作
import seaborn as sns  # グラフ描画

# ---------------------------
# 設定
# ---------------------------
CSV_PATH = "diary_log.csv"  # 行動・感情ログを保存するCSVファイルのパス
HABITS_PATH = "habits.json"  # 行動リストを保存するJSONファイルのパス
DEFAULT_HABITS = ["バーピー", "読書", "瞑想"]  # デフォルトの行動リスト
MODEL_NAME = "natutaro/line-distilbert-base-japanese-finetuned-emotion"  # 感情分析モデルの名前

# 日本語フォント設定（Mac用）
font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc"  # 使用するフォントのパス
if os.path.exists(font_path):
    font_name = fm.FontProperties(fname=font_path).get_name()  # フォント名を取得
    plt.rcParams["font.family"] = font_name  # グラフ描画時に日本語フォントを使用

# ---------------------------
# モデル読み込み
# ---------------------------
@st.cache_resource  # 再実行時にキャッシュして高速化
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # トークナイザー読み込み
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)  # モデル読み込み
    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPUが使えればGPU、なければCPU
    model.to(device)  # モデルをデバイスに配置
    return tokenizer, model, device

tokenizer, model, device = load_model()  # モデル・トークナイザー・デバイスを取得

# ---------------------------
# 関数
# ---------------------------
# テキストから感情スコアを予測（否定的スコア・肯定的スコア）
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)  # テキストをモデル入力形式に変換
    inputs.pop('token_type_ids', None)  # token_type_ids が存在する場合は削除（モデルによって不要な場合あり）
    inputs = inputs.to(device)  # デバイス（GPU/CPU）に転送
    with torch.no_grad():  # 推論のみなので勾配計算は不要
        outputs = model(**inputs)  # モデルで予測
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]  # 確率に変換
    return probs[0], probs[1]  # [否定的スコア, 肯定的スコア] を返す

# 保存している行動リスト（習慣）を読み込み
def load_habits():
    if os.path.exists(HABITS_PATH):
        try:
            with open(HABITS_PATH, "r", encoding="utf-8") as f:
                habits = json.load(f)  # JSONから読み込み
            if isinstance(habits, list):  # リスト形式か確認
                return habits
        except Exception:
            pass  # 読み込み失敗時はデフォルト値を返す
    return DEFAULT_HABITS.copy()  # 初期設定の行動リストを返す

# 行動リスト（習慣）を保存
def save_habits(habits):
    with open(HABITS_PATH, "w", encoding="utf-8") as f:
        json.dump(habits, f, ensure_ascii=False, indent=2)  # JSON形式で保存

# 1行の記録をCSVに追加
def append_row_to_csv(row: dict):
    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:  # CSVが存在して中身がある場合
        df = pd.read_csv(CSV_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)  # 新しい行を追加
    else:
        df = pd.DataFrame([row])  # 新規作成
    df.to_csv(CSV_PATH, index=False)  # 保存

# CSVから全データを読み込み
def load_df():
    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:  # CSVが存在して中身がある場合
        return pd.read_csv(CSV_PATH)
    else:
        return pd.DataFrame(columns=["date", "time", "datetime", "text", "habits", "negative", "positive"])  # 空のDataFrame

# ---------------------------
# サイドバー
# ---------------------------
# サイドバー設定・データ管理
st.sidebar.header("設定 / データ管理")

# 行動（習慣）リストを読み込み
habits = load_habits()

# --- 行動リスト削除 ---
with st.sidebar.expander("行動リストの削除"):
    to_delete = st.multiselect("削除する行動を選択", habits)  # 削除対象の選択
    if st.button("行動を削除", key="delete_habit"):
        if to_delete:
            habits = [h for h in habits if h not in to_delete]  # 選択された行動を削除
            save_habits(habits)  # 更新保存
            st.sidebar.success("選択した行動を削除しました。")
            st.rerun()

# --- 保存データの確認・削除 ---
with st.sidebar.expander("保存データの確認・削除"):
    df_sidebar = load_df()  # 保存データ読み込み
    if not df_sidebar.empty:
        st.write(f"記録数: {len(df_sidebar)}")  # 記録数表示
        st.dataframe(df_sidebar.tail(10))  # 最新10件を表示
        indices = st.multiselect("削除する行の index", df_sidebar.index.tolist(), key="delete_rows")
        if st.button("選択行を削除", key="delete_rows_button") and indices:
            df_sidebar = df_sidebar.drop(index=indices).reset_index(drop=True)  # 選択行を削除
            df_sidebar.to_csv(CSV_PATH, index=False)  # 保存
            st.sidebar.success("選択行を削除しました")
            st.rerun()
    else:
        st.info("まだ記録がありません。")

# --- 全データ削除 ---
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
            os.remove(CSV_PATH)  # CSV削除
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
# --- セッションステート初期化 ---
if "user_text" not in st.session_state:
    st.session_state["user_text"] = ""  # ユーザーが入力した文章
if "selected_habits" not in st.session_state:
    st.session_state["selected_habits"] = []  # 選択した行動
if "entry_time" not in st.session_state:
    st.session_state["entry_time"] = datetime.now().time()  # 記録時刻
if "reset_user_text" not in st.session_state:
    st.session_state["reset_user_text"] = False  # 文章リセットフラグ
if "reset_selected_habits" not in st.session_state:
    st.session_state["reset_selected_habits"] = False  # 行動リセットフラグ
if "new_habit_input" not in st.session_state:
    st.session_state["new_habit_input"] = ""  # 新しい行動入力
if "reset_new_habit_input" not in st.session_state:
    st.session_state["reset_new_habit_input"] = False  # 新しい行動入力リセットフラグ
if "reset_entry_time" not in st.session_state:
    st.session_state["reset_entry_time"] = False  # 時刻リセットフラグ

# --- 入力リセットフラグが立っていたらリセット ---
if st.session_state["reset_user_text"]:
    st.session_state["user_text"] = ""  # 日記テキストをリセット
    st.session_state["reset_user_text"] = False
if st.session_state["reset_selected_habits"]:
    st.session_state["selected_habits"] = []  # 選択した行動をリセット
    st.session_state["reset_selected_habits"] = False
if st.session_state["reset_new_habit_input"]:
    st.session_state["new_habit_input"] = ""  # 新しい行動入力をリセット
    st.session_state["reset_new_habit_input"] = False
if st.session_state["reset_entry_time"]:
    st.session_state["entry_time"] = datetime.now().time()  # 時刻をリセット
    st.session_state["reset_entry_time"] = False

# --- メイン画面タイトルとサブタイトル ---
st.title("行動・感情相関ダッシュボード")  # アプリタイトル
st.subheader("日々の行動と感情の関係をデータで見える化")  # 補足説明

# --- 入力UI ---
entry_date = st.date_input("日付", value=datetime.today())  # 記録する日付
entry_time = st.time_input("時間", key="entry_time")  # 記録する時間
user_text = st.text_area("今日の日記", height=200, key="user_text")  # 日記テキスト入力

# --- 行動追加 ---
col1, col2 = st.columns([4, 1])
with col1:
    new_habit_input = st.text_input("新しい行動を追加（例：散歩）", key="new_habit_input")  # 入力欄
with col2:
    if st.button("➕ 追加"):
        nh = st.session_state["new_habit_input"].strip()  # 前後の空白を除去
        if nh and nh not in habits:
            habits.append(nh)  # 行動リストに追加
            save_habits(habits)  # JSON保存
            st.session_state["reset_new_habit_input"] = True  # 入力欄リセットフラグ
            st.rerun()  # 再描画
        elif not nh:
            st.warning("行動名を入力してください。")  # 入力なし警告
        else:
            st.warning("すでに存在します。")  # 重複警告

# --- 今日実施した行動選択 ---
selected_habits = st.multiselect(
    "今日行った行動",
    options=habits,
    key="selected_habits"
)

# --- 保存ボタン ---
# --- 感情分析＆データ保存 ---
if st.button("感情分析＆保存"):
    neg_score, pos_score = predict_sentiment(st.session_state["user_text"])  # 感情分析（ネガ・ポジ）

    append_row_to_csv({
        "date": entry_date.strftime("%Y-%m-%d"),  # 日付
        "time": entry_time.strftime("%H:%M"),     # 時間
        "datetime": datetime.combine(entry_date, entry_time).isoformat(),  # 日時
        "text": st.session_state["user_text"],    # 日記本文
        "habits": ",".join(st.session_state.get("selected_habits", [])),   # 選択した行動
        "negative": neg_score,  # ネガティブスコア
        "positive": pos_score   # ポジティブスコア
    })

    st.success("日記を保存しました！")  # 保存完了メッセージ

    # --- 入力リセット ---
    st.session_state["reset_user_text"] = True     # 日記本文をリセット
    st.session_state["reset_selected_habits"] = True  # 選択した行動をリセット
    st.session_state["reset_entry_time"] = True   # 入力時間を現在時刻にリセット
    st.rerun()  # アプリを再描画

# ---------------------------
# 最新結果表示
# ---------------------------
st.markdown("---")
st.subheader("2) 最新の解析結果")  # 最新の行動・感情分析結果を表示
df = load_df()  # 保存済みデータを読み込み
if not df.empty:
    latest = df.tail(1).iloc[-1]  # 最新の行を取得
    st.write(f"日付: {latest['date']}")  # 日付表示
    st.write(f"時間: {latest['time']}")  # 時間表示
    st.write(f"習慣: {latest['habits']}")  # 実施した習慣表示
    st.write(f"ポジティブ確率: {latest['positive']:.2%}")  # ポジティブスコア表示
    st.write(f"ネガティブ確率: {latest['negative']:.2%}")  # ネガティブスコア表示
    st.write("テキスト:")  # 日記テキスト見出し
    st.write(latest['text'])  # 日記本文表示
else:
    st.info("まだ解析結果がありません。")  # データなしの場合の表示

# ---------------------------
# 期間選択UI
# ---------------------------
st.markdown("---")
period = st.selectbox("表示期間", ["日", "週", "月", "年"], key="period_select")  # 表示期間を選択

def filter_df_by_period(df, period):  # 選択期間でデータをフィルターする関数
    df["date_dt"] = pd.to_datetime(df["date"])  # 日付列をdatetime型に変換
    today = pd.Timestamp.today()  # 今日の日付を取得
    if period == "日":
        return df[df["date_dt"] == today.normalize()]  # 今日のデータのみ
    elif period == "週":
        start = today - pd.Timedelta(days=today.weekday())  # 今週の月曜日
        end = start + pd.Timedelta(days=6)  # 今週の日曜日
        return df[(df["date_dt"] >= start.normalize()) & (df["date_dt"] <= end.normalize())]
    elif period == "月":
        return df[df["date_dt"].dt.month == today.month]  # 今月のデータ
    elif period == "年":
        return df[df["date_dt"].dt.year == today.year]  # 今年のデータ
    else:
        return df  # デフォルトは全期間

filtered_df = filter_df_by_period(df, period)  # フィルター後のデータ

# ---------------------------
# 相関分析
# ---------------------------
st.subheader("3) 習慣と感情の相関")  # 習慣とポジティブ感情の関係を表示
if not filtered_df.empty:
    for h in habits:  # 各習慣について
        # その習慣を行ったかどうかを0/1で列に追加
        filtered_df[f"has_{h}"] = filtered_df["habits"].fillna("").apply(lambda x: 1 if h in x else 0)
    # 習慣列とポジティブスコアの相関を計算
    corr_data = filtered_df[[f"has_{h}" for h in habits] + ["positive"]].corr()
    habit_corr = corr_data.loc[[f"has_{h}" for h in habits], "positive"]

    fig, ax = plt.subplots(figsize=(8, max(2, len(habits)*0.5)))  # 図の大きさ設定
    # ヒートマップで相関を可視化
    sns.heatmap(habit_corr.to_frame().T, annot=True, cmap="coolwarm", center=0, ax=ax, xticklabels=habits)
    ax.set_yticklabels(["positive"], rotation=0)  # y軸ラベル設定
    st.pyplot(fig)  # Streamlitで表示
else:
    st.info("データがありません。")  # データなしメッセージ

# ---------------------------
# 時間ごとの感情推移グラフ
# ---------------------------
st.markdown("---")  # 区切り線を表示
st.subheader("4) 感情推移グラフ")  # サブタイトルを表示

if not filtered_df.empty:  # データが存在する場合のみ処理
    chart_type = st.radio("グラフタイプ", ["折れ線グラフ", "棒グラフ"], horizontal=True)  # グラフタイプ選択UI

    # データ整形
    if period == "日":  # 日単位の表示の場合
        filtered_df["time_dt"] = pd.to_datetime(filtered_df["time"], format="%H:%M")  # 時刻列をdatetime型に変換
        df_sorted = filtered_df.sort_values("time_dt")  # 時刻順にソート
        x = df_sorted["time_dt"]  # X軸データ（datetime型）
        x_labels = df_sorted["time"]  # X軸ラベル（文字列）
        y_pos = df_sorted["positive"]  # ポジティブスコア
        y_neg = df_sorted["negative"]  # ネガティブスコア
        xlabel = "時間"  # X軸ラベル名
    elif period in ["週", "月"]:  # 週・月単位の表示の場合
        filtered_df["date_dt"] = pd.to_datetime(filtered_df["date"])  # 日付列をdatetime型に変換
        df_grouped = filtered_df.groupby(filtered_df["date_dt"].dt.date).agg({"positive": "mean", "negative": "mean"}).reset_index()  # 日付ごとに平均スコアを計算
        x = pd.to_datetime(df_grouped["date_dt"])  # X軸データ（datetime型）
        x_labels = df_grouped["date_dt"].astype(str)  # X軸ラベル（文字列）
        y_pos = df_grouped["positive"]  # ポジティブ平均スコア
        y_neg = df_grouped["negative"]  # ネガティブ平均スコア
        xlabel = "日付"  # X軸ラベル名
    elif period == "年":  # 年単位の表示の場合
        filtered_df["date_dt"] = pd.to_datetime(filtered_df["date"])  # 日付列をdatetime型に変換
        df_grouped = filtered_df.groupby(filtered_df["date_dt"].dt.month).agg({"positive": "mean", "negative": "mean"}).reset_index()  # 月ごとに平均スコアを計算
        x = df_grouped["date_dt"].astype(int)  # X軸データ（月番号1〜12）
        x_labels = df_grouped["date_dt"].apply(lambda m: f"{int(m)}月")  # X軸ラベル（1月〜12月）
        y_pos = df_grouped["positive"]  # ポジティブ平均スコア
        y_neg = df_grouped["negative"]  # ネガティブ平均スコア
        xlabel = "月"  # X軸ラベル名
    else:  # 不正な期間が選択された場合
        st.info("期間が不正です。")  # メッセージ表示
        st.stop()  # 処理停止

    # グラフ描画
    fig, ax = plt.subplots(figsize=(8, 4))  # 描画領域作成（幅8インチ、高さ4インチ）

    if chart_type == "折れ線グラフ":  # 折れ線グラフの場合
        ind = np.arange(len(x))  # X軸インデックス
        ax.plot(ind, y_pos, label="ポジティブ", marker="o", color="#FFA500")  # ポジティブスコアを折れ線で描画
        ax.plot(ind, y_neg, label="ネガティブ", marker="o", color="#2F4F4F")  # ネガティブスコアを折れ線で描画
        ax.set_xticks(ind)  # X軸の位置設定
        ax.set_xticklabels(x_labels, rotation=45)  # X軸ラベル設定（45度回転）

    else:  # 棒グラフの場合
        width = 0.4  # 棒の幅
        if period == "日":  # 日単位の場合は時間で棒をずらして表示
            ax.bar(x - pd.Timedelta(minutes=15), y_pos, width=width, label="ポジティブ", color="#FFA500", align="center")  # ポジティブ棒
            ax.bar(x + pd.Timedelta(minutes=15), y_neg, width=width, label="ネガティブ", color="#2F4F4F", align="center")  # ネガティブ棒
            ax.set_xticks(x)  # X軸の位置
            ax.set_xticklabels(x_labels, rotation=45)  # X軸ラベル
        else:  # 週・月・年単位の場合
            ind = np.arange(len(x))  # X軸インデックス
            ax.bar(ind - width/2, y_pos, width=width, label="ポジティブ", color="#FFA500", align="center")  # ポジティブ棒
            ax.bar(ind + width/2, y_neg, width=width, label="ネガティブ", color="#2F4F4F", align="center")  # ネガティブ棒
            ax.set_xticks(ind)  # X軸の位置
            ax.set_xticklabels(x_labels, rotation=45)  # X軸ラベル

    ax.set_xlabel(xlabel)  # X軸ラベル設定
    ax.set_ylabel("感情スコア")  # Y軸ラベル設定
    ax.set_title(f"{period}ごとの感情推移")  # グラフタイトル
    ax.legend()  # 凡例表示
    st.pyplot(fig)  # Streamlitにグラフ表示
else:
    st.info("データがありません。")  # データがない場合のメッセージ表示