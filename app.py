import streamlit as st  # Streamlitライブラリをインポート（Webアプリ作成用）
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # BERTモデルとトークナイザーの読み込み
import torch  # PyTorchライブラリをインポート（モデル推論用）
import pandas as pd  # データフレーム操作用のPandasをインポート
import numpy as np  # 数値計算用のNumPyをインポート
import matplotlib.pyplot as plt  # グラフ描画用Matplotlibをインポート
import japanize_matplotlib  # Matplotlibで日本語表示対応
import matplotlib.font_manager as fm  # フォント管理用モジュールをインポート
from datetime import datetime  # 日付・時間操作用のdatetimeをインポート
import pytz  # タイムゾーン管理用ライブラリをインポート
import json  # JSON読み書き用ライブラリをインポート
import os  # ファイル・パス操作用ライブラリをインポート
import seaborn as sns  # データ可視化用Seabornをインポート

JST = pytz.timezone("Asia/Tokyo")  # 日本標準時（JST）のタイムゾーンを設定

# ---------------------------
# 設定
# ---------------------------
CSV_PATH = "diary_log.csv"  # 日記データを保存するCSVファイルのパス
HABITS_PATH = "habits.json"  # 行動リストを保存するJSONファイルのパス
DEFAULT_HABITS = ["バーピー", "読書", "瞑想"]  # 初期の行動リスト
MODEL_NAME = "LoneWolfgang/bert-for-japanese-twitter-sentiment"  # 使用する感情分析モデル名

# 日本語フォント設定（Mac用）
font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc"  # Mac標準フォントのパス
if os.path.exists(font_path):  # フォントが存在する場合
    font_name = fm.FontProperties(fname=font_path).get_name()  # フォント名を取得
    plt.rcParams["font.family"] = font_name  # Matplotlibで日本語表示に設定

# ---------------------------
# モデル読み込み
# ---------------------------
@st.cache_resource  # Streamlitで一度読み込んだモデルをキャッシュして再利用
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # トークナイザーをロード
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)  # 分類モデルをロード
    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPUがあればcuda、なければCPU
    model.to(device)  # モデルを指定デバイスに移動
    return tokenizer, model, device  # トークナイザー・モデル・デバイスを返す

tokenizer, model, device = load_model()  # モデルを読み込み、変数に代入

# ---------------------------
# ラベル設定（3分類）
# ---------------------------
LABELS_EN = ["negative", "neutral", "positive"]  # 英語ラベル
LABELS_JP = ["ネガティブ", "ニュートラル", "ポジティブ"]  # 日本語ラベル
LABELS_MAP = dict(zip(LABELS_EN, LABELS_JP))  # 英語→日本語ラベルの辞書

# ---------------------------
# 感情予測関数
# ---------------------------
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)  # テキストをトークン化
    inputs = inputs.to(device)  # デバイスに転送
    with torch.no_grad():  # 勾配計算を無効化（推論用）
        outputs = model(**inputs)  # モデルで予測
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]  # ロジットを確率に変換
    label_id = np.argmax(probs)  # 最も確率の高いラベルIDを取得
    return LABELS_EN[label_id], probs  # ラベルと確率を返す

# ---------------------------
# 習慣管理関数
# ---------------------------
def load_habits():
    if os.path.exists(HABITS_PATH):  # ファイルが存在する場合
        try:
            with open(HABITS_PATH, "r", encoding="utf-8") as f:  # JSONファイルを開く
                habits = json.load(f)  # 習慣リストを読み込む
            if isinstance(habits, list):  # 読み込んだデータがリストなら
                return habits  # リストを返す
        except Exception:  # エラーが出た場合
            pass  # 無視して次へ
    return DEFAULT_HABITS.copy()  # ファイルがなければデフォルト習慣を返す

def save_habits(habits):
    with open(HABITS_PATH, "w", encoding="utf-8") as f:  # JSONファイルを書き込みモードで開く
        json.dump(habits, f, ensure_ascii=False, indent=2)  # 習慣リストをJSON形式で保存

# ---------------------------
# CSV管理関数
# ---------------------------
def append_row_to_csv(row: dict):
    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:  # CSVが存在して空でない場合
        df = pd.read_csv(CSV_PATH)  # 既存CSVを読み込む
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)  # 新しい行を追加
    else:
        df = pd.DataFrame([row])  # CSVがなければ新規DataFrameを作成
    df.to_csv(CSV_PATH, index=False)  # CSVに保存（インデックスは不要）

def load_df():
    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:  # CSVが存在して空でない場合
        return pd.read_csv(CSV_PATH)  # CSVを読み込んで返す
    else:
        return pd.DataFrame(columns=["date", "time", "datetime", "text", "habits", "label"] + LABELS_EN)  # 空のDataFrameを返す

# ---------------------------
# サイドバー
# ---------------------------
st.sidebar.header("設定 / データ管理")  # サイドバーに見出しを表示
habits = load_habits()  # 行動リストを読み込む

# ---------------------------
# 行動リスト削除用UI
# ---------------------------
with st.sidebar.expander("行動リストの削除"):  # 折りたたみセクション
    to_delete = st.multiselect("削除する行動を選択", habits)  # 削除対象を選択
    if st.button("行動を削除", key="delete_habit"):  # ボタンが押されたら
        if to_delete:  # 選択がある場合
            habits = [h for h in habits if h not in to_delete]  # 選択行動を削除
            save_habits(habits)  # 保存
            st.sidebar.success("選択した行動を削除しました。")  # 成功メッセージ
            st.rerun()  # アプリを再実行して更新

# ---------------------------
# 保存データ確認・削除用UI
# ---------------------------
with st.sidebar.expander("保存データの確認・削除"):
    df_sidebar = load_df()  # CSVデータを読み込む
    if not df_sidebar.empty:  # データがある場合
        st.write(f"記録数: {len(df_sidebar)}")  # データ件数表示
        st.dataframe(df_sidebar.tail(10))  # 最新10件を表示
        indices = st.multiselect("削除する行の index", df_sidebar.index.tolist(), key="delete_rows")  # 削除対象の行選択
        if st.button("選択行を削除", key="delete_rows_button") and indices:  # 削除ボタン押下時
            df_sidebar = df_sidebar.drop(index=indices).reset_index(drop=True)  # 選択行を削除
            df_sidebar.to_csv(CSV_PATH, index=False)  # CSVに保存
            st.sidebar.success("選択行を削除しました。")  # 成功メッセージ
            st.rerun()  # 再実行して更新
    else:
        st.info("まだ記録がありません。")  # データがない場合の案内

st.sidebar.markdown("---")  # サイドバーに区切り線を追加

# ---------------------------
# 全データ削除ボタン（初回確認）
# ---------------------------
if st.sidebar.button("全データ削除（注意）"):  # ボタン押下で確認フラグをセット
    st.session_state["confirm_delete"] = True

if st.session_state.get("confirm_delete", False):  # 初回確認中
    st.sidebar.warning("本当に全データを削除しますか？")  # 警告表示
    if st.sidebar.button("はい、削除"):  # ユーザーが削除を承認
        st.session_state["confirm_delete_second"] = True  # 最終確認フラグをセット
        st.rerun()  # アプリ再実行
    if st.sidebar.button("キャンセル"):  # ユーザーが削除をキャンセル
        st.session_state["confirm_delete"] = False  # フラグリセット
        st.session_state["confirm_delete_second"] = False
        st.rerun()

# ---------------------------
# 全データ削除ボタン（最終確認）
# ---------------------------
if st.session_state.get("confirm_delete_second", False):  # 最終確認中
    st.sidebar.error("最終確認: この操作は元に戻せません。")  # 赤色で警告
    if st.sidebar.button("最終削除実行"):  # 最終削除ボタン
        if os.path.exists(CSV_PATH):  # CSVが存在する場合削除
            os.remove(CSV_PATH)
        st.sidebar.success("全データを削除しました。")  # 成功メッセージ
        st.session_state["confirm_delete"] = False  # フラグリセット
        st.session_state["confirm_delete_second"] = False
        st.rerun()  # 再実行して更新
    if st.sidebar.button("キャンセル（削除しない）"):  # 最終確認でキャンセル
        st.session_state["confirm_delete"] = False
        st.session_state["confirm_delete_second"] = False
        st.rerun()

# ---------------------------
# メインUIのセッション状態初期化
# ---------------------------
for key in ["user_text", "selected_habits", "entry_time", "reset_user_text",
            "reset_selected_habits", "new_habit_input", "reset_new_habit_input",
            "reset_entry_time"]:
    # セッションにキーがなければ初期値を設定
    if key not in st.session_state:
        st.session_state[key] = "" if "text" in key or "new_habit_input" in key else [] if "selected" in key else False if "reset" in key else datetime.now(JST).time()

# ---------------------------
# リセット処理
# ---------------------------
if st.session_state["reset_user_text"]:  # 入力テキストのリセット
    st.session_state["user_text"] = ""
    st.session_state["reset_user_text"] = False

if st.session_state["reset_selected_habits"]:  # 選択中の習慣リストのリセット
    st.session_state["selected_habits"] = []
    st.session_state["reset_selected_habits"] = False

if st.session_state["reset_new_habit_input"]:  # 新規習慣入力のリセット
    st.session_state["new_habit_input"] = ""
    st.session_state["reset_new_habit_input"] = False

if st.session_state["reset_entry_time"]:  # 入力時刻のリセット
    st.session_state["entry_time"] = datetime.now(JST).time()
    st.session_state["reset_entry_time"] = False

# ---------------------------
# メイン画面UI
# ---------------------------
st.title("行動・感情相関ダッシュボード")  # アプリタイトル
st.subheader("日々の行動と感情の関係をデータで見える化")  # サブタイトル

now_jst = datetime.now(JST)  # 現在の日本時間を取得

# 日付入力（デフォルトは今日）
entry_date = st.date_input("日付", value=now_jst.date())

# 時刻入力（セッション初期化済みでなければ現在時刻を設定）
if "entry_time" not in st.session_state:
    st.session_state["entry_time"] = now_jst.time()
entry_time = st.time_input("時間", key="entry_time")  # 時刻入力ウィジェット

# 日記テキスト入力
user_text = st.text_area("今日の日記", height=200, key="user_text")

# 2カラムレイアウト（習慣追加用）
col1, col2 = st.columns([4,1])
with col1:
    new_habit_input = st.text_input("新しい行動を追加（例：散歩）", key="new_habit_input")  # 習慣入力
with col2:
    if st.button("➕ 追加"):  # 習慣追加ボタン
        nh = st.session_state["new_habit_input"].strip()
        if nh and nh not in habits:  # 新しい習慣なら追加
            habits.append(nh)
            save_habits(habits)  # JSONに保存
            st.session_state["reset_new_habit_input"] = True  # 入力リセットフラグ
            st.rerun()  # UI更新
        elif not nh:  # 空文字なら警告
            st.warning("行動名を入力してください。")
        else:  # 既存の習慣なら警告
            st.warning("すでに存在します。")

# 今日実施した習慣を選択するマルチセレクト
selected_habits = st.multiselect("今日行った行動", options=habits, key="selected_habits")

# --- 感情分析＆保存処理 ---
if st.button("感情分析＆保存"):  # ボタン押下で処理開始
    label, probs = predict_sentiment(st.session_state["user_text"])  # テキストの感情予測

    # 保存用データ辞書を作成
    row = {
        "date": entry_date.strftime("%Y-%m-%d"),  # 日付文字列
        "time": entry_time.strftime("%H:%M"),     # 時刻文字列
        "datetime": datetime.combine(entry_date, entry_time).isoformat(),  # ISO形式日時
        "text": st.session_state["user_text"],   # 日記テキスト
        "habits": ",".join(st.session_state.get("selected_habits", [])),  # 選択習慣
        "label": label  # 予測ラベル
    }
    # 各感情ラベルの確率も保存
    for i, l in enumerate(LABELS_EN):
        row[l] = probs[i]

    append_row_to_csv(row)  # CSVに追加

    st.success("日記を保存しました！")  # 保存完了メッセージ

    # 入力リセットフラグを立てる（UIを初期状態に戻す）
    st.session_state["reset_user_text"] = True
    st.session_state["reset_selected_habits"] = True
    st.session_state["reset_new_habit_input"] = True
    st.session_state["reset_entry_time"] = True

    st.rerun()  # UI更新

# ---------------------------
# 最新保存結果表示
# ---------------------------
st.subheader("2) 保存済みデータの確認")  # セクションタイトル
df = load_df()  # CSVを読み込み
if not df.empty:
    st.dataframe(df.tail(10))  # 最新10件を表示
else:
    st.info("まだ保存された日記はありません。")  # データなしメッセージ

# ---------------------------
# 期間選択UI
# ---------------------------
st.markdown("---")  # 区切り線
period = st.selectbox("表示期間", ["日", "週", "月", "年"], key="period_select")  # 表示期間選択

import pytz  # タイムゾーン管理
JST = pytz.timezone("Asia/Tokyo")  # 日本時間設定

def filter_df_by_period(df, period):
    # 日付列を datetime 型に変換
    df["date_dt"] = pd.to_datetime(df["date"])

    # JSTタイムゾーンに統一（タイムゾーン未設定なら localize）
    df["date_dt"] = df["date_dt"].dt.tz_localize(JST)  

    today = pd.Timestamp(datetime.now(JST))  # 現在日時（JST）

    if period == "日":
        today_norm = today.normalize()  # 日付のみの正規化
        return df[df["date_dt"] == today_norm]  # 今日のデータ
    elif period == "週":
        start = (today - pd.Timedelta(days=today.weekday())).normalize()  # 週の開始（月曜）
        end = (start + pd.Timedelta(days=6)).normalize()  # 週の終了（日曜）
        return df[(df["date_dt"] >= start) & (df["date_dt"] <= end)]  # 今週のデータ
    elif period == "月":
        return df[df["date_dt"].dt.month == today.month]  # 今月のデータ
    elif period == "年":
        return df[df["date_dt"].dt.year == today.year]  # 今年のデータ
    else:
        return df  # デフォルトは全データ

filtered_df = filter_df_by_period(df, period)  # フィルター適用

# ---------------------------
# 感情推移グラフ
# ---------------------------
st.subheader("3) 感情推移")  # セクションタイトル

if not filtered_df.empty:  # データがある場合
    chart_type = st.radio("グラフタイプ", ["折れ線グラフ", "棒グラフ"], horizontal=True)  # グラフ選択

    if period == "日":  # 日単位なら時間ごとに
        filtered_df["time_dt"] = pd.to_datetime(filtered_df["time"], format="%H:%M")  # 時間列を datetime に変換
        df_sorted = filtered_df.sort_values("time_dt")  # 時間順に並べ替え
        x_labels = df_sorted["time"]  # x軸は時間
        y_dict = {l: df_sorted[l] for l in LABELS_EN}  # 各感情スコア
        xlabel = "時間"
    elif period in ["週", "月"]:  # 週・月単位なら日付ごとの平均
        df_grouped = filtered_df.groupby(filtered_df["date_dt"].dt.date).agg({l: "mean" for l in LABELS_EN}).reset_index()
        x_labels = df_grouped["date_dt"].astype(str)  # x軸は日付
        y_dict = {l: df_grouped[l] for l in LABELS_EN}  # 各感情スコア平均
        xlabel = "日付"
    elif period == "年":  # 年単位なら月ごとの平均
        df_grouped = filtered_df.groupby(filtered_df["date_dt"].dt.month).agg({l: "mean" for l in LABELS_EN}).reset_index()
        x_labels = df_grouped["date_dt"].apply(lambda m: f"{int(m)}月")  # x軸は月
        y_dict = {l: df_grouped[l] for l in LABELS_EN}  # 各感情スコア平均
        xlabel = "月"

    fig, ax = plt.subplots(figsize=(10, 5))  # 描画領域
    ind = np.arange(len(x_labels))  # x軸インデックス
    colors = ["#228B22", "#808080", "#FF6347"]  # ポジティブ, ニュートラル, ネガティブ

    if chart_type == "折れ線グラフ":  # 折れ線グラフ描画
        for i, l in enumerate(LABELS_EN):
            ax.plot(ind, y_dict[l], label=LABELS_MAP[l], marker="o", color=colors[i])
        ax.set_xticks(ind)
        ax.set_xticklabels(x_labels, rotation=45)
    else:  # 棒グラフ描画
        width = 0.25
        for i, l in enumerate(LABELS_EN):
            ax.bar(ind + (i - 1)*width, y_dict[l], width=width, label=LABELS_MAP[l], color=colors[i])
        ax.set_xticks(ind)
        ax.set_xticklabels(x_labels, rotation=45)

    ax.set_xlabel(xlabel)  # x軸ラベル
    ax.set_ylabel("感情スコア")  # y軸ラベル
    ax.set_title(f"{period}ごとの感情推移")  # グラフタイトル
    ax.legend()  # 凡例表示
    st.pyplot(fig)  # Streamlitに描画
else:
    st.info("データがありません。")  # データなしメッセージ

# ---------------------------
# 行動と感情の相関（ヒートマップ）
# ---------------------------
st.subheader("4) 行動と感情の相関")  # セクションタイトル

if not filtered_df.empty:  # データがある場合
    # ユニークな行動のリストを作成
    habit_set = set()
    for hlist in filtered_df["habits"].dropna():
        habit_set.update(hlist.split(","))
    habit_list = sorted(list(habit_set))  # ソートしてリスト化

    if habit_list:  # 行動がある場合
        # 行動データフレームを0で初期化
        habit_df = pd.DataFrame(0, index=filtered_df.index, columns=habit_list)
        # 行動があれば1に置き換え
        for idx, hlist in filtered_df["habits"].items():
            for h in str(hlist).split(","):
                if h in habit_df.columns:
                    habit_df.at[idx, h] = 1

        # 感情ラベルを数値化
        label_map_num = {l: i for i, l in enumerate(LABELS_EN)}
        filtered_df["label_num"] = filtered_df["label"].map(label_map_num)

        # 行動データとラベルを結合して相関を計算
        corr_df = pd.concat([habit_df, filtered_df["label_num"]], axis=1)
        corr = corr_df.corr()["label_num"].drop("label_num")

        # ヒートマップ描画
        fig2, ax2 = plt.subplots(figsize=(8, max(4, len(habit_list)*0.5)))
        sns.heatmap(corr.to_frame().T, annot=True, cmap="coolwarm", center=0, ax=ax2)
        ax2.set_xlabel("行動")
        ax2.set_ylabel("感情との相関")
        st.pyplot(fig2)  # Streamlitに表示
    else:
        st.info("行動データがありません。")  # 行動が空の場合
else:
    st.info("データがありません。")  # データフレームが空の場合