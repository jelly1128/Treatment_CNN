import pandas as pd

def extract_best_label(row, threshold):
    """
    row: DataFrameの1行分のデータ（Series）
    threshold: 閾値
    対象は Pred_Class_6 ～ Pred_Class_14 の各値
    閾値を超えているラベルがある場合、その中で最も確率が高いラベル番号（文字列）を返す。
    ない場合は "None" を返す。
    """
    # 対象のカラム名リストを作成
    target_cols = [f'Pred_Class_{i}' for i in range(6, 15)]
    # 対象の部分を抽出
    subset = row[target_cols]
    # 閾値を超えている部分だけ抽出
    filtered = subset[subset >= threshold]
    if filtered.empty:
        return ""
    else:
        # 最も高い予測確率のラベルのカラム名を取得
        best_col = filtered.idxmax()
        # カラム名は "Pred_Class_数字" になっているので、数字部分のみを返す
        return best_col.split('_')[-1]

def process_csv(csv_file_path, threshold=0.5):
    """
    指定したCSVファイルを読み込み、対象の予測ラベル（Pred_Class_6～Pred_Class_14）について、
    閾値を超えているかつ最も予測確率の高いラベルを抽出し、1行に画像パスと抽出ラベルを出力する。

    パラメータ:
      csv_file_path (str): 読み込むCSVファイルのパス（拡張子付き）
      threshold (float): 閾値
    """
    # CSVファイルの読み込み
    csv_file_path = f'{csv_file_path}.csv'
    df = pd.read_csv(csv_file_path)

    # 各行について、対象ラベルの中で閾値を超えているかつ最も高い予測確率のラベルを抽出
    df['Extracted_Label'] = df.apply(lambda row: extract_best_label(row, threshold), axis=1)

    # 出力対象は画像パスと抽出したラベルのみとする（必要に応じてTrue_Classなどは含めなくてよい）
    result_df = df[['Image_Path', 'Extracted_Label']]

    # 結果をCSVに保存
    output_path = csv_file_path.replace('.csv', f'_{int(threshold*100)}pct_best_label.csv')
    result_df.to_csv(output_path, index=False)
    print(f"結果を {output_path} に保存しました。")


# ラベリングsplitの定義
LABELING_SPLIT = (
    # "20230801-125025-ES06_20230801-125615-es06-hd",
    # "20230802-095553-ES09_20230802-101030-es09-hd",
    "20210119093456_000002-001",
    # "20210524100043_000001-001",
    # "20211223090943_000001-003",
    # "20220322102354_000001-002",
)

# 使用例
if __name__ == "__main__":
    # 例として1ファイルのみ処理（複数ファイルの場合はループで回してください）
    # csv_file = "example.csv"  # 実際のCSVファイルパスに置き換えてください
    # process_csv(csv_file, threshold=0.9)
    
    # ラベリングsplitの全てを変換（例：threshold=0.9）
    for split_index in range(len(LABELING_SPLIT)):
        csv_file_path = f'/home/tanaka/labeling/15class_1/{LABELING_SPLIT[split_index]}/raw_results'
        process_csv(csv_file_path, threshold=0.8)
