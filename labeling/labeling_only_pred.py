import pandas as pd
import re

# ラベリングスプリットの定義
LABELING_SPLIT = (
    "20230801-125025-ES06_20230801-125615-es06-hd",
    "20230802-095553-ES09_20230802-101030-es09-hd",
    "20211223090943_000001-002",
    "20210524100043_000001-001",
    "20211223090943_000001-003",
    "20220322102354_000001-002",
)

def extract_number(image_path):
    """
    画像パスから数値部分を抽出する関数
    """
    match = re.search(r'_(\d+)\.png$', image_path)
    if match:
        return int(match.group(1))
    return 0  # 数値がない場合は0を返す

def convert_to_multilabel(csv_file_path, threshold=0.5):
    """
    one-hot形式のCSVファイルから予測ラベルのみを抽出し、
    各画像ごとに1行で、画像パスとそれぞれの予測ラベルを別セルに入れる形式に変換して保存する関数

    Parameters:
        csv_file_path (str): CSVファイルのパス（拡張子なし）
        threshold (float): Pred_Classの閾値（デフォルトは0.5）

    Returns:
        None
    """
    # CSVファイルを読み込む
    df = pd.read_csv(f'{csv_file_path}.csv')

    # Pred_Classの列を抽出
    pred_columns = [col for col in df.columns if col.startswith('Pred_Class')]

    # 各行ごとに、threshold以上のクラス番号（文字列）をリストとして抽出
    def get_pred_labels(row, threshold):
        labels = []
        for idx, value in enumerate(row):
            if value >= threshold:
                labels.append(str(idx))  # クラス番号を文字列として追加
        return labels if labels else ['None']  # 該当ラベルがなければ ['None'] を返す

    # 各行に対して予測ラベルのリストを作成する
    df['pred_labels'] = df[pred_columns].apply(lambda row: get_pred_labels(row, threshold), axis=1)

    # リストからDataFrameに変換（各要素を1セルに配置）
    # 画像ごとに予測ラベルのリストを展開し、列として配置する
    labels_df = pd.DataFrame(df['pred_labels'].tolist())
    # 列名を "Pred_Class_1", "Pred_Class_2", ... と設定
    labels_df.columns = [f'Pred_Class_{i+1}' for i in range(labels_df.shape[1])]

    # 画像パスと予測ラベルの列を結合
    result_df = pd.concat([df['Image_Path'], labels_df], axis=1)

    # 画像パスから抽出した数値でソート
    result_df['Sort_Key'] = result_df['Image_Path'].apply(extract_number)
    result_df = result_df.sort_values(by='Sort_Key').drop(columns=['Sort_Key'])

    # 新しいCSVファイルに保存
    result_df.to_csv(f'{csv_file_path}_{int(threshold*100)}%_multilabel.csv', index=False)

# 使用例
if __name__ == "__main__":
    # ラベリングスプリットの全てを変換（例：threshold=0.9）
    for split_index in range(len(LABELING_SPLIT)):
        csv_file_path = f'/home/tanaka/labeling/15class_1/{LABELING_SPLIT[split_index]}/raw_results'
        convert_to_multilabel(csv_file_path, threshold=0.8)
