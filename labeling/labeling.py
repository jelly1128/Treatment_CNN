import pandas as pd
import re

# ラベリングsplitの定義
LABELING_SPLIT = (
    "20230801-125025-ES06_20230801-125615-es06-hd",
    "20230802-095553-ES09_20230802-101030-es09-hd",
    "20211223090943_000001-002",
    "20210524100043_000001-001",
    "20211223090943_000001-003",
    "20220322102354_000001-002",
)

def convert_to_multi_label(csv_file_path, threshold=0.5):
    """
    one-hot形式のCSVファイルをマルチラベル形式に変換する関数

    Parameters:
        csv_file_path (str): CSVファイルのパス（拡張子なし）
        threshold (float): Pred_Classの閾値（デフォルトは0.5）

    Returns:
        None
    """
    # CSVファイルを読み込む
    df = pd.read_csv(f'{csv_file_path}.csv')

    # Pred_ClassとTrue_Classの列を抽出
    pred_columns = [col for col in df.columns if col.startswith('Pred_Class')]
    true_columns = [col for col in df.columns if col.startswith('True_Class')]

    # Pred_Classをマルチラベル形式に変換
    def get_multi_label_labels(row, columns, threshold):
        labels = []
        for idx, value in enumerate(row):
            if value >= threshold:
                labels.append(str(idx))  # クラス番号をラベルとして追加
        return ' '.join(labels) if labels else 'None'  # ラベルがない場合は 'None' を返す

    # True_Classをマルチラベル形式に変換
    def get_true_labels(row, columns):
        labels = []
        for idx, value in enumerate(row):
            if value == 1.0:  # True_Classは1.0の場合のみラベルとして追加
                labels.append(str(idx))
        return ' '.join(labels) if labels else 'None'  # ラベルがない場合は 'None' を返す

    # Pred_ClassとTrue_Classをマルチラベル形式に変換
    df['Pred_Class'] = df[pred_columns].apply(lambda row: get_multi_label_labels(row, pred_columns, threshold), axis=1)
    df['True_Class'] = df[true_columns].apply(lambda row: get_true_labels(row, true_columns), axis=1)

    # 必要な列だけを選択
    result_df = df[['Image_Path', 'Pred_Class', 'True_Class']]

    # 画像パスを数値順にソート
    def extract_number(image_path):
        # 画像パスから数値部分を抽出
        match = re.search(r'_(\d+)\.png$', image_path)
        if match:
            return int(match.group(1))
        return 0  # 数値がない場合は0を返す

    # 画像パスに基づいてソート
    result_df = result_df.copy()  # Copyを作成して警告を回避
    result_df['Sort_Key'] = result_df['Image_Path'].apply(extract_number)
    result_df = result_df.sort_values(by='Sort_Key').drop(columns=['Sort_Key'])

    # 新しいCSVファイルに保存
    result_df.to_csv(f'{csv_file_path}_{threshold*100}%_multi_label.csv', index=False)

# 使用例
if __name__ == "__main__":
    # # ラベリングsplitのインデックスを指定
    # split_index = 4
    # csv_file_path = f'/home/tanaka/labeling/15class_1/{LABELING_SPLIT[split_index]}/threshold_results'

    # 関数を呼び出して変換を実行
    # convert_to_multi_label(csv_file_path, threshold=0.5)
    
    # ラベリングsplitの全てを変換
    for split_index in range(len(LABELING_SPLIT)):
        csv_file_path = f'/home/tanaka/labeling/15class_2/{LABELING_SPLIT[split_index]}/threshold_results'
        convert_to_multi_label(csv_file_path, threshold=0.9)