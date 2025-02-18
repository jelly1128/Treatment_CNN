import csv
from pathlib import Path
from labeling.label_converter import HardMultiLabelResult
from cross_validation import CrossValidationSplitter


def load_hard_multilabel_results(csv_path: Path) -> HardMultiLabelResult:
    """
    CSVファイルからマルチラベルの結果を読み込みます。

    Args:
        csv_path (Path): 読み込むCSVファイルのパス

    Returns:
        HardMultiLabelResult: 読み込んだ結果
    """
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # ヘッダー行をスキップ
        
        # 列のインデックスを特定
        num_labels = (len(header) - 1) // 2  # Image_Pathを除いた列数の半分
        pred_indices = range(1, num_labels + 1)  # 予測ラベルのインデックス
        true_indices = range(num_labels + 1, 2 * num_labels + 1)  # 正解ラベルのインデックス
        
        result = HardMultiLabelResult(image_paths=[], multilabels=[], ground_truth_labels=[])
        
        for row in reader:
            if not row:  # 空行をスキップ
                continue
                
            image_path = Path(row[0])
            
            # データの追加
            pred_labels = [int(row[i]) for i in pred_indices]
            true_labels = [int(row[i]) for i in true_indices]
            
            result.image_paths.append(str(image_path))
            result.multilabels.append(pred_labels)
            result.ground_truth_labels.append(true_labels)
    
    return result

splits = {
    'split1': [
        "20210119093456_000001-001",
        "20210531112330_000005-001",
        "20211223090943_000001-002",
        "20230718-102254-ES06_20230718-102749-es06-hd",
        "20230802-104559-ES09_20230802-105630-es09-hd"
    ],
    'split2': [
        "20210119093456_000001-002",
        "20210629091641_000001-002",
        "20211223090943_000001-003",
        "20230801-125025-ES06_20230801-125615-es06-hd",
        "20230803-110626-ES06_20230803-111315-es06-hd"
    ],
    'split3': [
        "20210119093456_000002-001",
        "20210630102301_000001-002",
        "20220322102354_000001-002",
        "20230802-095553-ES09_20230802-101030-es09-hd",
        "20230803-093923-ES09_20230803-094927-es09-hd"
    ],
    'split4': [
        "20210524100043_000001-001",
        "20210531112330_000001-001",
        "20211021093634_000001-001",
        "20211021093634_000001-003"
    ]
}


# 使用例
if __name__ == "__main__":
    # 交差検証のデータ分割
    splitter = CrossValidationSplitter(splits=splits)
    split_folders = splitter.get_split_folders()
    
    num_classes = 6
    source_dir = Path(f'/home/tanaka/0218/Treatment_CNN/{num_classes}class_resnet18_test')
    
    fold_results = {}
    
    for fold_idx, fold in enumerate(split_folders):
        test_data_dirs = fold['test']
        hard_multilabels_results = {}
        
        for folder_name in test_data_dirs:
            csv_path = Path(f'{source_dir}/{num_classes}class_resnet18_test_fold{fold_idx+1}/{folder_name}/threshold_50%/threshold_50%_results_{folder_name}.csv')
            hard_multilabels_results[folder_name] = load_hard_multilabel_results(csv_path)
        
        fold_results[f'fold{fold_idx+1}'] = hard_multilabels_results
        
    print(len(fold_results.keys()))
    print(fold_results.keys())
    print(fold_results['fold1'].keys())
    print(len(fold_results['fold1']['20210524100043_000001-001'].multilabels))
    