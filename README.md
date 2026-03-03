# 内視鏡処置分類モデル

## 概要
内視鏡検査動画内で行われている処置の種類を分類するディープラーニングモデルを実装

## 特徴
- ResNet18をベースとしたCNNによる特徴抽出
- **マルチタスクモード**: BCEWithLogitsLossによるマルチラベル分類（main_head + unclear_head）
- **シングルラベルモード**: CrossEntropyLossによる単一クラス分類
- スライディングウィンドウによる予測の安定化
- 交差検証（CV）による学習・評価

## ディレクトリ構造
```
Treatment_CNN/
├── config/                        # 設定ファイル
│   ├── config_loader.py           # YAML→ExperimentConfig変換
│   ├── schema.py                  # Pydanticスキーマ定義
│   ├── train_config.yaml          # マルチタスク学習設定
│   ├── test_config.yaml           # マルチタスクテスト設定（旧スキーマ）
│   ├── train_single_label_config.yaml  # シングルラベル学習設定
│   └── test_single_label_config.yaml   # シングルラベルテスト設定
├── data/                          # データ関連モジュール
│   ├── data_splitter.py           # CVSplitterによる交差検証データ分割
│   ├── dataloader.py              # DataLoader生成
│   ├── datasets.py                # BaseMultiLabelDataset / CustomSingleLabelDataset
│   └── transforms.py             # 画像変換
├── engine/                        # 学習・推論エンジン
│   ├── runner.py                  # CVRunner（全フローのオーケストレーター）
│   ├── trainer.py                 # 学習ループ
│   ├── validator.py               # バリデーションループ
│   └── inference.py               # 推論処理
├── evaluate/                      # 評価指標
│   ├── analyzer.py                # 評価指標の集計
│   ├── converter.py               # 結果の形式変換
│   ├── exporter.py                # 結果の保存
│   ├── metrics.py                 # 各種評価指標の計算
│   ├── result_types.py            # 結果データクラス定義
│   └── visualizer.py             # 結果の可視化
├── model/                         # モデル定義
│   └── cnn_models.py              # SingleLabelClassificationModel / MultiTaskClassificationModel
├── utils/                         # ユーティリティ
│   ├── logger.py                  # ロギング
│   ├── torch_utils.py             # PyTorch関連ユーティリティ
│   ├── training_monitor.py        # 学習状況の監視
│   └── window_key.py              # スライディングウィンドウキー管理
├── train.py                       # 学習実行スクリプト
├── test.py                        # テスト実行スクリプト
└── requirements.txt               # 依存ライブラリ
```

## 必要環境
- Python 3.10以上
- PyTorch 2.0以上
- CUDA 11.7以上（GPU使用時）

主な依存ライブラリ：
```
torch
torchvision
numpy
pandas
scikit-learn
pillow
pyyaml
pydantic
tqdm
```

## セットアップ
```bash
# 仮想環境の作成
python -m venv env
source env/bin/activate

# 依存ライブラリのインストール
pip install -r requirements.txt
```

## 使用方法

### データセットの準備
動画内の画像（`.png`）が保存されたディレクトリと、対応するラベルファイル（`.csv`）が必要。
ラベルCSVファイルとディレクトリは同名にする必要がある。

```
dataset_root/
├── video_data1/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── video_data2/
│   ├── image1.png
│   └── ...
├── video_data1.csv
├── video_data2.csv
└── ...
```

CSVファイルのフォーマット：
```
image_name.png,label1,label2,...
```

### 設定ファイル

`model.type` によって学習モードを切り替える：

| `model.type` | 説明 |
|-------------|------|
| `multitask` | マルチタスク分類（BCEWithLogitsLoss） |
| `single_label` | シングルラベル分類（CrossEntropyLoss） |

設定ファイルのYAML例（マルチタスク学習）：

```yaml
mode: train
model:
  architecture: 'resnet18'
  type: 'multitask'
  num_classes: 6

dataset:
  root: '/path/to/dataset'
  img_size: 224
  batch_size: 16

paths:
  save_dir: 'train_output_dir'

training:
  pretrained: true
  freeze_backbone: false
  learning_rate: 1e-4
  max_epochs: 10

cv_ratio:              # 交差検証の分割比率
  train: 2
  val: 1
  test: 1

cv_splits:             # 各splitに含む動画名リスト
  split1:
    - video_data1
    - video_data2
  split2:
    - video_data3
  split3:
    - video_data4
  split4:
    - video_data5
```

`cv_ratio` と `cv_splits` の関係：
- `cv_ratio: train:2, val:1, test:1` の場合、4 splits を循環的に割り当てる
- `fold_idx=0` → split1,2→train / split3→val / split4→test

テスト設定のYAML例：

```yaml
mode: test
model:
  architecture: 'resnet18'
  type: 'multitask'
  num_classes: 6

dataset:
  root: '/path/to/dataset'
  img_size: 224
  batch_size: 1

paths:
  save_dir: 'test_output_dir'
  model_paths:
    - 'train_output_dir/fold_0/best_model.pth'
    - 'train_output_dir/fold_1/best_model.pth'
    - 'train_output_dir/fold_2/best_model.pth'
    - 'train_output_dir/fold_3/best_model.pth'

cv_ratio:
  train: 2
  val: 1
  test: 1

cv_splits:
  split1:
    - video_data1
  split2:
    - video_data2
  split3:
    - video_data3
  split4:
    - video_data4
```

### 学習

```bash
# マルチタスクモード（全fold）
python train.py -c config/train_config.yaml

# シングルラベルモード（全fold）
python train.py -c config/train_single_label_config.yaml

# 特定のfoldのみ実行
python train.py -c config/train_config.yaml -f 0
```

### テスト

```bash
# マルチタスクモード（全fold）
python test.py -c config/test_config.yaml

# シングルラベルモード（全fold）
python test.py -c config/test_single_label_config.yaml

# 特定のfoldのみ実行
python test.py -c config/test_config.yaml -f 0
```

## 評価指標
- Accuracy
- Precision
- Recall
- F1-score
- 混同行列

## 結果の出力
- `paths.save_dir` で指定したディレクトリに fold ごとのサブディレクトリを作成
- クラスごとの精度指標（CSV）
- 時系列での予測可視化（SVG形式）
- スライディングウィンドウごとの性能比較
