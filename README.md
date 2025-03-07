# 内視鏡処置分類モデル

## 概要
内視鏡検査動画内で行われている処置の種類を分類するディープラーニングモデルを実装

## 特徴
- ResNet18をベースとしたCNNによる特徴抽出
- モデルの出力は各クラスの確率（ソフトラベル）
- スライディングウィンドウによる予測の安定化
- 詳細な評価指標の算出と可視化

## ディレクトリ構造
```
Treatment_CNN/
├── analyze/                  # 解析関連のモジュール
│   ├── window_key.py         # スライディングウィンドウのキー管理
│   └── analyzer.py           # 結果解析の実装
├── config/                   # 設定ファイル
│   ├── config_loader.py      # 設定ファイル読み込み
│   ├── schema.py             # 設定のスキーマ定義
│   ├── train_config.yaml     # 学習用設定
│   └── test_config.yaml      # テスト用設定
├── data/                     # データ関連のモジュール
│   ├── data_loader.py        # データローダの実装
│   ├── data_splitter.py      # データ分割の実装
│   ├── dataset_visualizer.py # データセットの可視化
│   ├── datasets.py           # データセットの定義
│   └── transforms.py         # データセットの変換
├── engine/                   # 学習・推論エンジン
│   ├── trainer.py            # 学習ループの実装
│   ├── validator.py          # バリデーションループの実装
│   └── inference.py          # 推論処理の実装
├── evaluate/                 # 評価指標の計算
│   ├── analyzer.py           # 評価指標の計算
│   ├── metrics.py            # 評価指標の計算
│   ├── model_evaluator.py    # モデルの評価
│   ├── results_visualizer.py # 結果の可視化
│   └── save_metrics.py       # 評価結果の保存
├── labeling/                 # ラベリング関連の処理
│   └── label_converter.py    # ラベル形式の変換
├── model/                    # モデル定義
│   ├── cnn_models.py         # CNNモデルの定義
│   └── setup_models.py       # モデルのセットアップ
├── utils/                    # ユーティリティ関数
│   ├── logger.py             # ロギング
│   ├── torch_utils.py.py     # PyTorch関連のユーティリティ
│   └── training_monitor.py   # 学習状況の監視
├── train.py                  # 学習実行スクリプト
├── test.py                   # テスト実行スクリプト
├── requirements.txt          # 依存ライブラリ
└── README.md                 # このファイル
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
```

## セットアップ
```bash
# 仮想環境の作成
python -m venv venv
source venv/bin/activate

# 依存ライブラリのインストール
pip install -r requirements.txt
```

## 使用方法

### データセットの準備
データセットは`data/`ディレクトリに配置
動画内の画像(.png)が保存されたディレクトリと，対応するラベルファイル(.csv)が必要．
ラベルが保存されたcsvファイルと画像が保存されたディレクトリは同名である必要がある．
```
data/
├── video_data1/
│   ├── image1.png
│   ├── image2.png
│   ├── ...
├── video_data2/
│   ├── image1.png
│   ├── image2.png
│   ├── ...
├── ...
├── video_data1.csv
├── video_data2.csv
├── ...
```

### 学習
```bash
python train.py -c config/train_config.yaml
```

### テスト
```bash
python test.py -c config/test_config.yaml
```

## 設定ファイル
`config/train_config.yaml`で学習時の設定を指定できます：

```yaml
training:
  img_size: 224                    # 画像サイズ
  num_classes: 6                   # クラス数
  model_architecture: 'resnet18'   # モデルアーキテクチャ
  model_type: 'multitask'          # モデルタイプ
  pretrained: True                 # 事前学習済みモデルの使用
  freeze_backbone: False           # バックボーンのフリーズ
  learning_rate: 1e-4              # 学習率
  batch_size: 16                   # バッチサイズ
  max_epochs: 10                   # 最大エポック数

paths:
  dataset_root: '../data'  # データセットのルートパス
  save_dir: '../results'           # 保存ディレクトリ

splits:                            # 交差検証用のデータ分割
  split1:
    - video_data1
    - video_data2
    - ...
  split2:
    - video_data3
    - video_data4
    - ...
```

## 評価指標
- Accuracy
- Precision
- Recall
- F1-score
- 混同行列

## 結果の出力
- クラスごとの精度指標
- 時系列での予測可視化（SVG形式）
- スライディングウィンドウごとの性能比較

## 注意事項
- データセットは含まれていません
- 