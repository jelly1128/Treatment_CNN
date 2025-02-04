import matplotlib.pyplot as plt
from collections import Counter
import torch

def plot_dataset_samples(save_dir, dataloader):
    # サンプルを表示し、1つの画像として保存
    num_samples_to_show = 10
    fig, axes = plt.subplots(1, num_samples_to_show, figsize=(20, 4))

    for i, (images, img_names, labels) in enumerate(dataloader):
        if i >= 1:  # 1バッチだけ処理
            break
        for j in range(num_samples_to_show):
            ax = axes[j]
            img = images[j].permute(1, 2, 0).numpy()  # CHW to HWC, tensor to numpy
            ax.imshow(img)
            ax.set_title(f"Label: {labels[j]}")
            ax.axis('off')
            print(f"Image path: {img_names[j]}, Label: {labels[j]}")

    plt.tight_layout()
    plt.savefig(f'{save_dir}/dataset_samples.png')
    
def show_dataset_stats(dataloader):
    # データセットの総数
    total_samples = len(dataloader.dataset)
    
    # ラベルの分布を計算
    all_labels = []
    for batch, (images, _, labels) in enumerate(dataloader):
        all_labels.extend(labels.cpu().tolist())
    
    # クラスごとのサンプル数をカウントするためのカウンターを初期化
    class_samples = Counter()

    # One-hotラベルを処理
    for one_hot in all_labels:
        # one_hotがリストであることを確認
        if isinstance(one_hot, list):
            one_hot = torch.tensor(one_hot)  # リストをテンソルに変換
        # 1の位置を見つけてカウントを更新
        for idx, value in enumerate(one_hot):
            if value == 1:
                class_samples[idx] += 1

    print(f"総サンプル数: {total_samples}")
    print("クラスごとのサンプル数:")
    for class_label, count in sorted(class_samples.items()):
        print(f"クラス {class_label}: {count}")