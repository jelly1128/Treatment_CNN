import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from collections import Counter
import torch
from torchvision.utils import make_grid

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
        
        
def visualize_dataset(dataset, output_dir, num_samples=500):
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(0, len(dataset), 100):
        images_list = []
        labels_list = []
        
        for j in range(i, min(i + 100, len(dataset))):
            images, _, label = dataset[j]
            last_image = images[-1]
            images_list.append(last_image)
            labels_list.append(label)
        
        grid = make_grid(torch.stack(images_list), nrow=10)
        
        # 画像をPIL Imageに変換
        pil_image = Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype('uint8'))
        
        # 画像にラベルを追加
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.load_default()
        label_text = ", ".join(map(str, set(labels_list)))
        draw.text((10, 10), f"Labels: {label_text}", fill=(255, 255, 255), font=font)
        
        # 画像を保存
        output_path = os.path.join(output_dir, f"sample_{i//100}.png")
        pil_image.save(output_path)
        
        print(f"Saved image with labels {label_text} to {output_path}")
        

def visualize_multilabel_timeline(df, save_dir, filename, num_classes):
    # Define the colors for each class
    label_colors = {
        0: (254, 195, 195),       # white
        1: (204, 66, 38),         # lugol
        2: (57, 103, 177),        # indigo
        3: (96, 165, 53),         # nbi
        4: (86, 65, 72),          # outside
        5: (159, 190, 183),       # bucket
    }

    # Default color for labels not specified in label_colors
    default_color = (148, 148, 148)

    # Extract the predicted labels columns
    predicted_labels = df[[col for col in df.columns if 'Predicted' in col]].values

    # Determine the number of images
    n_images = len(predicted_labels)
    
    # Set timeline height based on the number of labels
    timeline_width = n_images
    timeline_height = num_classes * (n_images // 10)

    # Create a blank image for the timeline
    timeline_image = Image.new('RGB', (timeline_width, timeline_height), (255, 255, 255))
    draw = ImageDraw.Draw(timeline_image)

    # Iterate over each image (row in the CSV)
    for i in range(n_images):
        # Get the predicted labels for the current image
        labels = predicted_labels[i]
        
        # Check each label and draw corresponding rectangles
        for label_idx, label_value in enumerate(labels):
            if label_value == 1:
                row_idx = label_idx

                # Calculate the position in the timeline
                x1 = i * (timeline_width // n_images)
                x2 = (i + 1) * (timeline_width // n_images)
                y1 = row_idx * (n_images // 10)
                y2 = (row_idx + 1) * (n_images // 10)
                
                # Get the color for the current label
                color = label_colors.get(label_idx, default_color)
                
                # Draw the rectangle for the label
                draw.rectangle([x1, y1, x2, y2], fill=color)
                
    # Save the image
    timeline_image.save(os.path.join(save_dir, f'{filename}_multilabel_timeline.png'))
    print(f'Timeline image saved at {os.path.join(save_dir, "multilabel_timeline.png")}')