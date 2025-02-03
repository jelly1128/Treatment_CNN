import pandas as pd
import numpy as np
import os
from PIL import Image, ImageDraw

# タイムラインの可視化関数
def visualize_multilabel_timeline(df, save_dir, filename, num_classes):
    """
    マルチラベルタイムラインを可視化して保存
    """
    # Define the colors for each class
    label_colors = {
        0: (254, 195, 195),       # white
        1: (204, 66, 38),         # lugol
        2: (57, 103, 177),        # indigo
        3: (96, 165, 53),         # nbi
        4: (86, 65, 72),          # custom color for label 4
        5: (159, 190, 183),       # custom color for label 5
    }

    # Default color for labels not specified in label_colors
    default_color = (148, 148, 148)

    # Extract the predicted labels columns
    predicted_labels = df[[col for col in df.columns if 'Pred_Class' in col]].values

    # Determine the number of images
    n_images = len(predicted_labels)
    
    # Set timeline height based on the number of labels
    timeline_width = n_images
    # timeline_height = 2 * (n_images // 10)  # 20 pixels per label row (2 rows total)
    timeline_height = (n_images // 10)  # 20 pixels per label row (1 rows total)

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
                # row_idx = label_idx
                row_idx = 0 if label_idx < 6 else 1
                # if label_idx < 6:
                    # row_idx = 0

                # Calculate the position in the timeline
                # x1 = i
                # x2 = i + 1
                # y1 = row_idx * 20
                # y2 = (row_idx + 1) * 20
                
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
    os.makedirs(save_dir, exist_ok=True)
    timeline_image.save(os.path.join(save_dir, f'{filename}_multilabel_timeline.png'))
    print(f'Timeline image saved at {os.path.join(save_dir, f"{filename}_multilabel_timeline.png")}')

# スムージングおよび可視化プロセス
def process_and_visualize(input_dir, output_dir, class_num=6):
    """
    各手法でスムージングした後、タイムラインを可視化
    """
    # data = pd.read_csv(input_csv)
    methods = ["majority_vote", "continuity", "longest_segment"]
    
    for method in methods:
        # スムージング後のデータを読み込む
        data = pd.read_csv(f"{input_dir}/{method}_predictions.csv")
        # print(data.shape)
        smoothed_data = data[[col for col in data.columns if method in col]]
        smoothed_data.columns = [f"{method}_Pred_Class_{i}" for i in range(class_num)]

        # タイムラインの可視化
        visualize_multilabel_timeline(
            df=smoothed_data,
            save_dir=output_dir,
            filename=method,
            num_classes=class_num
        )
        

# 使用例
# input_csv = "debug/majority_vote_predictions.csv"  # 入力ファイル例
# output_dir = "visualizations"  # 出力先ディレクトリ
# NUM_CLASSES = 6  # クラス数
# process_and_visualize(input_csv, output_dir)


def main():
    # folder = "debug"
    # class_num = 6
    # output_dir = "debug"
    
    folder = "results"
    class_num = 6
    output_dir = "visualizations"
    
    for fold in range(4):
        subfolder_names = [name for name in os.listdir(os.path.join(folder, f"{class_num}class_fold{fold + 1}")) if os.path.isdir(os.path.join(folder, f"{class_num}class_fold{fold + 1}", name))]
        print(subfolder_names)
        # 各テストフォルダ毎の結果の保存folderを作成
        if not os.path.exists(os.path.join(output_dir, f"{class_num}class_fold{fold + 1}")):
            os.mkdir(os.path.join(output_dir, f"{class_num}class_fold{fold + 1}"))
            
        for subfolder_name in subfolder_names:
            # 各テストフォルダ毎の結果の保存folderを作成
            if not os.path.exists(os.path.join(output_dir, f"{class_num}class_fold{fold + 1}", subfolder_name)):
                os.mkdir(os.path.join(output_dir, f"{class_num}class_fold{fold + 1}", subfolder_name))
                
            if not os.path.exists(os.path.join(output_dir, f"{class_num}class_fold{fold + 1}", subfolder_name, "images")):
                os.mkdir(os.path.join(output_dir, f"{class_num}class_fold{fold + 1}", subfolder_name, "images"))
                
            # process_and_evaluate(os.path.join(folder, f"{class_num}class_fold{fold + 1}", subfolder_name, "multilabels_test_with_labels.csv")    # input_dir
            #                      , os.path.join(output_dir, f"{class_num}class_fold{fold + 1}", subfolder_name))    # output_dir
            
            process_and_visualize(os.path.join(folder, f"{class_num}class_fold{fold + 1}", subfolder_name)    # input_dir
                                 , os.path.join(output_dir, f"{class_num}class_fold{fold + 1}", subfolder_name, "images")    # output_dir
                                 , class_num)
            
            # タイムラインの可視化
            # visualize_multilabel_timeline(
            #     df=data,
            #     save_dir=output_dir,
            #     filename="predicted",
            #     num_classes=class_num
            # )
            
            # visualize_multilabel_timeline(
            #     df=smoothed_data,
            #     save_dir=output_dir,
            #     filename=method,
            #     num_classes=class_num
            # )
            

if __name__ == '__main__':
    main()