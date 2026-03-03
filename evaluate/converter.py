from evaluate.result_types import InferenceResult, HardMultiLabelResult

"""
修正したい箇所
マルチラベルから単一ラベルへの変換ロジック（特に全てのラベルが0の場合の処理）。

"""


class MultiToSingleLabelConverter:
    def __init__(self, inference_results_dict: dict[str, InferenceResult], num_classes: int = 6):
        self.inference_results_dict = inference_results_dict
        self.num_classes = num_classes

    def convert_soft_to_hard_multi_labels(self, threshold: float = 0.5) -> dict[str, HardMultiLabelResult]:
        """
        確率のしきい値を使用して、マルチソフトラベルをマルチラベルに変換します。
        全てのラベルが閾値を超えていない場合、主ラベルの中で最も確率が高いものとその確率より高いサブラベルを選択します。

        Args:
            threshold (float): ラベルの割り当てを決定する確率のしきい値。

        Returns:
            dict[str, HardMultiLabelResult]: しきい値を超えるラベルのリストを含む辞書。
        """
        hard_multi_labels_results = {}
        # フォルダごとにマルチソフトラベルをマルチラベルに変換
        for video_name, inference_result in self.inference_results_dict.items():
            hard_multi_label_result = HardMultiLabelResult(image_paths=[], multi_labels=[], ground_truth_labels=[])

            for image_path, probabilities, labels in zip(inference_result.image_paths, inference_result.probabilities, inference_result.labels):
                # 通常の閾値処理
                hard_multi_label = [1 if prob > threshold else 0 for prob in probabilities]

                # 全てのラベルが0の場合の処理
                if sum(hard_multi_label) == 0:
                    # 主ラベル（0-5）の中で最も確率が高いものを見つける
                    main_probs = probabilities[:self.num_classes]  # 主ラベルの確率
                    max_main_prob = max(main_probs)
                    max_main_idx = main_probs.index(max_main_prob)

                    # 主ラベルの最大確率より高い確率を持つサブラベルを見つける
                    hard_multi_label = [0] * len(probabilities)
                    hard_multi_label[max_main_idx] = 1  # 最も確率の高い主ラベルを1に設定

                    # サブラベル（num_classes以降）で主ラベルの最大確率より高いものを1に設定
                    for i, prob in enumerate(probabilities[self.num_classes:], start=self.num_classes):
                        if prob > max_main_prob:
                            hard_multi_label[i] = 1

                hard_multi_label_result.image_paths.append(image_path)
                hard_multi_label_result.multi_labels.append(hard_multi_label)
                hard_multi_label_result.ground_truth_labels.append(labels)

            hard_multi_labels_results[video_name] = hard_multi_label_result

        return hard_multi_labels_results
