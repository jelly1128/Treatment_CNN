# anomaly_trainとtreatment_trainを実行

import os
import subprocess

# マルチラベル
# subprocess.run(["python3", "train.py", "-c", "config/train_config_6class.yaml"])
# subprocess.run(["python3", "train.py", "-c", "config/train_config_7class.yaml"])
# subprocess.run(["python3", "train.py", "-c", "config/train_config_15class.yaml"])

# subprocess.run(["python3", "test.py", "-c", "config/test_config_6class.yaml"])
# subprocess.run(["python3", "test.py", "-c", "config/test_config_7class.yaml"])
# subprocess.run(["python3", "test.py", "-c", "config/test_config_15class.yaml"])

# シングルラベル
subprocess.run(["python3", "train.py", "-c", "config/single_label_train_config_5class.yaml"])
subprocess.run(["python3", "test_single_label.py", "-c", "config/single_label_test_config_5class.yaml"])
