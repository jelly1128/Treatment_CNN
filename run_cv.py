# anomaly_trainとtreatment_trainを実行

import os
import subprocess

# subprocess.run(["python3", "train.py", "-c", "config/train_config_6class.yaml"])
# subprocess.run(["python3", "train.py", "-c", "config/train_config_7class.yaml"])
# subprocess.run(["python3", "train.py", "-c", "config/train_config_15class.yaml"])


subprocess.run(["python3", "test.py", "-c", "config/test_config_6class.yaml"])
subprocess.run(["python3", "test.py", "-c", "config/test_config_7class.yaml"])
subprocess.run(["python3", "test.py", "-c", "config/test_config_15class.yaml"])