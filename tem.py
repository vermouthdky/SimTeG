import os

import numpy as np

lms = ["all-MiniLM-L6-v2", "all-roberta-large-v1", "e5-large"]

dir = os.path.join("out", "ogbn-products", "SAGN_SCR", "main_X_")

for lm in lms:
    test_acc_list, valid_acc_list = [], []
    for i in range(10):
        log_dir = dir + lm + f"/log_{i}.txt"
        print(log_dir)
        with open(log_dir, "r") as f:
            lines = f.readlines()
            test_acc = float(lines[-1].split(" ")[-3][:-1])
            valid_acc = float(lines[-2].split(" ")[-3][:-1])
            test_acc_list.append(test_acc)
            valid_acc_list.append(valid_acc)
    test_accs, valid_accs = np.array(test_acc_list), np.array(valid_acc_list)
    print(f"test_acc: {test_accs.mean()} \(\pm\) {test_accs.std()}")
    print(f"valid_acc: {valid_accs.mean()} \(\pm\) {valid_accs.std()}")
