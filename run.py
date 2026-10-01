import subprocess
import sys

lens = [100, 150, 200, 300]
net_types = ['CNN', 'CRNN', 'L_CNN']
periods = [1, 3, 5, 10, 60]
houses = [1, 2, 3]

for len in lens:
    for net in net_types:
        for test_house in houses:
            train_houses = [h for h in houses if h != test_house]
            cmd = [
                sys.executable, "main.py",
                "--seed", "42",
                "--experiment_name", "Test_sh",
                "--model_name", str(net),
                "--window_len", str(len),
                "--house_indicies_train", *map(str, train_houses),
                "--house_indicies_test", str(test_house), 
            ]
            subprocess.run(cmd, check=True)