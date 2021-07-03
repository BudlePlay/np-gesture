import os
import numpy as np
import pandas as pd
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        classes = os.listdir(self.data_dir)
        classes.sort()

        lst_input = []
        lst_label = []

        for i, c in enumerate(classes):
            a = os.listdir(os.path.join(data_dir, c))
            print(a)
            for j in a:
                df = pd.read_csv(os.path.join(data_dir, c, j))
                lst_input.append(df)
                lst_label.append(i)

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):

        sample = self.lst_input[index]
        accel = sample.iloc[:, 1:4]
        gyro = sample.iloc[:, 1:4]

        accel = np.array(accel).reshape(-1)

        gyro = np.array(gyro).reshape(-1)

        accel = torch.tensor(accel, dtype=torch.float)
        gyro = torch.tensor(gyro, dtype=torch.float)

        label = self.lst_label[index]

        data = {'input': (accel, gyro), 'label': label}

        if self.transform:
            data = self.transform(data)

        return data
