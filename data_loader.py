import os
import numpy as np
import pandas as pd
import torch

from config import classes_order


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        classes_filename = os.listdir(self.data_dir)
        classes_filename.sort()

        classes = []

        # sorting class_order
        for i, (k, v) in enumerate(classes_order.items()):
            for filename in classes_filename:
                if v in filename:
                    classes.append(filename)

        print(classes)
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
        gyro = sample.iloc[:, 4:7]

        accel = np.array(accel)

        gyro = np.array(gyro)

        accel = torch.tensor(accel, dtype=torch.float)
        gyro = torch.tensor(gyro, dtype=torch.float)

        label = self.lst_label[index]

        data = {'input': (accel, gyro), 'label': label}

        if self.transform:
            data = self.transform(data)

        return data
