import numpy as np
import torch
import torch.optim as optim

from model import GestureClassifier
from data_loader import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

losses = []
PATH = 'model.pth'


def train(model, train_loader, optimizer):
    model.train()
    train_loss = 0
    for batch, data in enumerate(train_loader, 1):
        (accel, gyro) = data['input']
        label = data['label']
        label = label.to(device)

        out = model(accel.to(device), gyro.to(device))

        optimizer.zero_grad()
        loss = F.cross_entropy(out, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader.dataset)

    return train_loss


def evaluate(model, test_loader):
    model.eval()

    correct = 0

    with torch.no_grad():
        for batch, data in enumerate(test_loader, 1):
            (accel, gyro) = data['input']
            label = data['label']
            label = label.to(device)
            out = model(accel.to(device), gyro.to(device))

            pred = out.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_accuracy


def main():
    model = GestureClassifier(15 * 3).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    dataset_train = Dataset(data_dir='data/train', transform=None)
    loader_train = DataLoader(dataset_train, batch_size=1000, shuffle=False, num_workers=0)

    dataset_test = Dataset(data_dir='data/val', transform=None)
    loader_test = DataLoader(dataset_test, batch_size=1000, shuffle=False, num_workers=0)

    accuracies = []

    print('학습 시작')
    for i in range(300):
        loss = train(model, loader_train, optimizer)
        losses.append(loss)
        accuracy = evaluate(model, loader_test)
        accuracies.append(accuracy)

        print('epoch : ', i)

    torch.save(model, PATH)

    plt.title('loss')
    plt.plot(range(len(losses)), losses)
    plt.show()

    plt.title('accuracy')
    plt.plot(range(len(accuracies)), accuracies)
    plt.show()


if __name__ == "__main__":
    main()
