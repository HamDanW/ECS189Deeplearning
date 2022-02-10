from code.base_class.method import method

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


class Method_CNN(method, nn.Module):
    data = None
    learning_rate = 1e-5
    max_epoch = 10

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # update for different dataset
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(self, X, y):
        # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)

        for epoch in range (self.max_epoch):
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            y_true = torch.LongTensor(np.array(y))

            train_loss = loss_function(y_pred, y_true)

            optimizer.zero_grad()
            train_loss.backward()

            optimizer.step()

        print('Finished Training')

    def test(self, X):
        print('Finished Testing')

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])