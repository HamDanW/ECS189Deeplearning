from code.base_class.method import method
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_3_code.Evaluate_Precision import Evaluate_Precision
from code.stage_3_code.Evaluate_Recall import Evaluate_Recall
from code.stage_3_code.Evaluate_F1 import Evaluate_F1


class Method_CNN(method, nn.Module):
    # Channels an image has. 1 for gray images (MNIST). 3 for RGB (CIFAR). ORL uses 3?
    # Need to manually set
    channels = 3
    data = None
    # it defines the max rounds to train the model
    max_epoch = 100
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    batch_size = 10000

    # CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    # From Pytorch Example
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # CIFAR layers 32x32x3, 10 classes
        self.conv1 = nn.Conv2d(self.channels, 16, 3, padding=1).to(self.device)
        self.pool = nn.MaxPool2d(2, 2).to(self.device)
        self.conv2 = nn.Conv2d(16, 32, 3).to(self.device)
        self.conv3 = nn.Conv2d(32, 64, 3).to(self.device)
        # self.conv4 = nn.Conv2d(64, 128, 3).to(self.device)
        self.fc1 = nn.Linear(256, 512).to(self.device)
        self.fc2 = nn.Linear(512, 64).to(self.device)
        self.fc3 = nn.Linear(64, 10).to(self.device)
        self.Dropout = nn.Dropout(0.5)
        # self.fc4 = nn.Linear(30, 10).to(self.device)
        self.softmax = nn.Softmax(dim=1)

        # ORL layers 112x92x3, 40 classes
        '''
        self.conv1 = nn.Conv2d(self.channels, 16, (5, 5), padding=2).to(self.device)
        self.pool = nn.MaxPool2d(2, 2).to(self.device)
        self.conv2 = nn.Conv2d(16, 32, (3, 3)).to(self.device)
        self.conv3 = nn.Conv2d(32, 64, (3, 3)).to(self.device)
        self.conv4 = nn.Conv2d(64, 128, (3, 3)).to(self.device)
        self.fc1 = nn.Linear(420, 360).to(self.device)
        self.fc2 = nn.Linear(360, 200).to(self.device)
        self.fc3 = nn.Linear(200, 100).to(self.device)
        self.fc4 = nn.Linear(100, 40).to(self.device)
        '''

        # MNIST layers 28x28x1, 10 classes
        '''
        self.conv1 = nn.Conv2d(self.channels, 7, 5, padding=2).to(self.device)
        self.pool = nn.MaxPool2d(2, 2).to(self.device)
        self.conv2 = nn.Conv2d(7, 14, 5).to(self.device)
        self.fc1 = nn.Linear(504, 300).to(self.device)
        self.fc2 = nn.Linear(300, 200).to(self.device)
        self.fc3 = nn.Linear(200, 100).to(self.device)
        self.fc4 = nn.Linear(100, 10).to(self.device)
        '''

    # From Pytorch Example
    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x))).to(self.device)
        x = self.pool(F.relu(self.conv2(x))).to(self.device)
        x = self.pool(F.relu(self.conv3(x))).to(self.device)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.Dropout(F.relu(self.fc1(x)).to(self.device))
        x = self.Dropout(F.relu(self.fc2(x)).to(self.device))
        x = self.fc3(x).to(self.device)
        # activation4 = self.fc4(activation3).to(self.device)
        # y_pred = self.softmax(activation3)
        return x

        # ORL Dataset
        '''
        # x = x.unsqueeze(0)
        # x = torch.reshape(x, (self.channels, 1, 2, 3))
        # print(np.shape(x))
        conv1 = self.pool(F.relu(self.conv1(x))).t  o(self.device)
        conv2 = self.pool(F.relu(self.conv2(conv1))).to(self.device)
        conv3 = self.pool(F.relu(self.conv3(conv2))).to(self.device)
        conv4 = self.pool(F.relu(self.conv4(conv3))).to(self.device)
        flat = torch.flatten(conv4, -3).to(self.device)  # flatten all dimensions except batch
        activation1 = F.relu(self.fc1(flat)).to(self.device)
        activation2 = F.relu(self.fc2(activation1)).to(self.device)
        activation3 = self.fc3(activation2).to(self.device)
        activation4 = self.fc4(activation3).to(self.device)
        # y_pred = self.softmax(activation3)
        y_pred = activation4
        '''

        # return y_pred

    def train(self, X, y):
        # X has form: [[image1][image2]...[image n]]
        # y has form: [label1, label2, ..., label n]

        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        loss_function = nn.CrossEntropyLoss()

        # Transform X to have form [Channel, Height, Width]
        # transformed = torch.permute(torch.FloatTensor(np.array(X)), (2,0,1))
        for epoch in range(self.max_epoch + 1):  # you can do an early stop if self.max_epoch is too much...
            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            # optimizer.zero_grad()
            # Convert X and y to tensors

            # CIFAR
            tensorX = torch.FloatTensor(np.array(X)).to(self.device)

            # MNIST
            # tensorX = torch.FloatTensor(np.array(X)).unsqueeze(3).to(self.device)

            tensorY = torch.LongTensor(np.array(y)).to(self.device)

            permutation = torch.randperm(tensorX.size()[0]).to(self.device)
            for i in range(0, tensorX.size()[0], self.batch_size):
                indices = permutation[i:i + self.batch_size - 1]
                miniX, miniy = tensorX[indices], tensorY[indices]

                # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
                optimizer.zero_grad()

                # Non CUDA version
                temp = torch.permute(miniX, (0, 3, 1, 2)).to(self.device)
                y_pred = self.forward(temp).to(self.device)
                # convert y to torch.tensor as well
                y_true = miniy

                # fix tensor dimensions for ORL
                # y_pred = y_pred - 1
                # y_true = y_true - 1

                # calculate the training loss
                train_loss = loss_function(y_pred, y_true).to(self.device)

                # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
                # do the error backpropagation to calculate the gradients

                train_loss.backward()
                # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()

            # Create evaluation objects that represent evaluation metrics
            accuracy_evaluator = Evaluate_Accuracy('accuracy training evaluator', '')

            if epoch % 10 == 0:
                # accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                accuracy_evaluator.data = {'true_y': y_true.to('cpu'), 'pred_y': y_pred.to('cpu').max(1)[1]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

    def test(self, X):
        # do the testing, and result the result
        temp = torch.permute(torch.FloatTensor(np.array(X)), (0, 3, 1, 2)).to(self.device)
        y_pred = self.forward(temp).to(self.device)
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        # data has form:
        # {'train': {'X': trainX, 'y': trainy}, 'test': {'X': testX, 'y': testy}}
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
