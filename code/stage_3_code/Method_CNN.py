from code.base_class.method import method
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy

class Method_CNN(method, nn.Module):
    #Channels an image has. 1 for gray images. 3 for RGB.
    #Need to manually set
    channels = 3
    data = None
    # it defines the max rounds to train the model
    max_epoch = 10
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-5

    #From Pytorch Example
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(self.channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        #self.softmax = nn.Softmax(dim=1)

    #From Pytorch Example
    def forward(self,x):
        #x = x.unsqueeze(0)
        # x = torch.reshape(x, (self.batch_size,x.size(dim=1), x.size(dim=2), x.size(dim=3)))
        conv1 = self.pool(F.relu(self.conv1(x)))
        conv2 = self.pool(F.relu(self.conv2(conv1)))
        flat = torch.flatten(conv2, 1) # flatten all dimensions except batch
        activation1 = F.relu(self.fc1(flat))
        activation2 = F.relu(self.fc2(activation1))
        activation3 = self.fc3(activation2)
        #y_pred = self.softmax(activation3)
        y_pred = activation3
        return y_pred

    def train(self, X, y):
        # X has form: [[image1][image2]...[image n]]
        # y has form: [label1, label2, ..., label n]

        #Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()

        #Transform X to have form [Channel, Height, Width]
        #transformed = torch.permute(torch.FloatTensor(np.array(X)), (2,0,1))

        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()

            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it

            # Non CUDA version
            temp = torch.permute(torch.FloatTensor(np.array(X)), (0, 3, 1, 2))
            y_pred = self.forward(temp)
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))

            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # Create evaluation objects that represent evaluation metrics
            accuracy_evaluator = Evaluate_Accuracy('accuracy training evaluator', '')

            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            optimizer.zero_grad()
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            if epoch % 100 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())


    def test(self, X):
        # do the testing, and result the result
        temp = torch.permute(torch.FloatTensor(np.array(X)), (0, 3, 1, 2))
        y_pred = self.forward(temp)
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        #data has form:
        # {'train': {'X': trainX, 'y': trainy}, 'test': {'X': testX, 'y': testy}}
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
