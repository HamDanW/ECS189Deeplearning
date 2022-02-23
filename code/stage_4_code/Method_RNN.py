from code.base_class.method import method
import torch
from torch import nn as nn
import torch.nn.functional as F
import numpy as np


class Method_RNN(method, nn.Module):
    data = None

    max_epoch = 100
    learning_rate = .1
    '''
    def __init__(self, mName, mDescription):
        # TODO

    def forward(self, x):
        # TODO

    def train(self, X, y):
        # TODO

    def test(self, X):
        # TODO
    '''
