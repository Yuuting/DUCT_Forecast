# -*- coding: utf-8 -*-
# @Time    : 2021/7/16 12:36
# @Author  : yuting
# @FileName: NN.py

from torch import nn
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.hidden1 = nn.Sequential(
            nn.Linear(in_features=12, out_features=100, bias=True),
            nn.ReLU())
        self.hidden2 = nn.Sequential(
            nn.Linear(in_features=100, out_features=100, bias=True),
            nn.ReLU())
        self.hidden3 = nn.Sequential(
            nn.Linear(in_features=100, out_features=50, bias=True),
            nn.ReLU())
        self.predict = nn.Sequential(
            nn.Linear(in_features=50, out_features=13, bias=True),
            nn.ReLU())

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.predict(x)
        return x
