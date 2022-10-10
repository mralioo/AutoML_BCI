"""
Created on 18.08.21
@author :ali
"""
import numpy as np
import optuna
import torch
from torch import nn as nn
from torchsummary import summary


class CNNModel_Test(nn.Module):
    def __init__(self, trial=None):
        super(CNNModel_Test, self).__init__()

        # Put here the parameters you want to tune with optuna
        if isinstance(trial, optuna.Trial):
            self.fc2_input_dim = trial.suggest_int("fc2_input_dim", 512, 1024)
            self.dropout_rate = trial.suggest_float("dropout_rate", 0, 1)
        else :
            self.fc2_input_dim = 512
            self.dropout_rate = 0.5

        # input shape L1= (?, 62, 6000) --> (batch_size, channels, timepoints)
        # Conv --> (?, 32, 6000)
        # Pool --> (?, 32, 3000)
        self.conv_layer1 = self._conv_layer_set(62, 32)

        # L2 = (?, 16, 3000)
        # Conv --> (?, 16, 3000)
        # Pool --> (?, 16, 1500)
        self.conv_layer2 = self._conv_layer_set(32, 16)

        # L3 = (?, 16, 1500)
        # Conv --> (?, 8, 1500)
        # Pool --> (?, 8, 750)
        self.conv_layer3 = self._conv_layer_set(16, 8)

        # L3 FC = 8*7*750 = 168000

        self.fc1 = nn.Linear(8 * 750, self.fc2_input_dim)
        self.fc2 = nn.Linear(self.fc2_input_dim, 1)

        # example to tune the dropout_rate using optune

        self.conv2_drop = nn.Dropout2d(p=self.dropout_rate)

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        return conv_layer

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv2_drop(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


if __name__ == "__main__":
    from torch.autograd import Variable

    input = Variable(torch.from_numpy(np.random.rand(1, 62, 6000))).double()
    model = CNNModel_Test()
    summary(model, (62, 6000), device="cpu")
    # model.double()
    # y_pred = model(input)
