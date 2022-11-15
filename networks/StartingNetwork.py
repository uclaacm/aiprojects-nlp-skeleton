import torch
import torch.nn as nn


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression example. You may need to double check the dimensions :)
    """

    # TODO: change all input/output numbers to constants, etc. 
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(int(config["INPUT_DIMENSION"]), int(config["hidden1"]))
        self.fc2 = nn.Linear(int(config["hidden1"]), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        x (tensor): the input to the model
        '''
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


