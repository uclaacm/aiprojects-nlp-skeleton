from turtle import hideturtle
import torch
import torch.nn as nn


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression example. You may need to double check the dimensions :)
    """

    # TODO: change all input/output numbers to constants, etc. 
    def __init__(self, vocab_size, hidden1):
        super().__init__()
        self.fc1 = nn.Linear(vocab_size, hidden1) # What could that number mean!?!?!? Ask an officer to find out :)
        self.fc2 = nn.Linear(hidden1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        x (tensor): the input to the model
        '''
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


