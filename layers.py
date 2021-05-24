import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN_Encoder(nn.Module):
    """
    Encode an input sequence recurrently, considering word order.
    """
    def __init__(self, num_layers, hidden_size, drop_prob):
        super(RNN_Encoder, self).__init__()
        pass

    def forward(self, x):
        pass


class SentimentAnalysis(nn.Module):
    """
    Perform sentiment analysis on model. 
    """
    def __init__(self, num_layers, hidden_size, alpha, drop_prob):
        super(SentimentAnalysis, self).__init__()
        pass

    def forward(self, x):
        pass


class OutputLayer(nn.Module):
    """
    Two-stack of fully-connected output layers with 7 neurons. 
    To be used after sentiment analysis.
    """
    def __init__(self, input_size, hidden_size, alpha):
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(alpha),
            nn.Linear(hidden_size, 7),
            nn.Softmax()
        )

    def forward(self, x):
        return self.model(x)