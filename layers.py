import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb 

class SequentialEncoder(nn.Module):
    """
    Encode an input sequence, considering word order.
    """
    def __init__(self, num_layers, hidden_size):
        super(SequentialEncoder, self).__init__()
        