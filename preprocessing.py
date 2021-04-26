import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.tranasforms import ToTensor, Lambda
import matplotlib.pyplot as plt 

training_data = None

test_data = None