
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.pooling import MaxPool2d


class Net2(torch.nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3,32,3,1,1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(32 * 3 *3,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        conv_out = self.conv1(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense(res)
        return out

   
print("Method 2:")
model2 = Net2()
print(model2)