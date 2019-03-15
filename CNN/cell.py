# put normal and reduction cell here
import torch
from torch import nn
from ops import *


class NormalCell(nn.Module):
    def __init__(self, cell_channels, save_device, training, drop_prob):
        super(NormalCell, self).__init__()
        self.save_device = save_device
        self.training = training
        self.drop_prob = drop_prob

    def forward(self, x_in0, x_in1):
        # x_in0: output of h-1 cell
        # x_in1: output of h-2 cell
        pass


class ReductCell(nn.Module):
    def __init__(self, cell_channels, save_device, training, drop_prob):
        super(ReductCell, self).__init__()
        self.save_device = save_device
        self.training = training
        self.drop_prob = drop_prob

    def forward(self, x_in0, x_in1):
        # x_in0: output of h-1 cell
        # x_in1: output of h-2 cell
        pass
