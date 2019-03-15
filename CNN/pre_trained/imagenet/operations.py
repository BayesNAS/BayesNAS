import torch
import torch.nn as nn
import torch.nn.functional as F

class Sep3(nn.Module):
    def __init__(self, in_channels, reduction, save_device=torch.device('cpu')):
        super(Sep3, self).__init__()
        self.save_device = save_device
        self.reduction = reduction
        if self.reduction:
            stride = 2
        else:
            stride = 1
        self.stride = stride

        self.Sep_3_bn_0 = nn.BatchNorm2d(in_channels)
        self.Sep_3_0 = nn.Conv2d(in_channels, in_channels, 3, stride=stride, groups=in_channels, bias=False)
        self.Sep_3_ident_0 = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        self.Sep_3_bn_1 = nn.BatchNorm2d(in_channels)
        self.Sep_3_1 = nn.Conv2d(in_channels, in_channels, 3, stride=1, groups=in_channels, bias=False)
        self.Sep_3_ident_1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)

    def forward(self, x):
        x_sep_3 = self.Sep_3_bn_0(x)
        x_sep_3 = F.relu(x_sep_3)
        x_sep_3 = F.pad(x_sep_3, [1] * 4)
        x_sep_3 = self.Sep_3_0(x_sep_3)
        x_sep_3 = self.Sep_3_ident_0(x_sep_3)

        x_sep_3 = self.Sep_3_bn_1(x_sep_3)
        x_sep_3 = F.relu(x_sep_3)
        x_sep_3 = F.pad(x_sep_3, [1] * 4)
        x_sep_3 = self.Sep_3_1(x_sep_3)
        x_sep_3 = self.Sep_3_ident_1(x_sep_3)

        return x_sep_3


class Sep5(nn.Module):
    def __init__(self, in_channels, reduction, save_device=torch.device('cpu')):
        super(Sep5, self).__init__()
        self.save_device = save_device
        self.reduction = reduction
        if self.reduction:
            stride = 2
        else:
            stride = 1
        self.stride = stride

        self.Sep_5_bn_0 = nn.BatchNorm2d(in_channels)
        self.Sep_5_0 = nn.Conv2d(in_channels, in_channels, 5, stride=stride, groups=in_channels, bias=False)
        self.Sep_5_ident_0 = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        self.Sep_5_bn_1 = nn.BatchNorm2d(in_channels)
        self.Sep_5_1 = nn.Conv2d(in_channels, in_channels, 5, stride=1, groups=in_channels, bias=False)
        self.Sep_5_ident_1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)

    def forward(self, x):
        x_sep_5 = self.Sep_5_bn_0(x)
        x_sep_5 = F.relu(x_sep_5)
        x_sep_5 = F.pad(x_sep_5, [2] * 4)
        x_sep_5 = self.Sep_5_0(x_sep_5)
        x_sep_5 = self.Sep_5_ident_0(x_sep_5)

        x_sep_5 = self.Sep_5_bn_1(x_sep_5)
        x_sep_5 = F.relu(x_sep_5)
        x_sep_5 = F.pad(x_sep_5, [2] * 4)
        x_sep_5 = self.Sep_5_1(x_sep_5)
        x_sep_5 = self.Sep_5_ident_1(x_sep_5)

        return x_sep_5


class Dil3(nn.Module):
    def __init__(self, in_channels, reduction, save_device=torch.device('cpu')):
        super(Dil3, self).__init__()
        self.save_device = save_device
        self.reduction = reduction
        if self.reduction:
            stride = 2
        else:
            stride = 1
        self.stride = stride

        self.Dil_3_bn = nn.BatchNorm2d(in_channels)
        self.Dil_3 = nn.Conv2d(in_channels, in_channels, 3, stride=stride, dilation=2, bias=False)
        self.Dil_3_ident = nn.Conv2d(in_channels, in_channels, 1, bias=False)

    def forward(self, x):
        x_dil_3 = self.Dil_3_bn(x)
        x_dil_3 = F.relu(x_dil_3)
        x_dil_3 = F.pad(x_dil_3, [2] * 4)
        x_dil_3 = self.Dil_3(x_dil_3)
        x_dil_3 = self.Dil_3_ident(x_dil_3)

        return x_dil_3


class Dil5(nn.Module):
    def __init__(self, in_channels, reduction, save_device=torch.device('cpu')):
        super(Dil5, self).__init__()
        self.save_device = save_device
        self.reduction = reduction
        if self.reduction:
            stride = 2
        else:
            stride = 1
        self.stride = stride

        self.Dil_5_bn = nn.BatchNorm2d(in_channels)
        self.Dil_5 = nn.Conv2d(in_channels, in_channels, 5, stride=stride, dilation=2, bias=False)
        self.Dil_5_ident = nn.Conv2d(in_channels, in_channels, 1, bias=False)

    def forward(self, x):
        x_dil_5 = self.Dil_5_bn(x)
        x_dil_5 = F.relu(x_dil_5)
        x_dil_5 = F.pad(x_dil_5, [4] * 4)
        x_dil_5 = self.Dil_5(x_dil_5)
        x_dil_5 = self.Dil_5_ident(x_dil_5)

        return x_dil_5


class MaxPool(nn.Module):
    def __init__(self, in_channels, reduction, save_device=torch.device('cpu')):
        super(MaxPool, self).__init__()
        self.save_device = save_device
        self.reduction = reduction
        if self.reduction:
            stride = 2
        else:
            stride = 1
        self.stride = stride

        self.Max_Pool = nn.MaxPool2d(3, stride=stride, return_indices=True)
        self.pool_indices = None

    def forward(self, x):
        x_max = F.pad(x, [1] * 4)
        x_max, self.pool_indices = self.Max_Pool(x_max)

        return x_max


class AvgPool(nn.Module):
    def __init__(self, in_channels, reduction, save_device=torch.device('cpu')):
        super(AvgPool, self).__init__()
        self.save_device = save_device
        self.reduction = reduction
        if self.reduction:
            stride = 2
        else:
            stride = 1
        self.stride = stride

        self.Avg_Pool = nn.AvgPool2d(3, stride=stride)

    def forward(self, x):
        x_avg = F.pad(x, [1] * 4)
        x_avg = self.Avg_Pool(x_avg)

        return x_avg


class Ident(nn.Module):
    def __init__(self, in_channels, reduction, save_device=torch.device('cpu')):
        super(Ident, self).__init__()
        self.save_device = save_device
        self.reduction = reduction
        if self.reduction:
            stride = 2
            # op6 identity
            self.Ident_bn = nn.BatchNorm2d(in_channels)
            self.Ident = nn.Conv2d(in_channels, in_channels, 1, stride=stride, bias=False)
        else:
            stride = 1
        self.stride = stride

    def forward(self, x):
        if self.reduction:
            x_ident = self.Ident_bn(x)
            x_ident = self.Ident(x_ident)
        else:
            x_ident = x

        return x_ident