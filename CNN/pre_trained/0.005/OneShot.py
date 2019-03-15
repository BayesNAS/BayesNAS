import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda')


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


class NormalCell(nn.Module):
    def __init__(self, out_channels, training, drop_prob, save_device=torch.device('cpu')):
        super(NormalCell, self).__init__()
        self.save_device = save_device
        self.training = training
        self.drop_prob = drop_prob
        # in0: closer source
        self.in0_1 = Ident(out_channels, reduction=False, save_device=save_device)

        self.in0_2 = AvgPool(out_channels, reduction=False, save_device=save_device)

        self.in1_0_0 = MaxPool(out_channels, reduction=False, save_device=save_device)
        self.in1_0_1 = AvgPool(out_channels, reduction=False, save_device=save_device)
        self.in1_0_2 = Ident(out_channels, reduction=False, save_device=save_device)

        self.in1_1 = Sep5(out_channels, reduction=False, save_device=save_device)

        self.in1_2_0 = Ident(out_channels, reduction=False, save_device=save_device)
        self.in1_2_1 = MaxPool(out_channels, reduction=False, save_device=save_device)

        self.in1_3 = Sep5(out_channels, reduction=False, save_device=save_device)

        self.sel_0_3 = MaxPool(out_channels, reduction=False, save_device=save_device)

    def forward(self, x_in0, x_in1):
        x_in1_0 = F.dropout2d(self.in1_0_0(x_in1), training=self.training, p=self.drop_prob) + \
                  F.dropout2d(self.in1_0_1(x_in1), training=self.training, p=self.drop_prob) + \
                  F.dropout2d(self.in1_0_2(x_in1), training=self.training, p=self.drop_prob)
        x_0 = x_in1_0

        x_1 = F.dropout2d(self.in0_1(x_in0), training=self.training, p=self.drop_prob) + \
              F.dropout2d(self.in1_1(x_in1), training=self.training, p=self.drop_prob)

        x_2 = F.dropout2d(self.in0_2(x_in0), training=self.training, p=self.drop_prob) + \
              F.dropout2d(self.in1_2_0(x_in1), training=self.training, p=self.drop_prob) + \
              F.dropout2d(self.in1_2_1(x_in1), training=self.training, p=self.drop_prob)

        x_3 = F.dropout2d(self.in1_3(x_in1), training=self.training, p=self.drop_prob) + \
              F.dropout2d(self.sel_0_3(x_0), training=self.training, p=self.drop_prob)

        x = torch.cat([x_0, x_1, x_2, x_3], dim=1)

        return x


class ReductCell(nn.Module):
    def __init__(self, out_channels, drop_prob, training, save_device=torch.device('cpu')):
        super(ReductCell, self).__init__()
        self.save_device = save_device
        self.training = training
        self.drop_prob = drop_prob
        # in0: closer source
        self.in0_0_0 = Sep3(out_channels, reduction=True, save_device=save_device)
        self.in0_0_1 = Sep5(out_channels, reduction=True, save_device=save_device)
        self.in0_0_2 = MaxPool(out_channels, reduction=True, save_device=save_device)

        self.in1_0_0 = AvgPool(out_channels, reduction=True, save_device=save_device)
        self.in1_0_1 = MaxPool(out_channels, reduction=True, save_device=save_device)

        self.in1_2_0 = Ident(out_channels, reduction=True, save_device=save_device)
        self.in1_2_1 = MaxPool(out_channels, reduction=True, save_device=save_device)

        self.in1_3_0 = Sep3(out_channels, reduction=True, save_device=save_device)
        self.in1_3_1 = MaxPool(out_channels, reduction=True, save_device=save_device)

        self.sel_0_1_0 = MaxPool(out_channels, reduction=False, save_device=save_device)
        self.sel_0_1_1 = AvgPool(out_channels, reduction=False, save_device=save_device)
        self.sel_0_1_2 = Ident(out_channels, reduction=False, save_device=save_device)

        self.sel_0_3 = MaxPool(out_channels, reduction=False, save_device=save_device)

    def forward(self, x_in0, x_in1):
        x_in0_0 = F.dropout2d(self.in0_0_0(x_in0), training=self.training, p=self.drop_prob) + \
                  F.dropout2d(self.in0_0_1(x_in0), training=self.training, p=self.drop_prob) + \
                  F.dropout2d(self.in0_0_2(x_in0), training=self.training, p=self.drop_prob)
        x_in1_0 = F.dropout2d(self.in1_0_0(x_in1), training=self.training, p=self.drop_prob) + \
                  F.dropout2d(self.in1_0_1(x_in1), training=self.training, p=self.drop_prob)
        x_0 = x_in0_0 + x_in1_0

        x_1 = F.dropout2d(self.sel_0_1_0(x_0), training=self.training, p=self.drop_prob) + \
              F.dropout2d(self.sel_0_1_1(x_0), training=self.training, p=self.drop_prob) + \
              F.dropout2d(self.sel_0_1_2(x_0), training=self.training, p=self.drop_prob)

        x_2 = F.dropout2d(self.in1_2_0(x_in1), training=self.training, p=self.drop_prob) + \
              F.dropout2d(self.in1_2_1(x_in1), training=self.training, p=self.drop_prob)

        x_3 = F.dropout2d(self.in1_3_0(x_in1), training=self.training, p=self.drop_prob) + \
              F.dropout2d(self.in1_3_1(x_in1), training=self.training, p=self.drop_prob) + \
              F.dropout2d(self.sel_0_3(x_0), training=self.training, p=self.drop_prob)

        x = torch.cat([x_0, x_1, x_2, x_3], dim=1)

        return x


class CReduct(nn.Module):
    def __init__(self, in_channels, out_channels, size_reduct=False, save_device=torch.device('cpu')):
        super(CReduct, self).__init__()
        self.save_device = save_device

        self.size_reduct = size_reduct
        if size_reduct:
            self.size_reduct = nn.Conv2d(in_channels, in_channels//2, 1, stride=2, bias=False)
            self.size_reduct_2 = nn.Conv2d(in_channels, in_channels//2, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        if self.size_reduct:
            x_0 = self.size_reduct(x)
            x_1 = self.size_reduct_2(x[:,:,1:,1:])
            x = torch.cat([x_0,x_1],dim=1)
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv(x)

        return x


class NetWork(nn.Module):
    def __init__(self, num_classes, save_device=torch.device('cpu'), training=True, drop_prob=0.2):
        super(NetWork, self).__init__()
        self.save_device = save_device
        self.drop_prob = drop_prob

        cell_channels = 36
        stem_channels = cell_channels * 3
        self.conv = nn.Conv2d(3, stem_channels, 3, padding=1, bias=False)
        # layer 0
        self.reduct_0 = CReduct(stem_channels, cell_channels, save_device=save_device)
        self.reduct_1 = CReduct(stem_channels, cell_channels, save_device=save_device)
        self.norm_cell_0 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)
        ########################################################
        self.reduct_2 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_3 = CReduct(stem_channels, cell_channels, save_device=save_device)
        self.norm_cell_1 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)
        ########################################################
        self.reduct_4 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_5 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.norm_cell_2 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)
        ########################################################
        self.reduct_6 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_7 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.norm_cell_3 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)
        ########################################################
        self.reduct_8 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_9 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.norm_cell_4 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)
        ########################################################
        self.reduct_10 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_11 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.norm_cell_5 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)

        ########################################################
        self.reduct_12 = CReduct(4 * cell_channels, 2 * cell_channels, save_device=save_device)
        self.reduct_13 = CReduct(4 * cell_channels, 2 * cell_channels, save_device=save_device)
        cell_channels *= 2
        self.reduct_cell_0 = ReductCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)

        # layer 1
        self.reduct_14 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_15 = CReduct(2 * cell_channels, cell_channels, save_device=save_device, size_reduct=True)
        self.norm_cell_6 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)
        ########################################################
        self.reduct_16 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_17 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.norm_cell_7 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)
        ########################################################
        self.reduct_18 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_19 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.norm_cell_8 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)
        ########################################################
        self.reduct_20 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_21 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.norm_cell_9 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)
        ########################################################
        self.reduct_22 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_23 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.norm_cell_10 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)
        ########################################################
        self.reduct_24 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_25 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.norm_cell_11 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)

        ########################################################
        self.reduct_26 = CReduct(4 * cell_channels, 2 * cell_channels, save_device=save_device)
        self.reduct_27 = CReduct(4 * cell_channels, 2 * cell_channels, save_device=save_device)
        cell_channels *= 2
        self.reduct_cell_1 = ReductCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)

        # layer 2
        self.reduct_28 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_29 = CReduct(2 * cell_channels, cell_channels, save_device=save_device, size_reduct=True)
        self.norm_cell_12 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)
        ########################################################
        self.reduct_30 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_31 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.norm_cell_13 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)
        ########################################################
        self.reduct_32 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_33 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.norm_cell_14 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)
        ########################################################
        self.reduct_34 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_35 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.norm_cell_15 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)
        ########################################################
        self.reduct_36 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_37 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.norm_cell_16 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)
        ########################################################
        self.reduct_38 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_39 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.norm_cell_17 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)

        self.bn = nn.BatchNorm2d(4 * cell_channels)
        self.pool = nn.AvgPool2d(8)
        self.linear = nn.Linear(4 * cell_channels, num_classes)

    def forward(self, x):
        x_prev_prev = self.conv(x)
        x_prev = x_prev_prev
        # norm_cell_0
        x, x_ = self.reduct_0(x_prev), self.reduct_1(x_prev_prev)
        x_prev = self.norm_cell_0(x, x_)
        # norm_cell_1
        x, x_ = self.reduct_2(x_prev), self.reduct_3(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_1(x, x_), x_prev
        # norm_cell_1
        x, x_ = self.reduct_4(x_prev), self.reduct_5(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_2(x, x_), x_prev
        # norm_cell_1
        x, x_ = self.reduct_6(x_prev), self.reduct_7(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_3(x, x_), x_prev
        # norm_cell_1
        x, x_ = self.reduct_8(x_prev), self.reduct_9(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_4(x, x_), x_prev
        # norm_cell_1
        x, x_ = self.reduct_10(x_prev), self.reduct_11(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_5(x, x_), x_prev

        # reduct_cell_0
        x, x_ = self.reduct_12(x_prev), self.reduct_13(x_prev_prev)
        x_prev, x_prev_prev = self.reduct_cell_0(x, x_), x_prev

        # norm_cell_2
        x, x_ = self.reduct_14(x_prev), self.reduct_15(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_6(x, x_), x_prev
        # norm_cell_3
        x, x_ = self.reduct_16(x_prev), self.reduct_17(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_7(x, x_), x_prev
        # norm_cell_2
        x, x_ = self.reduct_18(x_prev), self.reduct_19(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_8(x, x_), x_prev
        # norm_cell_3
        x, x_ = self.reduct_20(x_prev), self.reduct_21(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_9(x, x_), x_prev
        # norm_cell_2
        x, x_ = self.reduct_22(x_prev), self.reduct_23(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_10(x, x_), x_prev
        # norm_cell_3
        x, x_ = self.reduct_24(x_prev), self.reduct_25(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_11(x, x_), x_prev

        # reduct_cell_0
        x, x_ = self.reduct_26(x_prev), self.reduct_27(x_prev_prev)
        x_prev, x_prev_prev = self.reduct_cell_1(x, x_), x_prev

        # norm_cell_2
        x, x_ = self.reduct_28(x_prev), self.reduct_29(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_12(x, x_), x_prev
        # norm_cell_3
        x, x_ = self.reduct_30(x_prev), self.reduct_31(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_13(x, x_), x_prev
        # norm_cell_2
        x, x_ = self.reduct_32(x_prev), self.reduct_33(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_14(x, x_), x_prev
        # norm_cell_3
        x, x_ = self.reduct_34(x_prev), self.reduct_35(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_15(x, x_), x_prev
        # norm_cell_2
        x, x_ = self.reduct_36(x_prev), self.reduct_37(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_16(x, x_), x_prev
        # norm_cell_5
        x, x_ = self.reduct_38(x_prev), self.reduct_39(x_prev_prev)
        x_prev = self.norm_cell_17(x, x_)

        x = self.bn(x_prev)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)

        return x

    def no_drop(self):
        for idx in range(18):
            norm_cell = getattr(self, 'norm_cell_' + str(idx))
            norm_cell.training = False
        for idx in range(2):
            reduct_cell = getattr(self, 'reduct_cell_' + str(idx))
            reduct_cell.training = False

    def drop(self):
        for idx in range(18):
            norm_cell = getattr(self, 'norm_cell_' + str(idx))
            norm_cell.training = True
        for idx in range(2):
            reduct_cell = getattr(self, 'reduct_cell_' + str(idx))
            reduct_cell.training = True

    def drop_prob_update(self, epoch, total_epoch):
        drop_prob = self.drop_prob * epoch / total_epoch
        for idx in range(18):
            norm_cell = getattr(self, 'norm_cell_' + str(idx))
            norm_cell.drop_prob = drop_prob
        for idx in range(2):
            reduct_cell = getattr(self, 'reduct_cell_' + str(idx))
            reduct_cell.drop_prob = drop_prob
