import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda')


class Selection(nn.Module):
    def __init__(self, in_channels, reduction, save_device=torch.device('cpu')):
        super(Selection, self).__init__()
        self.save_device = save_device

        self.reduction = reduction
        if self.reduction:
            stride = 2
            self.Ident_bn = nn.BatchNorm2d(in_channels)
            self.Ident = nn.Conv2d(in_channels, in_channels, 1, stride=stride, bias=False)
        else:
            stride = 1
        self.stride = stride
        self.Sep_3_bn_0 = nn.BatchNorm2d(in_channels)
        self.Sep_3_0 = nn.Conv2d(in_channels, in_channels, 3, stride=stride, groups=in_channels, bias=False)
        self.Sep_3_ident_0 = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        self.Sep_3_bn_1 = nn.BatchNorm2d(in_channels)
        self.Sep_3_1 = nn.Conv2d(in_channels, in_channels, 3, stride=1, groups=in_channels, bias=False)
        self.Sep_3_ident_1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        self.Sep_5_bn_0 = nn.BatchNorm2d(in_channels)
        self.Sep_5_0 = nn.Conv2d(in_channels, in_channels, 5, stride=stride, groups=in_channels, bias=False)
        self.Sep_5_ident_0 = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        self.Sep_5_bn_1 = nn.BatchNorm2d(in_channels)
        self.Sep_5_1 = nn.Conv2d(in_channels, in_channels, 5, stride=1, groups=in_channels, bias=False)
        self.Sep_5_ident_1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        self.Dil_3_bn = nn.BatchNorm2d(in_channels)
        self.Dil_3 = nn.Conv2d(in_channels, in_channels, 3, stride=stride, dilation=2, bias=False)
        self.Dil_3_ident = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        self.Dil_5_bn = nn.BatchNorm2d(in_channels)
        self.Dil_5 = nn.Conv2d(in_channels, in_channels, 5, stride=stride, dilation=2, bias=False)
        self.Dil_5_ident = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        self.Max_Pool = nn.MaxPool2d(3, stride=stride, return_indices=True)
        self.pool_indices = None

        self.Avg_Pool = nn.AvgPool2d(3, stride=stride)

        self.switch_all = nn.Parameter(torch.ones(1))
        self.switch_sep_3 = nn.Parameter(torch.ones(1))
        self.switch_sep_5 = nn.Parameter(torch.ones(1))
        self.switch_dil_3 = nn.Parameter(torch.ones(1))
        self.switch_dil_5 = nn.Parameter(torch.ones(1))
        self.switch_max_pool = nn.Parameter(torch.ones(1))
        self.switch_avg_pool = nn.Parameter(torch.ones(1))
        self.switch_ident = nn.Parameter(torch.ones(1))

        self.data = None

    def forward(self, x):
        x = x * self.switch_all
        self.data = {'input': x.detach().to(self.save_device)}

        x_sep_3 = self.Sep_3_bn_0(x)
        self.data['sep_3_bn_0'] = x_sep_3.detach().to(self.save_device)
        x_sep_3 = F.relu(x_sep_3)
        x_sep_3.retain_grad()
        self.data['sep_3_relu_0'] = x_sep_3
        x_sep_3 = F.pad(x_sep_3, [1] * 4)
        x_sep_3.retain_grad()
        self.data['sep_3_padded_0'] = x_sep_3
        x_sep_3 = self.Sep_3_0(x_sep_3)
        x_sep_3.retain_grad()
        self.data['sep_3_0'] = x_sep_3
        x_sep_3 = self.Sep_3_ident_0(x_sep_3)
        self.data['sep_3_ident_0'] = x_sep_3.detach().to(self.save_device)

        x_sep_3 = self.Sep_3_bn_1(x_sep_3)
        self.data['sep_3_bn_1'] = x_sep_3.detach().to(self.save_device)
        x_sep_3 = F.relu(x_sep_3)
        x_sep_3.retain_grad()
        self.data['sep_3_relu_1'] = x_sep_3
        x_sep_3 = F.pad(x_sep_3, [1] * 4)
        x_sep_3.retain_grad()
        self.data['sep_3_padded_1'] = x_sep_3
        x_sep_3 = self.Sep_3_1(x_sep_3)
        x_sep_3.retain_grad()
        self.data['sep_3_1'] = x_sep_3
        x_sep_3 = self.Sep_3_ident_1(x_sep_3)
        self.data['sep_3_ident_1'] = x_sep_3.detach().to(self.save_device)
        x_sep_3 = x_sep_3 * self.switch_sep_3
        self.data['sep_3'] = x_sep_3.detach().to(self.save_device)

        x_sep_5 = self.Sep_5_bn_0(x)
        self.data['sep_5_bn_0'] = x_sep_5.detach().to(self.save_device)
        x_sep_5 = F.relu(x_sep_5)
        x_sep_5.retain_grad()
        self.data['sep_5_relu_0'] = x_sep_5
        x_sep_5 = F.pad(x_sep_5, [2] * 4)
        x_sep_5.retain_grad()
        self.data['sep_5_padded_0'] = x_sep_5
        x_sep_5 = self.Sep_5_0(x_sep_5)
        x_sep_5.retain_grad()
        self.data['sep_5_0'] = x_sep_5
        x_sep_5 = self.Sep_5_ident_0(x_sep_5)
        self.data['sep_5_ident_0'] = x_sep_5.detach().to(self.save_device)

        x_sep_5 = self.Sep_5_bn_1(x_sep_5)
        self.data['sep_5_bn_1'] = x_sep_5.detach().to(self.save_device)
        x_sep_5 = F.relu(x_sep_5)
        x_sep_5.retain_grad()
        self.data['sep_5_relu_1'] = x_sep_5
        x_sep_5 = F.pad(x_sep_5, [2] * 4)
        x_sep_5.retain_grad()
        self.data['sep_5_padded_1'] = x_sep_5
        x_sep_5 = self.Sep_5_1(x_sep_5)
        x_sep_5.retain_grad()
        self.data['sep_5_1'] = x_sep_5
        x_sep_5 = self.Sep_5_ident_1(x_sep_5)
        self.data['sep_5_ident_1'] = x_sep_5.detach().to(self.save_device)
        x_sep_5 = x_sep_5 * self.switch_sep_5
        self.data['sep_5'] = x_sep_5.detach().to(self.save_device)

        x_dil_3 = self.Dil_3_bn(x)
        self.data['dil_3_bn'] = x_dil_3.detach().to(self.save_device)
        x_dil_3 = F.relu(x_dil_3)
        x_dil_3.retain_grad()
        self.data['dil_3_relu'] = x_dil_3
        x_dil_3 = F.pad(x_dil_3, [2] * 4)
        x_dil_3.retain_grad()
        self.data['dil_3_padded'] = x_dil_3
        x_dil_3 = self.Dil_3(x_dil_3)
        x_dil_3.retain_grad()
        self.data['dil_3_'] = x_dil_3
        x_dil_3 = self.Dil_3_ident(x_dil_3)
        self.data['dil_3_ident'] = x_dil_3.detach().to(self.save_device)
        x_dil_3 = x_dil_3 * self.switch_dil_3
        self.data['dil_3'] = x_dil_3.detach().to(self.save_device)

        x_dil_5 = self.Dil_5_bn(x)
        self.data['dil_5_bn'] = x_dil_5.detach().to(self.save_device)
        x_dil_5 = F.relu(x_dil_5)
        x_dil_5.retain_grad()
        self.data['dil_5_relu'] = x_dil_5
        x_dil_5 = F.pad(x_dil_5, [4] * 4)
        x_dil_5.retain_grad()
        self.data['dil_5_padded'] = x_dil_5
        x_dil_5 = self.Dil_5(x_dil_5)
        x_dil_5.retain_grad()
        self.data['dil_5_'] = x_dil_5
        x_dil_5 = self.Dil_5_ident(x_dil_5)
        self.data['dil_5_ident'] = x_dil_5.detach().to(self.save_device)
        x_dil_5 = x_dil_5 * self.switch_dil_5
        self.data['dil_5'] = x_dil_5.detach().to(self.save_device)

        x_max = F.pad(x, [1] * 4)
        self.data['max_pool_padded'] = x_max.detach().to(self.save_device)
        x_max, self.pool_indices = self.Max_Pool(x_max)
        self.data['max_pool_'] = x_max.detach().to(self.save_device)
        x_max = x_max * self.switch_max_pool
        self.data['max_pool'] = x_max.detach().to(self.save_device)

        x_avg = F.pad(x, [1] * 4)
        self.data['avg_pool_padded'] = x_avg.detach().to(self.save_device)
        x_avg = self.Avg_Pool(x_avg)
        self.data['avg_pool_'] = x_avg.detach().to(self.save_device)
        x_avg = x_avg * self.switch_avg_pool
        self.data['avg_pool'] = x_avg.detach().to(self.save_device)

        if self.reduction:
            x_ident = self.Ident_bn(x)
            x_ident.retain_grad()
            self.data['ident_bn'] = x_ident
            x_ident = self.Ident(x_ident)
            self.data['ident_'] = x_ident.detach().to(self.save_device)
        else:
            x_ident = x
            self.data['ident_'] = x_ident.detach().to(self.save_device)
        x_ident = x_ident * self.switch_ident
        self.data['ident'] = x_ident.detach().to(self.save_device)

        x = x_sep_3 + x_sep_5 + x_dil_3 + x_dil_5 + x_max + x_avg + x_ident
        self.data['sum'] = x.detach().to(self.save_device)

        return x


class NormalCell(nn.Module):
    def __init__(self, out_channels, save_device=torch.device('cpu')):
        super(NormalCell, self).__init__()
        self.save_device = save_device
        # in0: closer source
        self.in0_0 = Selection(out_channels, reduction=False, save_device=save_device)
        self.in0_1 = Selection(out_channels, reduction=False, save_device=save_device)
        self.in0_2 = Selection(out_channels, reduction=False, save_device=save_device)

        self.in1_0 = Selection(out_channels, reduction=False, save_device=save_device)
        self.in1_1 = Selection(out_channels, reduction=False, save_device=save_device)
        self.in1_2 = Selection(out_channels, reduction=False, save_device=save_device)
        self.in1_3 = Selection(out_channels, reduction=False, save_device=save_device)

        self.sel_0_3 = Selection(out_channels, reduction=False, save_device=save_device)

        self.data = None

    def forward(self, x_in0, x_in1):
        self.data = {'x_in0': x_in0.detach().to(self.save_device), 'x_in1': x_in1.detach().to(self.save_device)}

        x_in0_0 = self.in0_0(x_in0)
        self.data['x_in0_0'] = x_in0_0.detach().to(self.save_device)
        x_in1_0 = self.in1_0(x_in1)
        self.data['x_in1_0'] = x_in1_0.detach().to(self.save_device)
        x_0 = x_in0_0 + x_in1_0
        self.data['x_0'] = x_0.detach().to(self.save_device)

        x_in0_1 = self.in0_1(x_in0)
        self.data['x_in0_1'] = x_in0_1.detach().to(self.save_device)
        x_in1_1 = self.in1_1(x_in1)
        self.data['x_in1_1'] = x_in0_1.detach().to(self.save_device)
        x_1 = x_in0_1 + x_in1_1
        self.data['x_1'] = x_1.detach().to(self.save_device)

        x_in0_2 = self.in0_2(x_in0)
        self.data['x_in0_2'] = x_in0_2.detach().to(self.save_device)
        x_in1_2 = self.in1_2(x_in1)
        self.data['x_in1_2'] = x_in1_2.detach().to(self.save_device)
        x_2 = x_in0_2 + x_in1_2
        self.data['x_2'] = x_2.detach().to(self.save_device)

        x_in1_3 = self.in1_3(x_in0)
        self.data['x_in1_3'] = x_in1_3.detach().to(self.save_device)
        x_0_3 = self.sel_0_3(x_0)
        self.data['x_0_3'] = x_0_3.detach().to(self.save_device)
        x_3 = x_in1_3 + x_0_3
        self.data['x_3'] = x_3.detach().to(self.save_device)

        x = torch.cat([x_0, x_1, x_2, x_3], dim=1)
        self.data['cat'] = x.detach().to(self.save_device)

        return x


class ReductCell(nn.Module):
    def __init__(self, out_channels, save_device=torch.device('cpu')):
        super(ReductCell, self).__init__()
        self.save_device = save_device
        # in0: closer source
        self.in0_0 = Selection(out_channels, reduction=True, save_device=save_device)

        self.in1_0 = Selection(out_channels, reduction=True, save_device=save_device)
        self.in1_1 = Selection(out_channels, reduction=True, save_device=save_device)
        self.in1_2 = Selection(out_channels, reduction=True, save_device=save_device)
        self.in1_3 = Selection(out_channels, reduction=True, save_device=save_device)

        self.sel_0_1 = Selection(out_channels, reduction=False, save_device=save_device)
        self.sel_0_2 = Selection(out_channels, reduction=False, save_device=save_device)
        self.sel_0_3 = Selection(out_channels, reduction=False, save_device=save_device)

        self.data = None

    def forward(self, x_in0, x_in1):
        self.data = {'x_in0': x_in0.detach().to(self.save_device), 'x_in1': x_in1.detach().to(self.save_device)}
        x_in0_0 = self.in0_0(x_in0)
        self.data['x_in0_0'] = x_in0_0.detach().to(self.save_device)
        x_in1_0 = self.in1_0(x_in1)
        self.data['x_in1_0'] = x_in1_0.detach().to(self.save_device)
        x_0 = x_in0_0 + x_in1_0
        self.data['x_0'] = x_0.detach().to(self.save_device)

        x_in1_1 = self.in1_1(x_in1)
        self.data['x_in1_1'] = x_in1_1.detach().to(self.save_device)
        x_0_1 = self.sel_0_1(x_0)
        self.data['x_0_1'] = x_0_1.detach().to(self.save_device)
        x_1 = x_in1_1 + x_0_1
        self.data['x_1'] = x_1.detach().to(self.save_device)

        x_in1_2 = self.in1_2(x_in1)
        self.data['x_in1_2'] = x_in1_2.detach().to(self.save_device)
        x_0_2 = self.sel_0_2(x_0)
        self.data['x_0_2'] = x_0_2.detach().to(self.save_device)
        x_2 = x_in1_2 + x_0_2
        self.data['x_2'] = x_2.detach().to(self.save_device)

        x_in1_3 = self.in1_3(x_in1)
        self.data['x_in1_3'] = x_in1_3.detach().to(self.save_device)
        x_0_3 = self.sel_0_3(x_0)
        self.data['x_0_3'] = x_0_3.detach().to(self.save_device)
        x_3 = x_in1_3 + x_0_3
        self.data['x_3'] = x_3.detach().to(self.save_device)

        x = torch.cat([x_0, x_1, x_2, x_3], dim=1)
        self.data['cat'] = x.detach().to(self.save_device)

        return x


class CReduct(nn.Module):
    def __init__(self, in_channels, out_channels, size_reduct=False, save_device=torch.device('cpu')):
        super(CReduct, self).__init__()
        self.save_device = save_device

        self.size_reduct = size_reduct
        if size_reduct:
            self.size_reduct = nn.Conv2d(in_channels, in_channels, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        self.data = None

    def forward(self, x):
        x.retain_grad()
        self.data = {'input': x}
        if self.size_reduct:
            x = self.size_reduct(x)
            self.data['reducted'] = x.detach().to(self.save_device)
        x = self.bn(x)
        self.data['bn'] = x.detach().to(self.save_device)
        x = F.relu(x)
        x.retain_grad()
        self.data['relu'] = x
        x = self.conv(x)
        self.data['conv'] = x.detach().to(self.save_device)

        return x


class NetWork(nn.Module):
    def __init__(self, num_classes, save_device=torch.device('cpu')):
        super(NetWork, self).__init__()
        self.save_device = save_device

        cell_channels = 36
        stem_channels = cell_channels * 3
        self.conv = nn.Conv2d(3, stem_channels, 3, padding=1, bias=False)

        self.reduct_0 = CReduct(stem_channels, cell_channels, save_device=save_device)
        self.reduct_1 = CReduct(stem_channels, cell_channels, save_device=save_device)
        self.norm_cell_0 = NormalCell(cell_channels, save_device=save_device)

        self.reduct_2 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_3 = CReduct(stem_channels, cell_channels, save_device=save_device)
        self.norm_cell_1 = NormalCell(cell_channels, save_device=save_device)

        self.reduct_4 = CReduct(4 * cell_channels, 2 * cell_channels, save_device=save_device)
        self.reduct_5 = CReduct(4 * cell_channels, 2 * cell_channels, save_device=save_device)
        cell_channels *= 2
        self.reduct_cell_0 = ReductCell(cell_channels, save_device=save_device)

        self.reduct_6 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_7 = CReduct(2 * cell_channels, cell_channels, size_reduct=True, save_device=save_device)
        self.norm_cell_2 = NormalCell(cell_channels, save_device=save_device)

        self.reduct_8 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_9 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.norm_cell_3 = NormalCell(cell_channels, save_device=save_device)

        self.reduct_10 = CReduct(4 * cell_channels, 2 * cell_channels, save_device=save_device)
        self.reduct_11 = CReduct(4 * cell_channels, 2 * cell_channels, save_device=save_device)
        cell_channels *= 2
        self.reduct_cell_1 = ReductCell(cell_channels, save_device=save_device)

        self.reduct_12 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_13 = CReduct(2 * cell_channels, cell_channels, size_reduct=True, save_device=save_device)
        self.norm_cell_4 = NormalCell(cell_channels, save_device=save_device)

        self.reduct_14 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_15 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.norm_cell_5 = NormalCell(cell_channels, save_device=save_device)

        self.bn = nn.BatchNorm2d(4 * cell_channels)
        self.pool = nn.AvgPool2d(8)
        self.linear = nn.Linear(4 * cell_channels, num_classes)

        self.data = None

    def forward(self, x):
        self.data = {}
        x_prev_prev = self.conv(x)
        x_prev = x_prev_prev

        x, x_ = self.reduct_0(x_prev), self.reduct_1(x_prev_prev)
        x_prev = self.norm_cell_0(x, x_)

        x, x_ = self.reduct_2(x_prev), self.reduct_3(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_1(x, x_), x_prev

        x, x_ = self.reduct_4(x_prev), self.reduct_5(x_prev_prev)
        x_prev, x_prev_prev = self.reduct_cell_0(x, x_), x_prev

        x, x_ = self.reduct_6(x_prev), self.reduct_7(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_2(x, x_), x_prev

        x, x_ = self.reduct_8(x_prev), self.reduct_9(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_3(x, x_), x_prev

        x, x_ = self.reduct_10(x_prev), self.reduct_11(x_prev_prev)
        x_prev, x_prev_prev = self.reduct_cell_1(x, x_), x_prev

        x, x_ = self.reduct_12(x_prev), self.reduct_13(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_4(x, x_), x_prev

        x, x_ = self.reduct_14(x_prev), self.reduct_15(x_prev_prev)
        x_prev = self.norm_cell_5(x, x_)

        x = self.bn(x_prev)
        self.data['bn'] = x.detach().to(self.save_device)
        x = F.relu(x)
        x.retain_grad()
        self.data['relu'] = x
        x = self.pool(x)
        self.data['pool'] = x.detach().to(self.save_device)
        x = x.view(x.size(0), -1)
        x.retain_grad()
        self.data['flattened'] = x
        x = self.linear(x)
        self.data['fc'] = x.detach().to(self.save_device)
        x = F.log_softmax(x, dim=1)
        self.data['output'] = x.detach().to(self.save_device)

        return x

    def return_params(self):
        normal = [param for name, param in self.named_parameters() if 'switch' not in name]
        switch = [param for name, param in self.named_parameters() if 'switch' in name]

        return normal, switch
