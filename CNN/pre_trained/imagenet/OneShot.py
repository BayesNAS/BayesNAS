from operations import *

DEVICE = torch.device('cuda')


class NormalCell(nn.Module):
    def __init__(self, out_channels, training, drop_prob, save_device=torch.device('cpu')):
        super(NormalCell, self).__init__()
        self.save_device = save_device
        self.training = training
        self.drop_prob = drop_prob
        # in0: closer source
        self.in0_0 = Sep5(out_channels, reduction=False, save_device=save_device)

        self.in0_1_0 = MaxPool(out_channels, reduction=False, save_device=save_device)
        self.in0_1_1 = Ident(out_channels, reduction=False, save_device=save_device)

        self.in0_2 = Ident(out_channels, reduction=False, save_device=save_device)

        self.in1_0_0 = Sep5(out_channels, reduction=False, save_device=save_device)
        self.in1_0_1 = MaxPool(out_channels, reduction=False, save_device=save_device)
        self.in1_0_2 = AvgPool(out_channels, reduction=False, save_device=save_device)
        self.in1_0_3 = Ident(out_channels, reduction=False, save_device=save_device)

        self.in1_2 = MaxPool(out_channels, reduction=False, save_device=save_device)

        self.sel_0_3 = AvgPool(out_channels, reduction=False, save_device=save_device)

    def forward(self, x_in0, x_in1):
        x_in0_0 = F.dropout2d(self.in0_0(x_in0), training=self.training, p=self.drop_prob)
        x_in1_0 = F.dropout2d(self.in1_0_0(x_in1), training=self.training, p=self.drop_prob) + \
                  F.dropout2d(self.in1_0_1(x_in1), training=self.training, p=self.drop_prob) + \
                  F.dropout2d(self.in1_0_2(x_in1), training=self.training, p=self.drop_prob) + \
                  F.dropout2d(self.in1_0_3(x_in1), training=self.training, p=self.drop_prob)
        x_0 = x_in0_0 + x_in1_0

        x_1 = F.dropout2d(self.in0_1_0(x_in0), training=self.training, p=self.drop_prob) + \
              F.dropout2d(self.in0_1_1(x_in0), training=self.training, p=self.drop_prob)

        x_2 = F.dropout2d(self.in0_2(x_in0), training=self.training, p=self.drop_prob) + \
              F.dropout2d(self.in1_2(x_in1), training=self.training, p=self.drop_prob)

        x_3 = F.dropout2d(self.sel_0_3(x_0), training=self.training, p=self.drop_prob)

        x = torch.cat([x_0, x_1, x_2, x_3], dim=1)

        return x


class ReductCell(nn.Module):
    def __init__(self, out_channels, drop_prob, training, save_device=torch.device('cpu')):
        super(ReductCell, self).__init__()
        self.save_device = save_device
        self.training = training
        self.drop_prob = drop_prob
        # in0: closer source
        self.in0_0 = Ident(out_channels, reduction=True, save_device=save_device)

        self.in1_0_0 = Sep3(out_channels, reduction=True, save_device=save_device)
        self.in1_0_1 = AvgPool(out_channels, reduction=True, save_device=save_device)
        self.in1_0_2 = MaxPool(out_channels, reduction=True, save_device=save_device)
        self.in1_0_3 = Sep5(out_channels, reduction=True, save_device=save_device)

        self.in1_1_0 = MaxPool(out_channels, reduction=True, save_device=save_device)
        self.in1_1_1 = AvgPool(out_channels, reduction=True, save_device=save_device)

        self.in1_2_0 = Sep3(out_channels, reduction=True, save_device=save_device)
        self.in1_2_1 = AvgPool(out_channels, reduction=True, save_device=save_device)

        self.sel_0_3 = Sep5(out_channels, reduction=False, save_device=save_device)

    def forward(self, x_in0, x_in1):
        x_in0_0 = F.dropout2d(self.in0_0(x_in0), training=self.training, p=self.drop_prob)
        x_in1_0 = F.dropout2d(self.in1_0_0(x_in1), training=self.training, p=self.drop_prob) + \
                  F.dropout2d(self.in1_0_1(x_in1), training=self.training, p=self.drop_prob) + \
                  F.dropout2d(self.in1_0_2(x_in1), training=self.training, p=self.drop_prob) + \
                  F.dropout2d(self.in1_0_3(x_in1), training=self.training, p=self.drop_prob)
        x_0 = x_in0_0 + x_in1_0

        x_1 = F.dropout2d(self.in1_1_0(x_in1), training=self.training, p=self.drop_prob) + \
              F.dropout2d(self.in1_1_1(x_in1), training=self.training, p=self.drop_prob)

        x_2 = F.dropout2d(self.in1_2_0(x_in1), training=self.training, p=self.drop_prob) + \
              F.dropout2d(self.in1_2_1(x_in1), training=self.training, p=self.drop_prob)

        x_3 = F.dropout2d(self.sel_0_3(x_0), training=self.training, p=self.drop_prob)

        x = torch.cat([x_0, x_1, x_2, x_3], dim=1)

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

    def forward(self, x):
        if self.size_reduct:
            x = self.size_reduct(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv(x)

        return x


class NetWork(nn.Module):
    def __init__(self, num_classes, save_device=torch.device('cpu'), training=True, drop_prob=0):
        super(NetWork, self).__init__()
        self.save_device = save_device
        self.drop_prob = drop_prob

        cell_channels = 48
        self.stem_0 = nn.Sequential(
            nn.Conv2d(3, cell_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cell_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(cell_channels // 2, cell_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cell_channels),
        )
        self.stem_1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(cell_channels, cell_channels, 3, stride=2, padding=1, bias=False),
        )
        # layer 0
        self.reduct_0 = CReduct(cell_channels, cell_channels, save_device=save_device)
        self.reduct_1 = CReduct(cell_channels, cell_channels, save_device=save_device, size_reduct=True)
        self.norm_cell_0 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)
        ########################################################
        self.reduct_2 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_3 = CReduct(cell_channels, cell_channels, save_device=save_device)
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
        self.reduct_8 = CReduct(4 * cell_channels, 2 * cell_channels, save_device=save_device)
        self.reduct_9 = CReduct(4 * cell_channels, 2 * cell_channels, save_device=save_device)
        cell_channels *= 2
        self.reduct_cell_0 = ReductCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)

        # layer 1
        self.reduct_10 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_11 = CReduct(2 * cell_channels, cell_channels, save_device=save_device, size_reduct=True)
        self.norm_cell_4 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)
        ########################################################
        self.reduct_12 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_13 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.norm_cell_5 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)
        ########################################################
        self.reduct_14 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_15 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.norm_cell_6 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)
        ########################################################
        self.reduct_16 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_17 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.norm_cell_7 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)

        ########################################################
        self.reduct_18 = CReduct(4 * cell_channels, 2 * cell_channels, save_device=save_device)
        self.reduct_19 = CReduct(4 * cell_channels, 2 * cell_channels, save_device=save_device)
        cell_channels *= 2
        self.reduct_cell_1 = ReductCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)

        # layer 2
        self.reduct_20 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_21 = CReduct(2 * cell_channels, cell_channels, save_device=save_device, size_reduct=True)
        self.norm_cell_8 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)
        ########################################################
        self.reduct_22 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_23 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.norm_cell_9 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)
        ########################################################
        self.reduct_24 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_25 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.norm_cell_10 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)
        ########################################################
        self.reduct_26 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.reduct_27 = CReduct(4 * cell_channels, cell_channels, save_device=save_device)
        self.norm_cell_11 = NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob)

        self.bn = nn.BatchNorm2d(4 * cell_channels)
        self.pool = nn.AvgPool2d(7)
        self.linear = nn.Linear(4 * cell_channels, num_classes)

    def forward(self, x):
        x_prev_prev = self.stem_0(x)
        x_prev = self.stem_1(x_prev_prev)

        # norm_cell_0
        x, x_ = self.reduct_0(x_prev), self.reduct_1(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_0(x, x_), x_prev
        # norm_cell_1
        x, x_ = self.reduct_2(x_prev), self.reduct_3(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_1(x, x_), x_prev
        # norm_cell_1
        x, x_ = self.reduct_4(x_prev), self.reduct_5(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_2(x, x_), x_prev
        # norm_cell_1
        x, x_ = self.reduct_6(x_prev), self.reduct_7(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_3(x, x_), x_prev

        # reduct_cell_0
        x, x_ = self.reduct_8(x_prev), self.reduct_9(x_prev_prev)
        x_prev, x_prev_prev = self.reduct_cell_0(x, x_), x_prev

        # norm_cell_2
        x, x_ = self.reduct_10(x_prev), self.reduct_11(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_4(x, x_), x_prev
        # norm_cell_3
        x, x_ = self.reduct_12(x_prev), self.reduct_13(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_5(x, x_), x_prev
        # norm_cell_2
        x, x_ = self.reduct_14(x_prev), self.reduct_15(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_6(x, x_), x_prev
        # norm_cell_3
        x, x_ = self.reduct_16(x_prev), self.reduct_17(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_7(x, x_), x_prev

        # reduct_cell_0
        x, x_ = self.reduct_18(x_prev), self.reduct_19(x_prev_prev)
        x_prev, x_prev_prev = self.reduct_cell_1(x, x_), x_prev

        # norm_cell_2
        x, x_ = self.reduct_20(x_prev), self.reduct_21(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_8(x, x_), x_prev
        # norm_cell_3
        x, x_ = self.reduct_22(x_prev), self.reduct_23(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_9(x, x_), x_prev
        # norm_cell_2
        x, x_ = self.reduct_24(x_prev), self.reduct_25(x_prev_prev)
        x_prev, x_prev_prev = self.norm_cell_10(x, x_), x_prev
        # norm_cell_3
        x, x_ = self.reduct_26(x_prev), self.reduct_27(x_prev_prev)
        x_prev = self.norm_cell_11(x, x_)

        x = self.bn(x_prev)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        # no log_softmax

        return x

    def no_drop(self):
        for idx in range(12):
            norm_cell = getattr(self, 'norm_cell_' + str(idx))
            norm_cell.training = False
        for idx in range(2):
            reduct_cell = getattr(self, 'reduct_cell_' + str(idx))
            reduct_cell.training = False

    def drop(self):
        for idx in range(12):
            norm_cell = getattr(self, 'norm_cell_' + str(idx))
            norm_cell.training = True
        for idx in range(2):
            reduct_cell = getattr(self, 'reduct_cell_' + str(idx))
            reduct_cell.training = True

    def drop_prob_update(self, epoch, total_epoch):
        drop_prob = self.drop_prob * epoch / total_epoch
        for idx in range(12):
            norm_cell = getattr(self, 'norm_cell_' + str(idx))
            norm_cell.drop_prob = drop_prob
        for idx in range(2):
            reduct_cell = getattr(self, 'reduct_cell_' + str(idx))
            reduct_cell.drop_prob = drop_prob
