import torch
import torch.nn.functional as F
import numpy as np


class Algorithm:
    def __init__(self, model, device, lambda_child, lambda_origin):
        self.device = device
        self.model = model.to(self.device)
        self.lambda_child = lambda_child
        self.lambda_origin = lambda_origin
        self.switch_names = ['switch_all', 'switch_sep_3', 'switch_sep_5', 'switch_dil_3', 'switch_dil_5',
                             'switch_max_pool', 'switch_avg_pool', 'switch_ident']
        self.norm_op_names = ['in0_0', 'in0_1', 'in0_2', 'in1_0', 'in1_1', 'in1_2', 'in1_3', 'sel_0_3']
        self.gamma_norm_cells = {}
        self.omega_norm_cells = {}
        self.cov_norm_cells = {}
        self.hessian_norm_cells = {}
        for cell_idx in range(6):
            self.gamma_norm_cells['norm_cell_' + str(cell_idx)] = {}
            self.omega_norm_cells['norm_cell_' + str(cell_idx)] = {}
            self.cov_norm_cells['norm_cell_' + str(cell_idx)] = {}
            self.hessian_norm_cells['norm_cell_' + str(cell_idx)] = {}
            for op_name in self.norm_op_names:
                self.gamma_norm_cells['norm_cell_' + str(cell_idx)][op_name] = {}
                self.omega_norm_cells['norm_cell_' + str(cell_idx)][op_name] = {}
                self.cov_norm_cells['norm_cell_' + str(cell_idx)][op_name] = {}
                self.hessian_norm_cells['norm_cell_' + str(cell_idx)][op_name] = {}
                for switch_name in self.switch_names:
                    self.gamma_norm_cells['norm_cell_' + str(cell_idx)][op_name][switch_name] = torch.rand(1).to(
                        self.device)
                    self.omega_norm_cells['norm_cell_' + str(cell_idx)][op_name][switch_name] = torch.rand(1).to(
                        self.device)
                    self.cov_norm_cells['norm_cell_' + str(cell_idx)][op_name][switch_name] = torch.rand(1).to(
                        self.device)
                    self.hessian_norm_cells['norm_cell_' + str(cell_idx)][op_name][switch_name] = torch.rand(1).to(
                        self.device)
        self.reduct_op_names = ['in0_0', 'in1_0', 'in1_1', 'in1_2', 'in1_3', 'sel_0_1', 'sel_0_2', 'sel_0_3']
        self.gamma_reduct_cells = {}
        self.omega_reduct_cells = {}
        self.cov_reduct_cells = {}
        self.hessian_reduct_cells = {}
        for cell_idx in range(2):
            self.gamma_reduct_cells['reduct_cell_' + str(cell_idx)] = {}
            self.omega_reduct_cells['reduct_cell_' + str(cell_idx)] = {}
            self.cov_reduct_cells['reduct_cell_' + str(cell_idx)] = {}
            self.hessian_reduct_cells['reduct_cell_' + str(cell_idx)] = {}
            for op_name in self.reduct_op_names:
                self.gamma_reduct_cells['reduct_cell_' + str(cell_idx)][op_name] = {}
                self.omega_reduct_cells['reduct_cell_' + str(cell_idx)][op_name] = {}
                self.cov_reduct_cells['reduct_cell_' + str(cell_idx)][op_name] = {}
                self.hessian_reduct_cells['reduct_cell_' + str(cell_idx)][op_name] = {}
                for switch_name in self.switch_names:
                    self.gamma_reduct_cells['reduct_cell_' + str(cell_idx)][op_name][switch_name] = torch.rand(1).to(
                        self.device)
                    self.omega_reduct_cells['reduct_cell_' + str(cell_idx)][op_name][switch_name] = torch.rand(1).to(
                        self.device)
                    self.cov_reduct_cells['reduct_cell_' + str(cell_idx)][op_name][switch_name] = torch.rand(1).to(
                        self.device)
                    self.hessian_reduct_cells['reduct_cell_' + str(cell_idx)][op_name][switch_name] = torch.rand(1).to(
                        self.device)

    def tail_pass(self, target):
        output = self.model.data['output']
        row, column = output.size()
        pre_hessian = torch.zeros(row, column).to(self.device)
        for i in range(row):
            index_temp = target[i]
            temp_value = torch.exp(output[i, index_temp])
            pre_hessian[i, index_temp] = temp_value * (1 - temp_value)
        pre_hessian = self.fc_local_hession(pre_hessian, self.model.linear.weight, self.model.data['flattened'],
                                            self.model.data['flattened'].grad, 'linear')
        pre_hessian = self.tail_fc2conv(pre_hessian, self.model.data['pool'])
        pre_hessian = self.hessian_average_upsample(pre_hessian, 8, 8)
        pre_hessian = self.fc_local_hession(pre_hessian, torch.ones(1).to(self.device), self.model.data['bn'],
                                            self.model.data['relu'].grad, 'relu')
        pre_hessian = self.bn_hessian(pre_hessian, self.model.bn)

        del self.model.data
        return pre_hessian

    def sel_pass(self, pre_hessian, selection):
        hessian = {}
        if selection.reduction:
            hessian_scalar, pre_hessian_op6 = self.sel_ident_edge(pre_hessian, selection.Ident.weight,
                                                                  selection.switch_ident,
                                                                  selection.data['ident_bn'],
                                                                  selection.data['ident_bn'].grad,
                                                                  selection.Ident_bn,
                                                                  selection.data['ident'], selection.stride)
        else:
            hessian_scalar, pre_hessian_op6 = self.hessian_control_scalar(pre_hessian, selection.switch_ident,
                                                                          selection.data['ident'])
        pre_hessian_input = pre_hessian_op6.to(self.device)
        del pre_hessian_op6
        hessian['switch_ident'] = hessian_scalar
        hessian_scalar, pre_hessian_op5 = self.sel_avg_edge(pre_hessian, selection.switch_avg_pool,
                                                            selection.data['avg_pool'], 3, selection.stride,
                                                            output_size=selection.data['input'].size())
        pre_hessian_input += pre_hessian_op5.to(self.device)
        del pre_hessian_op5
        hessian['switch_avg_pool'] = hessian_scalar
        hessian_scalar, pre_hessian_op4 = self.sel_max_edge(pre_hessian, selection.switch_max_pool,
                                                            selection.data['max_pool'],
                                                            selection.pool_indices, selection.stride,
                                                            output_size=selection.data['input'].size())
        pre_hessian_input += pre_hessian_op4.to(self.device)
        del pre_hessian_op4
        hessian['switch_max_pool'] = hessian_scalar
        hessian_scalar, pre_hessian_tmp = self.hessian_control_scalar(pre_hessian, selection.switch_dil_5,
                                                                      selection.data['dil_5'])
        pre_hessian_op3 = self.sel_dilation_edge(pre_hessian_tmp, 4,
                                                 selection.Dil_5_ident.weight, selection.data['dil_5_'],
                                                 selection.data['dil_5_'].grad,
                                                 selection.Dil_5_ident.stride[0],
                                                 selection.data['dil_5_ident'],
                                                 selection.Dil_5.weight,
                                                 F.pad(selection.data['dil_5_bn'], [4] * 4),
                                                 selection.data['dil_5_padded'].grad, selection.Dil_5.stride[0],
                                                 selection.data['dil_5_'], selection.Dil_5.groups,
                                                 selection.Dil_5.dilation[0],
                                                 selection.data['dil_5_relu'].grad,
                                                 selection.Dil_5_bn, selection.data['dil_5_bn'])
        pre_hessian_input += pre_hessian_op3.to(self.device)
        del pre_hessian_op3
        hessian['switch_dil_5'] = hessian_scalar
        hessian_scalar, pre_hessian_tmp = self.hessian_control_scalar(pre_hessian, selection.switch_dil_3,
                                                                      selection.data['dil_3'])
        pre_hessian_op2 = self.sel_dilation_edge(pre_hessian_tmp, 2,
                                                 selection.Dil_3_ident.weight, selection.data['dil_3_'],
                                                 selection.data['dil_3_'].grad,
                                                 selection.Dil_3_ident.stride[0],
                                                 selection.data['dil_3_ident'],
                                                 selection.Dil_3.weight, F.pad(selection.data['dil_3_bn'], [2] * 4),
                                                 selection.data['dil_3_padded'].grad, selection.Dil_3.stride[0],
                                                 selection.data['dil_3_'], selection.Dil_3.groups,
                                                 selection.Dil_3.dilation[0],
                                                 selection.data['dil_3_relu'].grad,
                                                 selection.Dil_3_bn, selection.data['dil_3_bn'])
        pre_hessian_input += pre_hessian_op2.to(self.device)
        del pre_hessian_op2
        hessian['switch_dil_3'] = hessian_scalar
        hessian_scalar, pre_hessian_tmp = self.hessian_control_scalar(pre_hessian, selection.switch_sep_5,
                                                                      selection.data['sep_5'])
        pre_hessian_tmp = self.sel_bn_relu_conv_edge(pre_hessian_tmp, 2,
                                                     selection.Sep_5_ident_1.weight, selection.data['sep_5_1'],
                                                     selection.data['sep_5_1'].grad, selection.Sep_5_ident_1.stride[0],
                                                     selection.data['sep_5_ident_1'],
                                                     selection.Sep_5_1.weight, selection.data['sep_5_padded_1'],
                                                     selection.data['sep_5_padded_1'].grad, selection.Sep_5_1.stride[0],
                                                     selection.data['sep_5_1'],
                                                     selection.Sep_5_1.groups, selection.data['sep_5_relu_1'].grad,
                                                     selection.Sep_5_bn_1,
                                                     selection.data['sep_5_bn_1'])
        pre_hessian_op1 = self.sel_bn_relu_conv_edge(pre_hessian_tmp, 2,
                                                     selection.Sep_5_ident_0.weight, selection.data['sep_5_0'],
                                                     selection.data['sep_5_0'].grad, selection.Sep_5_ident_0.stride[0],
                                                     selection.data['sep_5_ident_0'],
                                                     selection.Sep_5_0.weight, selection.data['sep_5_padded_0'],
                                                     selection.data['sep_5_padded_0'].grad, selection.Sep_5_0.stride[0],
                                                     selection.data['sep_5_0'],
                                                     selection.Sep_5_0.groups, selection.data['sep_5_relu_0'].grad,
                                                     selection.Sep_5_bn_0,
                                                     selection.data['sep_5_bn_0'])
        pre_hessian_input += pre_hessian_op1.to(self.device)
        del pre_hessian_op1
        hessian['switch_sep_5'] = hessian_scalar
        hessian_scalar, pre_hessian_tmp = self.hessian_control_scalar(pre_hessian, selection.switch_sep_3,
                                                                      selection.data['sep_3'])
        pre_hessian_tmp = self.sel_bn_relu_conv_edge(pre_hessian_tmp, 1,
                                                     selection.Sep_3_ident_1.weight, selection.data['sep_3_1'],
                                                     selection.data['sep_3_1'].grad, selection.Sep_3_ident_1.stride[0],
                                                     selection.data['sep_5_ident_1'],
                                                     selection.Sep_3_1.weight, selection.data['sep_3_padded_1'],
                                                     selection.data['sep_3_padded_1'].grad, selection.Sep_3_1.stride[0],
                                                     selection.data['sep_3_1'],
                                                     selection.Sep_3_1.groups, selection.data['sep_3_relu_1'].grad,
                                                     selection.Sep_3_bn_1,
                                                     selection.data['sep_3_bn_1'])

        pre_hessian_op0 = self.sel_bn_relu_conv_edge(pre_hessian_tmp, 1,
                                                     selection.Sep_3_ident_0.weight, selection.data['sep_3_0'],
                                                     selection.data['sep_3_0'].grad, selection.Sep_3_ident_0.stride[0],
                                                     selection.data['sep_3_ident_0'],
                                                     selection.Sep_3_0.weight, selection.data['sep_3_padded_0'],
                                                     selection.data['sep_3_padded_0'].grad, selection.Sep_3_0.stride[0],
                                                     selection.data['sep_3_0'],
                                                     selection.Sep_3_0.groups, selection.data['sep_3_relu_0'].grad,
                                                     selection.Sep_3_bn_0,
                                                     selection.data['sep_3_bn_0'])
        pre_hessian_input += pre_hessian_op0.to(self.device)
        del pre_hessian_op0
        hessian['switch_sep_3'] = hessian_scalar

        hessian_scalar, pre_hessian_input_all = self.hessian_control_scalar(pre_hessian_input, selection.switch_all,
                                                                            selection.data['input'])
        hessian['switch_all'] = hessian_scalar
        del selection.data
        return pre_hessian_input_all, hessian

    def norm_pass(self, pre_hessian, cell):
        hessian = {}
        _, channels, _, _ = pre_hessian.size()
        pre_hessian_0, pre_hessian_1, pre_hessian_2, pre_hessian_3 = torch.split(pre_hessian, int(channels / 4),
                                                                                 dim=1)
        pre_hessian_in1_3, hessian_in1_3 = self.sel_pass(pre_hessian_3, cell.in1_3)
        hessian['in1_3'] = hessian_in1_3
        pre_hessian_sel_0_3, hessian_sel_0_3 = self.sel_pass(pre_hessian_3, cell.sel_0_3)
        hessian['sel_0_3'] = hessian_sel_0_3
        pre_hessian_0 += pre_hessian_sel_0_3
        pre_hessian_in1_0, hessian_in1_0 = self.sel_pass(pre_hessian_0, cell.in1_0)
        hessian['in1_0'] = hessian_in1_0
        pre_hessian_in0_0, hessian_in0_0 = self.sel_pass(pre_hessian_0, cell.in0_0)
        hessian['in0_0'] = hessian_in0_0
        pre_hessian_in0_1, hessian_in0_1 = self.sel_pass(pre_hessian_1, cell.in0_1)
        hessian['in0_1'] = hessian_in0_1
        pre_hessian_in1_1, hessian_in1_1 = self.sel_pass(pre_hessian_1, cell.in1_1)
        hessian['in1_1'] = hessian_in1_1
        pre_hessian_in0_2, hessian_in0_2 = self.sel_pass(pre_hessian_2, cell.in0_2)
        hessian['in0_2'] = hessian_in0_2
        pre_hessian_in1_2, hessian_in1_2 = self.sel_pass(pre_hessian_2, cell.in1_2)
        hessian['in1_2'] = hessian_in1_2

        pre_hessian_in0 = pre_hessian_in0_0 + pre_hessian_in0_1 + pre_hessian_in0_2
        pre_hessian_in1 = pre_hessian_in1_3 + pre_hessian_in1_0 + pre_hessian_in1_1 + pre_hessian_in1_2

        return pre_hessian_in0, pre_hessian_in1, hessian

    def reduct_pass(self, pre_hessian, cell):
        hessian = {}
        _, channels, _, _ = pre_hessian.size()
        pre_hessian_0, pre_hessian_1, pre_hessian_2, pre_hessian_3 = torch.split(pre_hessian, int(channels / 4),
                                                                                 dim=1)
        pre_hessian_in1_3, hessian_in1_3 = self.sel_pass(pre_hessian_3, cell.in1_3)
        hessian['in1_3'] = hessian_in1_3
        pre_hessian_sel_0_3, hessian_sel_0_3 = self.sel_pass(pre_hessian_3, cell.sel_0_3)
        hessian['sel_0_3'] = hessian_sel_0_3
        pre_hessian_in1_2, hessian_in1_2 = self.sel_pass(pre_hessian_2, cell.in1_2)
        hessian['in1_2'] = hessian_in1_2
        pre_hessian_sel_0_2, hessian_sel_0_2 = self.sel_pass(pre_hessian_2, cell.sel_0_2)
        hessian['sel_0_2'] = hessian_sel_0_2
        pre_hessian_sel_0_1, hessian_sel_0_1 = self.sel_pass(pre_hessian_1, cell.sel_0_1)
        hessian['sel_0_1'] = hessian_sel_0_1
        pre_hessian_in1_1, hessian_in1_1 = self.sel_pass(pre_hessian_1, cell.in1_1)
        hessian['in1_1'] = hessian_in1_1

        pre_hessian_0 += pre_hessian_sel_0_1 + pre_hessian_sel_0_2 + pre_hessian_sel_0_3
        pre_hessian_in1_0, hessian_in1_0 = self.sel_pass(pre_hessian_0, cell.in1_0)
        hessian['in1_0'] = hessian_in1_0
        pre_hessian_in0_0, hessian_in0_0 = self.sel_pass(pre_hessian_0, cell.in0_0)
        hessian['in0_0'] = hessian_in0_0
        pre_hessian_in0 = pre_hessian_in0_0
        pre_hessian_in1 = pre_hessian_in1_3 + pre_hessian_in1_0 + pre_hessian_in1_1 + pre_hessian_in1_2

        return pre_hessian_in0, pre_hessian_in1, hessian

    def channel_reduct_pass(self, pre_hessian, C_reduct):
        pre_hessian = self.conv_local_hession(pre_hessian, C_reduct.conv.weight, C_reduct.data['bn'],
                                              C_reduct.data['relu'].grad, C_reduct.data['conv'], 'relu',
                                              C_reduct.conv.stride[0])
        if C_reduct.size_reduct:
            pre_hessian = self.bn_hessian(pre_hessian, C_reduct.bn)
            pre_hessian = self.conv_local_hession(pre_hessian, C_reduct.size_reduct.weight, C_reduct.data['input'],
                                                  C_reduct.data['input'].grad, C_reduct.data['reducted'], 'linear',
                                                  C_reduct.size_reduct.stride[0])
        else:
            pre_hessian = self.bn_hessian(pre_hessian, C_reduct.bn)

        del C_reduct.data
        return pre_hessian

    def compute_hessian(self, target):
        with torch.no_grad():
            hessian_norm = {}
            hessian_reduct = {}
            pre_hessian = self.tail_pass(target)
            pre_hessian_in0_n5, pre_hessian_in1_n5, hessian = self.norm_pass(pre_hessian, self.model.norm_cell_5)
            hessian_norm['norm_cell_5'] = hessian
            pre_hessian_r14 = self.channel_reduct_pass(pre_hessian_in0_n5, self.model.reduct_14)
            pre_hessian_r15 = self.channel_reduct_pass(pre_hessian_in1_n5, self.model.reduct_15)
            print('norm_cell_5')
            pre_hessian_in0_n4, pre_hessian_in1_n4, hessian = self.norm_pass(pre_hessian_r14, self.model.norm_cell_4)
            hessian_norm['norm_cell_4'] = hessian
            pre_hessian_r12 = self.channel_reduct_pass(pre_hessian_in0_n4, self.model.reduct_12)
            pre_hessian_r13 = self.channel_reduct_pass(pre_hessian_in1_n4, self.model.reduct_13)
            pre_hessian_r12 += pre_hessian_r15
            print('norm_cell_4')
            pre_hessian_in0_r1, pre_hessian_in1_r1, hessian = self.reduct_pass(pre_hessian_r12,
                                                                               self.model.reduct_cell_1)
            hessian_reduct['reduct_cell_1'] = hessian
            pre_hessian_r10 = self.channel_reduct_pass(pre_hessian_in0_r1, self.model.reduct_10)
            pre_hessian_r11 = self.channel_reduct_pass(pre_hessian_in1_r1, self.model.reduct_11)
            pre_hessian_r10 += pre_hessian_r13
            print('reduct_cell_1')
            pre_hessian_in0_n3, pre_hessian_in1_n3, hessian = self.norm_pass(pre_hessian_r10, self.model.norm_cell_3)
            hessian_norm['norm_cell_3'] = hessian
            pre_hessian_r8 = self.channel_reduct_pass(pre_hessian_in0_n3, self.model.reduct_8)
            pre_hessian_r9 = self.channel_reduct_pass(pre_hessian_in1_n3, self.model.reduct_9)
            pre_hessian_r8 += pre_hessian_r11
            print('norm_cell_3')
            pre_hessian_in0_n2, pre_hessian_in1_n2, hessian = self.norm_pass(pre_hessian_r8, self.model.norm_cell_2)
            hessian_norm['norm_cell_2'] = hessian
            pre_hessian_r6 = self.channel_reduct_pass(pre_hessian_in0_n2, self.model.reduct_6)
            pre_hessian_r7 = self.channel_reduct_pass(pre_hessian_in1_n2, self.model.reduct_7)
            pre_hessian_r6 += pre_hessian_r9
            print('norm_cell_2')
            pre_hessian_in0_r0, pre_hessian_in1_r0, hessian = self.reduct_pass(pre_hessian_r6, self.model.reduct_cell_0)
            hessian_reduct['reduct_cell_0'] = hessian
            pre_hessian_r4 = self.channel_reduct_pass(pre_hessian_in0_r0, self.model.reduct_4)
            pre_hessian_r5 = self.channel_reduct_pass(pre_hessian_in1_r0, self.model.reduct_5)
            pre_hessian_r4 += pre_hessian_r7
            print('reduct_cell_0')
            pre_hessian_in0_n1, pre_hessian_in1_n1, hessian = self.norm_pass(pre_hessian_r4, self.model.norm_cell_1)
            hessian_norm['norm_cell_1'] = hessian
            print('norm_cell_1')
            pre_hessian_r2 = self.channel_reduct_pass(pre_hessian_in0_n1, self.model.reduct_2)
            pre_hessian_r2 += pre_hessian_r5
            pre_hessian_in0_n0, pre_hessian_in1_n0, hessian = self.norm_pass(pre_hessian_r2, self.model.norm_cell_0)
            hessian_norm['norm_cell_0'] = hessian
            print('norm_cell_0')

            return hessian_norm, hessian_reduct

    def update(self, target):
        with torch.no_grad():
            hessian_norm_cells, hessian_reduct_cells = self.compute_hessian(target)
            hessian_norm_cells = self.convert(hessian_norm_cells, cell_type='norm')
            hessian_reduct_cells = self.convert(hessian_reduct_cells, cell_type='reduct')
            gamma_norm_cells = self.convert(self.gamma_norm_cells, cell_type='norm')
            gamma_reduct_cells = self.convert(self.gamma_reduct_cells, cell_type='reduct')
            omega_norm_cells = self.convert(self.omega_norm_cells, cell_type='norm')
            omega_reduct_cells = self.convert(self.omega_reduct_cells, cell_type='reduct')
            for op_name in self.norm_op_names:
                for switch_name in self.switch_names:
                    switch = []
                    for cell_idx in range(6):
                        switch.append(getattr(getattr(getattr(self.model, 'norm_cell_' + str(cell_idx)), op_name),
                                              switch_name))
                    hessian = torch.stack(hessian_norm_cells[op_name][switch_name], dim=0)
                    gamma = torch.stack(gamma_norm_cells[op_name][switch_name], dim=0)
                    switch = torch.stack(switch, dim=0)
                    gamma_inverse = 1 / gamma
                    if switch_name == 'switch_all':
                        hessian = 1 / self.lambda_origin * hessian
                    else:
                        hessian = 1 / self.lambda_child * hessian
                    for i in range(len(hessian)):
                        self.hessian_norm_cells['norm_cell_' + str(i)][op_name][switch_name] = hessian[i]
                    c = gamma_inverse + hessian
                    c = torch.reciprocal(c)
                    for i in range(len(c)):
                        self.cov_norm_cells['norm_cell_' + str(i)][op_name][switch_name] = c[i]
                    omega = torch.sqrt(torch.sum(torch.abs(-c / gamma[cell_idx].pow(2) + 1 / gamma[cell_idx])))
                    omega_norm_cells[op_name][switch_name] = [omega] * 6
                    if switch_name == 'switch_all':
                        gamma = torch.norm(switch, 2) / omega
                        gamma_norm_cells[op_name][switch_name] = [gamma] * 6
                        gamma_all = gamma
                    else:
                        gamma = torch.norm(switch, 2) / omega
                        gamma = 1 / (1 / gamma + 1 / gamma_all)
                        gamma_norm_cells[op_name][switch_name] = [gamma] * 6
            for op_name in self.reduct_op_names:
                for switch_name in self.switch_names:
                    switch = []
                    for cell_idx in range(2):
                        switch.append(getattr(getattr(getattr(self.model, 'reduct_cell_' + str(cell_idx)), op_name),
                                              switch_name))
                    hessian = torch.stack(hessian_reduct_cells[op_name][switch_name], dim=0)
                    gamma = torch.stack(gamma_reduct_cells[op_name][switch_name], dim=0)
                    switch = torch.stack(switch, dim=0)
                    gamma_inverse = 1 / gamma
                    if switch_name == 'switch_all':
                        hessian = 1 / self.lambda_origin * hessian
                    else:
                        hessian = 1 / self.lambda_child * hessian
                    for i in range(len(hessian)):
                        self.hessian_reduct_cells['reduct_cell_' + str(i)][op_name][switch_name] = hessian[i]
                    c = gamma_inverse + hessian
                    c = torch.reciprocal(c)
                    for i in range(len(c)):
                        self.cov_reduct_cells['reduct_cell_' + str(i)][op_name][switch_name] = c[i]
                    omega = torch.sqrt(torch.sum(torch.abs(-c / gamma[cell_idx].pow(2) + 1 / gamma[cell_idx])))
                    omega_reduct_cells[op_name][switch_name] = [omega] * 2
                    if switch_name == 'switch_all':
                        gamma = torch.norm(switch, 2) / omega
                        gamma_reduct_cells[op_name][switch_name] = [gamma] * 2
                        gamma_all = gamma
                    else:
                        gamma = torch.norm(switch, 2) / omega
                        gamma = 1 / (1 / gamma + 1 / gamma_all)
                        gamma_reduct_cells[op_name][switch_name] = [gamma] * 2

            self.gamma_norm_cells = self.revert(gamma_norm_cells, cell_type='norm')
            self.gamma_reduct_cells = self.revert(gamma_reduct_cells, cell_type='reduct')
            self.omega_norm_cells = self.revert(omega_norm_cells, cell_type='norm')
            self.omega_reduct_cells = self.revert(omega_reduct_cells, cell_type='reduct')

            return

    def loss_cal(self, prediction, target):
        loss_prediction = F.nll_loss(prediction, target)
        loss_reg = 0
        for op_name in self.norm_op_names:
            for switch_name in self.switch_names:
                switch_value_square = 0
                for cell_idx in range(6):
                    cell = getattr(self.model, 'norm_cell_' + str(cell_idx))
                    op = getattr(cell, op_name)
                    switch = getattr(op, switch_name)
                    switch_value_square += switch.pow(2)
                if switch_name == 'switch_all':
                    loss_reg_tmp = switch_value_square * self.omega_norm_cells['norm_cell_0'][op_name][
                        switch_name] ** 2
                    loss_reg += loss_reg_tmp.sqrt() * self.lambda_origin
                else:
                    loss_reg_tmp = switch_value_square * self.omega_norm_cells['norm_cell_0'][op_name][
                        switch_name] ** 2
                    loss_reg += loss_reg_tmp.sqrt() * self.lambda_child
        for op_name in self.reduct_op_names:
            for switch_name in self.switch_names:
                switch_value_square = 0
                for cell_idx in range(2):
                    cell = getattr(self.model, 'reduct_cell_' + str(cell_idx))
                    op = getattr(cell, op_name)
                    switch = getattr(op, switch_name)
                    switch_value_square += switch.pow(2)
                if switch_name == 'switch_all':
                    loss_reg_tmp = switch_value_square * self.omega_reduct_cells['reduct_cell_0'][op_name][
                        switch_name] ** 2
                    loss_reg += loss_reg_tmp.sqrt() * self.lambda_origin
                else:
                    loss_reg_tmp = switch_value_square * self.omega_reduct_cells['reduct_cell_0'][op_name][
                        switch_name] ** 2
                    loss_reg += loss_reg_tmp.sqrt() * self.lambda_child

        loss = loss_prediction + loss_reg

        return loss, loss_prediction

    def convert(self, obj, cell_type):
        obj_alt = {}
        if cell_type == 'norm':
            num_cells = 6
        else:
            num_cells = 2
        for op_name in getattr(self, cell_type + '_op_names'):
            obj_alt[op_name] = {}
            for switch_name in self.switch_names:
                obj_alt[op_name][switch_name] = []
                for i in range(num_cells):
                    obj_alt[op_name][switch_name].append(obj[cell_type + '_cell_' + str(i)][op_name][switch_name])

        return obj_alt

    def revert(self, obj, cell_type):
        obj_alt = {}
        if cell_type == 'norm':
            num_cells = 6
        else:
            num_cells = 2
        for cell_idx in range(num_cells):
            obj_alt[cell_type + '_cell_' + str(cell_idx)] = {}
            for op_name in getattr(self, cell_type + '_op_names'):
                obj_alt[cell_type + '_cell_' + str(cell_idx)][op_name] = {}
                for switch_name in self.switch_names:
                    obj_alt[cell_type + '_cell_' + str(cell_idx)][op_name][switch_name] = obj[op_name][switch_name][
                        cell_idx]

        return obj_alt

    def extract_info(self):
        switch_norm_value = {}
        gamma_norm_value = {}
        omega_norm_value = {}
        hessian_norm_value = {}
        cov_norm_value = {}

        switch_reduct_value = {}
        gamma_reduct_value = {}
        omega_reduct_value = {}
        hessian_reduct_value = {}
        cov_reduct_value = {}
        for cell_idx in range(6):
            cell = getattr(self.model, 'norm_cell_' + str(cell_idx))
            switch_norm_value['norm_cell_' + str(cell_idx)] = {}
            gamma_norm_value['norm_cell_' + str(cell_idx)] = {}
            omega_norm_value['norm_cell_' + str(cell_idx)] = {}
            hessian_norm_value['norm_cell_' + str(cell_idx)] = {}
            cov_norm_value['norm_cell_' + str(cell_idx)] = {}
            for op_name in self.norm_op_names:
                op = getattr(cell, op_name)
                switch_norm_value['norm_cell_' + str(cell_idx)][op_name] = {}
                gamma_norm_value['norm_cell_' + str(cell_idx)][op_name] = {}
                omega_norm_value['norm_cell_' + str(cell_idx)][op_name] = {}
                hessian_norm_value['norm_cell_' + str(cell_idx)][op_name] = {}
                cov_norm_value['norm_cell_' + str(cell_idx)][op_name] = {}
                for switch_name in self.switch_names:
                    switch = getattr(op, switch_name)
                    switch_norm_value['norm_cell_' + str(cell_idx)][op_name][
                        switch_name] = switch.detach().cpu().numpy()
                    gamma_norm_value['norm_cell_' + str(cell_idx)][op_name][
                        switch_name] = self.gamma_norm_cells['norm_cell_' + str(cell_idx)][op_name][
                        switch_name].cpu().numpy()
                    omega_norm_value['norm_cell_' + str(cell_idx)][op_name][
                        switch_name] = self.omega_norm_cells['norm_cell_' + str(cell_idx)][op_name][
                        switch_name].cpu().numpy()
                    hessian_norm_value['norm_cell_' + str(cell_idx)][op_name][
                        switch_name] = self.hessian_norm_cells['norm_cell_' + str(cell_idx)][op_name][
                        switch_name].cpu().numpy()
                    cov_norm_value['norm_cell_' + str(cell_idx)][op_name][
                        switch_name] = self.cov_norm_cells['norm_cell_' + str(cell_idx)][op_name][
                        switch_name].cpu().numpy()
        for cell_idx in range(2):
            cell = getattr(self.model, 'reduct_cell_' + str(cell_idx))
            switch_reduct_value['reduct_cell_' + str(cell_idx)] = {}
            gamma_reduct_value['reduct_cell_' + str(cell_idx)] = {}
            omega_reduct_value['reduct_cell_' + str(cell_idx)] = {}
            hessian_reduct_value['reduct_cell_' + str(cell_idx)] = {}
            cov_reduct_value['reduct_cell_' + str(cell_idx)] = {}
            for op_name in self.reduct_op_names:
                op = getattr(cell, op_name)
                switch_reduct_value['reduct_cell_' + str(cell_idx)][op_name] = {}
                gamma_reduct_value['reduct_cell_' + str(cell_idx)][op_name] = {}
                omega_reduct_value['reduct_cell_' + str(cell_idx)][op_name] = {}
                hessian_reduct_value['reduct_cell_' + str(cell_idx)][op_name] = {}
                cov_reduct_value['reduct_cell_' + str(cell_idx)][op_name] = {}
                for switch_name in self.switch_names:
                    switch = getattr(op, switch_name)
                    switch_reduct_value['reduct_cell_' + str(cell_idx)][op_name][
                        switch_name] = switch.detach().cpu().numpy()
                    gamma_reduct_value['reduct_cell_' + str(cell_idx)][op_name][
                        switch_name] = self.gamma_reduct_cells['reduct_cell_' + str(cell_idx)][op_name][
                        switch_name].cpu().numpy()
                    omega_reduct_value['reduct_cell_' + str(cell_idx)][op_name][
                        switch_name] = self.omega_reduct_cells['reduct_cell_' + str(cell_idx)][op_name][
                        switch_name].cpu().numpy()
                    hessian_reduct_value['reduct_cell_' + str(cell_idx)][op_name][
                        switch_name] = self.hessian_reduct_cells['reduct_cell_' + str(cell_idx)][op_name][
                        switch_name].cpu().numpy()
                    cov_reduct_value['reduct_cell_' + str(cell_idx)][op_name][
                        switch_name] = self.cov_reduct_cells['reduct_cell_' + str(cell_idx)][op_name][
                        switch_name].cpu().numpy()

        return switch_norm_value, gamma_norm_value, omega_norm_value, hessian_norm_value, cov_norm_value, \
               switch_reduct_value, gamma_reduct_value, omega_reduct_value, hessian_reduct_value, cov_reduct_value

    def de_padding(self, tensor, padding):
        _, _, h, _ = tensor.size()
        target_size = h - padding * 2
        return tensor.narrow(2, padding, target_size).narrow(3, padding, target_size)

    def hessian_control_scalar(self, pre_hessian_next, control_scalar, output_variable):
        pre_hessian = pre_hessian_next * control_scalar.pow(2)
        hessian_current_layer = torch.sum(torch.mul((output_variable / control_scalar).pow(2), pre_hessian_next))
        return hessian_current_layer, pre_hessian

    def conv_local_hession(self, pre_hessian, filter_weight, act_before_input, grad_act, conv_output, activation_fuc,
                           stride):
        pre_hessian = self.output_conv2fc(pre_hessian)
        currentweight = self.filter_conv2fc(filter_weight)
        act_before_input = self.input_conv2fc(act_before_input, conv_output, filter_weight, stride)
        grad_act_fc = self.input_conv2fc(grad_act, conv_output, filter_weight, stride)
        pre_hessian_record = []
        hessian_next = torch.mean(pre_hessian, 0)
        if activation_fuc == 'relu':
            B = self.relu_gradient(act_before_input)
            D = self.relu_hessian(act_before_input)
        if activation_fuc == 'tanh':
            B = self.tanh_gradient(act_before_input)
            D = self.tanh_hessian(act_before_input)
        if activation_fuc == 'sigmoid':
            B = self.sigmoid_gradient(act_before_input)
            D = self.sigmoid_hessian(act_before_input)
        if activation_fuc == 'linear':
            B = self.linear_gradient(act_before_input)
            D = self.linear_hessian(act_before_input)
        Da = grad_act_fc
        B = torch.mean(B, 0)
        D = torch.mean(D, 0)
        Da = torch.mean(Da, 0)
        D_vector = torch.mul(D, Da)
        pre_hessian = torch.mul(B.pow(2), torch.matmul(currentweight.pow(2), hessian_next)) + D_vector
        for index in range(len(act_before_input)):
            pre_hessian_record.append(pre_hessian)
        pre_hessian_record = torch.stack(pre_hessian_record, 0)
        del Da, D, B

        pre_hessian = self.hessian_fc2conv(pre_hessian_record, grad_act, 1, filter_weight, conv_output)
        return pre_hessian

    def conv_local_groups_hession(self, pre_hessian_big, filter_weight_big, act_before_input_big, grad_act_big,
                                  conv_output_big, conv_groups, activation_fuc, stride):
        _, channels, _, _ = pre_hessian_big.size()
        weight_groups = torch.split(filter_weight_big, int(channels / conv_groups), dim=0)
        pre_hessian_groups = torch.split(pre_hessian_big, int(channels / conv_groups), dim=1)

        act_before_input_groups = torch.split(act_before_input_big, int(channels / conv_groups), dim=1)
        grad_act_groups = torch.split(grad_act_big, int(channels / conv_groups), dim=1)
        conv_output_groups = torch.split(conv_output_big, int(channels / conv_groups), dim=1)

        pre_hessian_record_ = []
        for i in range(conv_groups):
            filter_weight = weight_groups[i]
            pre_hessian = pre_hessian_groups[i]
            act_before_input = act_before_input_groups[i]
            grad_act = grad_act_groups[i]
            conv_output = conv_output_groups[i]
            pre_hessian = self.output_conv2fc(pre_hessian)
            currentweight = self.filter_conv2fc(filter_weight)
            act_before_input = self.input_conv2fc(act_before_input, conv_output, filter_weight, stride)
            grad_act_fc = self.input_conv2fc(grad_act, conv_output, filter_weight, stride)
            pre_hessian_record = []
            hessian_next = torch.mean(pre_hessian, 0)
            if activation_fuc == 'relu':
                B = self.relu_gradient(act_before_input)
                D = self.relu_hessian(act_before_input)
            if activation_fuc == 'tanh':
                B = self.tanh_gradient(act_before_input)
                D = self.tanh_hessian(act_before_input)
            if activation_fuc == 'sigmoid':
                B = self.sigmoid_gradient(act_before_input)
                D = self.sigmoid_hessian(act_before_input)
            if activation_fuc == 'linear':
                B = self.linear_gradient(act_before_input)
                D = self.linear_hessian(act_before_input)
            Da = grad_act_fc
            B = torch.mean(B, 0)
            D = torch.mean(D, 0)
            Da = torch.mean(Da, 0)
            D_vector = torch.mul(D, Da)
            pre_hessian = torch.mul(B.pow(2), torch.matmul(currentweight.pow(2), hessian_next)) + D_vector
            for index in range(len(act_before_input)):
                pre_hessian_record.append(pre_hessian)
            pre_hessian_record = torch.stack(pre_hessian_record, 0)
            del Da, D, B
            pre_hessian = self.hessian_fc2conv(pre_hessian_record, grad_act, 1, filter_weight, conv_output)
            pre_hessian_record_.append(pre_hessian)
        pre_hessian = torch.cat(pre_hessian_record_, dim=1)
        return pre_hessian

    def conv_local_dilation_hession(self, pre_hessian, filter_weight, act_before_input, grad_act, conv_output,
                                    activation_fuc,
                                    stride, dilation):
        out_channel, in_channel, kw, kh = filter_weight.size()
        kw_equal = kw + (dilation - 1) * (kw - 1)
        filter_equal = torch.zeros(out_channel, in_channel, kw_equal, kw_equal).to(self.device)
        index_list = []
        for i in range(kw):
            index_list.append(dilation * i)
        for i in range(len(index_list)):
            for j in range(len(index_list)):
                filter_equal[out_channel - 1, in_channel - 1, index_list[i], index_list[j]] = filter_weight[
                    out_channel - 1, in_channel - 1, i, j]
        pre_hessian = self.output_conv2fc(pre_hessian)
        currentweight = self.filter_conv2fc(filter_equal)
        act_before_input = self.input_conv2fc(act_before_input, conv_output, filter_equal, stride)
        grad_act_fc = self.input_conv2fc(grad_act, conv_output, filter_equal, stride)
        pre_hessian_record = []
        hessian_next = torch.mean(pre_hessian, 0)
        if activation_fuc == 'relu':
            B = self.relu_gradient(act_before_input)
            D = self.relu_hessian(act_before_input)
        if activation_fuc == 'tanh':
            B = self.tanh_gradient(act_before_input)
            D = self.tanh_hessian(act_before_input)
        if activation_fuc == 'sigmoid':
            B = self.sigmoid_gradient(act_before_input)
            D = self.sigmoid_hessian(act_before_input)
        if activation_fuc == 'linear':
            B = self.linear_gradient(act_before_input)
            D = self.linear_hessian(act_before_input)
        Da = grad_act_fc
        B = torch.mean(B, 0)
        D = torch.mean(D, 0)
        Da = torch.mean(Da, 0)
        D_vector = torch.mul(D, Da)
        pre_hessian = torch.mul(B.pow(2), torch.matmul(currentweight.pow(2), hessian_next)) + D_vector
        for index in range(len(act_before_input)):
            pre_hessian_record.append(pre_hessian)
        pre_hessian_record = torch.stack(pre_hessian_record, 0)
        del Da, D, B
        pre_hessian = self.hessian_fc2conv(pre_hessian_record, grad_act, 1, filter_equal, conv_output)
        return pre_hessian

    def output_conv2fc(self, output_conv):
        batch_size, output_channels, w_out, h_out = output_conv.size()
        fc_out = output_conv.contiguous().view((batch_size * w_out * h_out, output_channels))
        return fc_out

    def input_conv2fc(self, input_conv, output_conv, filters, stride):
        _, _, in_w, in_h = input_conv.size()
        batch_size, output_channels, w_out, h_out = output_conv.size()
        output_channels, input_channels, kw, kh = filters.size()
        fc_out = torch.zeros((batch_size * w_out * h_out, kw * kh * input_channels)).to(self.device)
        for k in range(batch_size):
            current_input = input_conv[k, :, :, :]
            index_temp = k * h_out * w_out
            count_index = 0
            for i in range(0, in_h - kh + 1, stride):
                for j in range(0, in_w - kw + 1, stride):
                    patch = current_input[:, i:i + kh, j:j + kw].clone().view(-1)
                    fc_out[index_temp + count_index] = patch
                    count_index += 1
        return fc_out

    def hessian_fc2conv(self, pre_act_hession_fc, conv_input, stride, filters, conv_output):
        pre_act_hession_conv = torch.zeros(conv_input.size()).to(self.device)
        _, in_channels, W_in, H_in = conv_input.size()
        out_channels, in_channels, kw, kh = filters.size()
        batch_size, out_channels, W_out, H_out = conv_output.size()
        for i in range(batch_size):
            for m in range(0, H_out):
                for k in range(0, W_out):
                    current_hession = pre_act_hession_fc[i * W_out * H_out + m * H_out + k].view(in_channels, kw, kh)
                    pre_act_hession_conv[i, :, :, :][:, m * stride:m * stride + kh,
                    k * stride:k * stride + kw] += current_hession
        return pre_act_hession_conv

    def tail_fc2conv(self, pre_hessian, avg_pool_output):
        batch_size, channels, w, h = avg_pool_output.size()
        pre_hessian_next = pre_hessian.view(batch_size, channels, w, h)

        return pre_hessian_next

    def fc_local_hession(self, pre_hessian_next, currentweight, act_before_input, grad_act, activation_fuc):
        pre_hessian_record = []
        hessian_next = torch.mean(pre_hessian_next, 0)
        if activation_fuc == 'relu':
            B = self.relu_gradient(act_before_input)
            D = self.relu_hessian(act_before_input)
        if activation_fuc == 'tanh':
            B = self.tanh_gradient(act_before_input)
            D = self.tanh_hessian(act_before_input)
        if activation_fuc == 'sigmoid':
            B = self.sigmoid_gradient(act_before_input)
            D = self.sigmoid_hessian(act_before_input)
        if activation_fuc == 'linear':
            B = self.linear_gradient(act_before_input)
            D = self.linear_hessian(act_before_input)
        Da = grad_act
        B = torch.mean(B, 0)
        D = torch.mean(D, 0)
        Da = torch.mean(Da, 0)
        D_vector = torch.mul(D, Da)
        if len(currentweight.size()) == 1:
            pre_hessian = torch.mul(B.pow(2), hessian_next) + D_vector
        else:
            pre_hessian = torch.mul(B.pow(2), torch.matmul(currentweight.pow(2).t(), hessian_next)) + D_vector
        for index in range(len(act_before_input)):
            pre_hessian_record.append(pre_hessian)
        pre_hessian_record = torch.stack(pre_hessian_record, 0)
        del Da, D, B
        return pre_hessian_record

    def bn_hessian(self, pre_hessian_next, op):
        pre_hessian_next *= op.weight.pow(2).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        pre_hessian_next /= op.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return pre_hessian_next

    def filter_conv2fc(self, filters):
        return filters.view(filters.size(0), -1).t()

    def sel_bn_relu_conv_edge(self, pre_hessian, padding_size, filter_2, conv2_input, conv2_input_grad, conv2_stride,
                              conv2_output,
                              filter_1, conv1_input, conv1_input_grad, conv1_stride, conv1_output, conv1_groups,
                              relu_out_grad, bn_op, bn_out):
        pre_hessian = self.conv_local_hession(pre_hessian, filter_2, conv2_input, conv2_input_grad, conv2_output,
                                              'linear', conv2_stride)
        pre_hessian = self.conv_local_groups_hession(pre_hessian, filter_1, conv1_input, conv1_input_grad, conv1_output,
                                                     conv1_groups, 'linear', conv1_stride)
        pre_hessian = self.de_padding(pre_hessian, padding_size)
        pre_hessian = self.fc_local_hession(pre_hessian, torch.ones(1), bn_out, relu_out_grad, 'relu')
        pre_hessian = self.bn_hessian(pre_hessian, bn_op)
        return pre_hessian

    def sel_dilation_edge(self, pre_hessian, padding_size, filter_2, conv2_input, conv2_input_grad, conv2_stride,
                          conv2_output,
                          filter_1, conv1_input, conv1_input_grad, conv1_stride, conv1_output, conv1_groups,
                          dilation, relu_out_grad, bn_op, bn_out):
        pre_hessian = self.conv_local_hession(pre_hessian, filter_2, conv2_input, conv2_input_grad, conv2_output,
                                              'linear', conv2_stride)
        pre_hessian = self.conv_local_dilation_hession(pre_hessian, filter_1, conv1_input, conv1_input_grad,
                                                       conv1_output, 'linear', conv1_stride, dilation)
        pre_hessian = self.de_padding(pre_hessian, padding_size)
        pre_hessian = self.fc_local_hession(pre_hessian, torch.ones(1), bn_out, relu_out_grad, 'relu')
        pre_hessian = self.bn_hessian(pre_hessian, bn_op)
        return pre_hessian

    def sel_ident_edge(self, pre_hessian, conv_filter, switch, conv_input, conv_input_grad, bn_op, edge_output,
                       stride):
        hessian_scalar, pre_hessian = self.hessian_control_scalar(pre_hessian, switch, edge_output)
        pre_hessian = self.conv_local_hession(pre_hessian, conv_filter,
                                              conv_input, conv_input_grad, edge_output, 'linear', stride)
        pre_hessian = self.bn_hessian(pre_hessian, bn_op)
        return hessian_scalar, pre_hessian

    def sel_avg_edge(self, pre_hessian, switch, edge_output, pooling_size, stride, output_size):
        hessian_scalar, pre_hessian = self.hessian_control_scalar(pre_hessian, switch, edge_output)
        pre_hessian = self.hessian_average_upsample(pre_hessian, pooling_size, stride)

        size_diff = output_size[-1] - pre_hessian.size(-1)
        if size_diff:
            pre_hessian = F.pad(pre_hessian, [size_diff, 0, size_diff, 0])
        return hessian_scalar, pre_hessian

    def sel_max_edge(self, pre_hessian, switch, edge_output, maxpool_indices, stride, output_size=None):
        hessian_scalar, pre_hessian = self.hessian_control_scalar(pre_hessian, switch, edge_output)
        pre_hessian = self.hessian_max_upsample(pre_hessian, maxpool_indices, 3, stride, output_size=output_size)
        return hessian_scalar, pre_hessian

    def hessian_max_upsample(self, pre_hessian_next, pool_index, kernel_size, stride, output_size=None):
        pre_hessian = F.max_unpool2d(pre_hessian_next, pool_index, kernel_size, stride, 1, output_size=output_size)
        return pre_hessian

    def hessian_average_upsample(self, pre_hessian_next, kernel_size, stride):
        if kernel_size == stride:
            operator = 0.5 * torch.ones((kernel_size, kernel_size)).to(self.device)
            pre_hessian = np.kron(pre_hessian_next.data.cpu().numpy(), operator.data.cpu().numpy())
            pre_hessian = torch.from_numpy(pre_hessian).to(self.device)
        else:
            operator = 1 / kernel_size ** 2 * torch.ones((kernel_size, kernel_size, 1, 1)).to(self.device)
            batchsize, out_channel, W_out, H_out = pre_hessian_next.size()
            W_in = (W_out - 1) * stride + kernel_size
            H_in = (H_out - 1) * stride + kernel_size
            pre_hessian = torch.zeros(batchsize, out_channel, W_in, H_in)
            for i in range(W_out):
                for j in range(H_out):
                    tmp = torch.from_numpy(
                        np.kron(pre_hessian_next[:, :, i, j].data.cpu().numpy(), operator.data.cpu().numpy()))
                    tmp = tmp.transpose(0, 2).transpose(1, 3)
                    pre_hessian[:, :, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size] += tmp
        return pre_hessian

    def relu_gradient(self, x):
        h = (torch.abs(torch.sign(x)) + torch.sign(x)) / 2
        return h

    def relu_hessian(self, x):
        h = x * 0
        return h

    def tanh_gradient(self, x):
        h = 1 - torch.tanh(x) * torch.tanh(x)
        return h

    def tanh_hessian(self, x):
        h = 1 - torch.tanh(x) * torch.tanh(x)
        h = -2 * h * torch.tanh(x)
        return h

    def sigmoid_gradient(self, x):
        h = torch.mul(torch.sigmoid(x), 1 - torch.sigmoid(x))
        return h

    def sigmoid_hessian(self, x):
        h = self.sigmoid_gradient(x) - 2 * torch.mul(self.sigmoid_gradient(x), torch.sigmoid(x))
        return h

    def linear_gradient(self, x):
        h = torch.ones(x.size()).to(self.device)
        return h

    def linear_hessian(self, x):
        h = x * 0
        return h

    def max_pool(self, x, ksize, stride):
        maxPool_layer = torch.nn.MaxPool2d((ksize, ksize), stride=(stride, stride), return_indices=True)
        output, maxPool_index = maxPool_layer(x)
        ones = torch.ones(output.size()).cpu()
        unPool_layer = torch.nn.MaxUnpool2d((ksize, ksize), stride=(stride, stride))
        mask = unPool_layer(ones, maxPool_index)
        return mask
