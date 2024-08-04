'''
Function:
  Down-Up-CNN training

Main functions:
  * pre_train_Q(): Pre-training of QT net (output qt depth map)
  * pre_train_BD(): Pre-training of MTT net (output mtt depth map and direction map)
  * train_QBD(): Joint training of QT net and MTT net

Note:
  * More training information can be found in the paper.
Author: Aolin Feng
'''

import argparse
import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from torch.utils.data import DataLoader, Dataset, TensorDataset
import itertools
from einops import rearrange
from Metrics import Load_Pre_VP_Dataset, adjust_learning_rate, validation_QBD, pre_validation, load_pretrain_model

L1_Loss = nn.L1Loss()
L2_Loss = nn.MSELoss()
Cross_Entropy = nn.CrossEntropyLoss()
base_loss = L1_Loss

# Using weight matrix is trying to add the weight of non-zero value of the batch in the loss function,
# which is decided by the ratio of 0 and 1 in the training dataset.
luma_weight_mat = 0.5 * np.array([[1.0, 0.73, 0.15], [2.43, 0.35, 0.10], [0.96, 0.23, 0.07], [0.59, 0.16, 0.05]])
chroma_weight_mat = 0.5 * np.array([[17.83, 0.49, 0.11], [1.20, 0.25, 0.07], [0.58, 0.17, 0.05], [0.38, 0.12, 0.04]])


class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[0.2, 0.3, 0.5], gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1)  # 对模型裸输出做softmax再取log, shape=(bs, 3)

        logpt = torch.gather(log_softmax, dim=1, index=target.reshape(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.reshape(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  # 对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


def loss_func_MSBD(bd_out_batch0, bd_out_batch1, bd_out_batch2, bt_label_batch, dire_label_batch_reg, isLuma):
    if isLuma:
        weight_mat = luma_weight_mat
    else:
        weight_mat = chroma_weight_mat
    # dire_label_batch_reg's possible values only indlude 0 and 1
    weight_d0 = dire_label_batch_reg[:, 0:1, :, :] * dire_label_batch_reg[:, 0:1, :, :] + weight_mat[int((args.qp - 22) / 5)][0]
    weight_d1 = dire_label_batch_reg[:, 1:2, :, :] * dire_label_batch_reg[:, 1:2, :, :] + weight_mat[int((args.qp - 22) / 5)][1]
    weight_d2 = dire_label_batch_reg[:, 2:3, :, :] * dire_label_batch_reg[:, 2:3, :, :] + weight_mat[int((args.qp - 22) / 5)][2]
    if args.qp == 22:
        weight_d0 = 1.0
    return args.lambb0 * base_loss(bd_out_batch0[:, 0:1, :, :], bt_label_batch[:, 0:1, :, :]) + args.lambb1 * base_loss(bd_out_batch1[:, 0:1, :, :], bt_label_batch[:, 1:2, :, :]) + args.lambb2 * base_loss(bd_out_batch2[:, 0:1, :, :], bt_label_batch[:, 2:3, :, :]) + args.lambd0 * base_loss(
        weight_d0 * bd_out_batch0[:, 1:2, :, :], weight_d0 * dire_label_batch_reg[:, 0:1, :, :]) + args.lambd1 * base_loss(weight_d1 * bd_out_batch1[:, 1:2, :, :], weight_d1 * dire_label_batch_reg[:, 1:2, :, :]) + args.lambd2 * base_loss(weight_d2 * bd_out_batch2[:, 1:2, :, :],
                                                                                                                                                                                                                                              weight_d2 * dire_label_batch_reg[:, 2:3, :, :]) + args.lambresb0 * base_loss(
        weight_d0 * bd_out_batch0[:, 0:1, :, :], weight_d0 * bt_label_batch[:, 0:1, :, :]) + args.lambresb1 * base_loss(weight_d1 * (bd_out_batch1[:, 0:1, :, :] - bd_out_batch0[:, 0:1, :, :]), weight_d1 * (bt_label_batch[:, 1:2, :, :] - bt_label_batch[:, 0:1, :, :])) + args.lambresb2 * base_loss(
        weight_d2 * (bd_out_batch2[:, 0:1, :, :] - bd_out_batch1[:, 0:1, :, :]), weight_d2 * (bt_label_batch[:, 2:3, :, :] - bt_label_batch[:, 1:2, :, :]))


def loss_func_MSBD_cls(b_out_batch0, b_out_batch1, b_out_batch2, d_out_batch0, d_out_batch1, d_out_batch2, bt_label_batch, dire_label_batch_reg, isLuma, qt_out_batch=None, qt_label_batch=None):
    loss = 0

    if args.focal_loss:
        # depth model
        weight_list_b0 = [1 / max((torch.sum(torch.round(bt_label_batch[:,0]) == i).item() / float(bt_label_batch[:,0].numel())), args.min_ratio) for i in range(3)]
        sum_weight_list_b0 = sum(weight_list_b0)
        weight_list_b0 = [ele / sum_weight_list_b0 for ele in weight_list_b0]
        weight_list_b1 = [1 / max((torch.sum(torch.round((bt_label_batch[:,1] - bt_label_batch[:,0])) == i).item() / float(bt_label_batch[:,1].numel())), args.min_ratio) for i in range(3)]
        sum_weight_list_b1 = sum(weight_list_b1)
        weight_list_b1 = [ele / sum_weight_list_b1 for ele in weight_list_b1]
        weight_list_b2 = [1 / max((torch.sum(torch.round((bt_label_batch[:,2] - bt_label_batch[:,1])) == i).item() / float(bt_label_batch[:,2].numel())), args.min_ratio) for i in range(3)]
        sum_weight_list_b2 = sum(weight_list_b2)
        weight_list_b2 = [ele / sum_weight_list_b2 for ele in weight_list_b2]
        focal_loss_b0 = MultiClassFocalLossWithAlpha(alpha=weight_list_b0, gamma=args.focal_gamma)
        focal_loss_b1 = MultiClassFocalLossWithAlpha(alpha=weight_list_b1, gamma=args.focal_gamma)
        focal_loss_b2 = MultiClassFocalLossWithAlpha(alpha=weight_list_b2, gamma=args.focal_gamma)
        loss += focal_loss_b0(b_out_batch0.reshape(-1, 3), bt_label_batch[:,0].long().reshape(-1)) + \
                focal_loss_b1(b_out_batch1.reshape(-1, 3), (bt_label_batch[:,1] - bt_label_batch[:,0]).long().reshape(-1)) + \
                focal_loss_b2(b_out_batch2.reshape(-1, 3), (bt_label_batch[:,2] - bt_label_batch[:,1]).long().reshape(-1))
        # direction model
        # depth model
        weight_list_d0 = [1 / max((torch.sum(torch.round((dire_label_batch_reg[:,0] + 1)) == i).item() / float(bt_label_batch[:,0].numel())), args.min_ratio) for i in range(3)]
        weight_list_d1 = [1 / max((torch.sum(torch.round((dire_label_batch_reg[:,1] + 1)) == i).item() / float(bt_label_batch[:,1].numel())), args.min_ratio) for i in range(3)]
        weight_list_d2 = [1 / max((torch.sum(torch.round((dire_label_batch_reg[:,2] + 1)) == i).item() / float(bt_label_batch[:,2].numel())), args.min_ratio) for i in range(3)]
        focal_loss_d0 = MultiClassFocalLossWithAlpha(alpha=weight_list_d0, gamma=args.focal_gamma)
        focal_loss_d1 = MultiClassFocalLossWithAlpha(alpha=weight_list_d1, gamma=args.focal_gamma)
        focal_loss_d2 = MultiClassFocalLossWithAlpha(alpha=weight_list_d2, gamma=args.focal_gamma)
        loss += focal_loss_d0(d_out_batch0.reshape(-1, 3), (dire_label_batch_reg[:,0] + 1).long().reshape(-1)) + \
                focal_loss_d1(d_out_batch1.reshape(-1, 3), (dire_label_batch_reg[:,1] + 1).long().reshape(-1)) + \
                focal_loss_d2(d_out_batch2.reshape(-1, 3), (dire_label_batch_reg[:,2] + 1).long().reshape(-1))
    else:
        # depth loss
        if args.depth_label and qt_out_batch is not None:
            pred_d0 = F.interpolate(torch.argmax(qt_out_batch, dim=-1).unsqueeze(1).float(), scale_factor=2, mode='nearest').squeeze(1)
            pred_d1 = torch.argmax(b_out_batch0, dim=-1) + pred_d0
            pred_d2 = torch.argmax(b_out_batch1, dim=-1) + pred_d1
            bt_accu_d0 = bt_label_batch[:,0] + F.interpolate(qt_label_batch, scale_factor=2).squeeze(1)
            bt_accu_d1 = bt_accu_d0 + bt_label_batch[:,1]
            bt_accu_d2 = bt_accu_d1 + bt_label_batch[:,2]
            bt_label_d0 = (bt_accu_d0 - pred_d0).clip(max=2, min=0)
            bt_label_d1 = (bt_accu_d1 - pred_d1).clip(max=2, min=0)
            bt_label_d2 = (bt_accu_d2 - pred_d2).clip(max=2, min=0)
            loss += F.cross_entropy(b_out_batch0.reshape(-1, 3), bt_label_d0.long().reshape(-1)) * 2 + \
                    F.cross_entropy(b_out_batch1.reshape(-1, 3), bt_label_d1.long().reshape(-1)) + \
                    F.cross_entropy(b_out_batch2.reshape(-1, 3), bt_label_d2.long().reshape(-1))
        else:
            loss += F.cross_entropy(b_out_batch0.reshape(-1, 3), bt_label_batch[:,0].long().reshape(-1)) + \
                F.cross_entropy(b_out_batch1.reshape(-1, 3), (bt_label_batch[:,1] - bt_label_batch[:,0]).long().reshape(-1)) + \
                F.cross_entropy(b_out_batch2.reshape(-1, 3), (bt_label_batch[:,2] - bt_label_batch[:,1]).long().reshape(-1)) * 2

        # -1,0,1 direction loss
        loss += F.cross_entropy(d_out_batch0.reshape(-1, 3), (dire_label_batch_reg[:,0] + 1).long().reshape(-1)) + \
                F.cross_entropy(d_out_batch1.reshape(-1, 3), (dire_label_batch_reg[:,1] + 1).long().reshape(-1))  + \
                F.cross_entropy(d_out_batch2.reshape(-1, 3), (dire_label_batch_reg[:,2] + 1).long().reshape(-1))

    return loss


def loss_func_QBD(qt_out_batch, bd_out_batch0, bd_out_batch1, bd_out_batch2, qt_label_batch, bt_label_batch, dire_label_batch_reg, isLuma):
    if isLuma:
        weight_mat = luma_weight_mat
    else:
        weight_mat = chroma_weight_mat
    weight_d0 = dire_label_batch_reg[:, 0:1, :, :] * dire_label_batch_reg[:, 0:1, :, :] + weight_mat[int((args.qp - 22) / 5)][0]
    weight_d1 = dire_label_batch_reg[:, 1:2, :, :] * dire_label_batch_reg[:, 1:2, :, :] + weight_mat[int((args.qp - 22) / 5)][1]
    weight_d2 = dire_label_batch_reg[:, 2:3, :, :] * dire_label_batch_reg[:, 2:3, :, :] + weight_mat[int((args.qp - 22) / 5)][2]
    if args.qp == 22:
        weight_d0 = 1.0
    return args.lambq * base_loss(qt_out_batch, qt_label_batch) + args.lambb0 * base_loss(bd_out_batch0[:, 0:1, :, :], bt_label_batch[:, 0:1, :, :]) + args.lambb1 * base_loss(bd_out_batch1[:, 0:1, :, :], bt_label_batch[:, 1:2, :, :]) + args.lambb2 * base_loss(bd_out_batch2[:, 0:1, :, :],
                                                                                                                                                                                                                                                                    bt_label_batch[:, 2:3, :, :]) + args.lambd0 * base_loss(
        weight_d0 * bd_out_batch0[:, 1:2, :, :], weight_d0 * dire_label_batch_reg[:, 0:1, :, :]) + args.lambd1 * base_loss(weight_d1 * bd_out_batch1[:, 1:2, :, :], weight_d1 * dire_label_batch_reg[:, 1:2, :, :]) + args.lambd2 * base_loss(weight_d2 * bd_out_batch2[:, 1:2, :, :],
                                                                                                                                                                                                                                              weight_d2 * dire_label_batch_reg[:, 2:3, :, :]) + args.lambresb0 * base_loss(
        weight_d0 * bd_out_batch0[:, 0:1, :, :], weight_d0 * bt_label_batch[:, 0:1, :, :]) + args.lambresb1 * base_loss(weight_d1 * (bd_out_batch1[:, 0:1, :, :] - bd_out_batch0[:, 0:1, :, :]), weight_d1 * (bt_label_batch[:, 1:2, :, :] - bt_label_batch[:, 0:1, :, :])) + args.lambresb2 * base_loss(
        weight_d2 * (bd_out_batch2[:, 0:1, :, :] - bd_out_batch1[:, 0:1, :, :]), weight_d2 * (bt_label_batch[:, 2:3, :, :] - bt_label_batch[:, 1:2, :, :]))


def loss_func_D(dire_out_batch, dire_label_batch):  # b*9*16*16, b*3*16*16 unused
    loss = 0
    dire_out_batch = dire_out_batch.permute((0, 2, 3, 1))
    vec_dire_out_batch = dire_out_batch.reshape((-1, 9))
    for i in range(3):
        vec_dire_out_batch_i = vec_dire_out_batch[:, i * 3:(i + 1) * 3]
        vec_dire_label_batch_i = dire_label_batch[:, i, :, :].reshape(-1)
        loss += Cross_Entropy(vec_dire_out_batch_i, vec_dire_label_batch_i)
    return loss


def paras_print(args):
    print('************************** Parameters *************************')
    print('--jobID', args.jobID)
    print('--inputDir', args.inputDir)
    print('--outDir', args.outDir)
    print('--lr', args.lr)
    print('--dr', args.dr)
    print('--qp', args.qp)
    print('--epoch', args.epoch)
    print('--batchSize', args.batchSize)
    print('--lamb0', args.lamb0)
    print('--lamb1', args.lamb1)
    print('--lamb2', args.lamb2)
    print('--model_type', args.model_type)
    print("--classification", args.classification)
    print("--focal_loss", args.focal_loss)
    if args.focal_loss:
        print("--min_ratio", args.min_ratio)
        print("--focal_gamma", args.focal_gamma)
    print('************************** Parameters *************************')


def pre_train_Q(args):
    if args.isLuma:
        Net = model.Luma_Q_Net(classification=args.classification)
        net_Q_path = "/code/Debug/Models/" + "Pre_Luma_Q_QP22.pkl"
        # /ghome/fengal/VVC_Fast_Partition_DP/Models/
        comp = "Luma"
    else:
        Net = model.Chroma_Q_Net()
        net_Q_path = "/code/Debug/Models/" + "Pre_Chroma_Q_QP22.pkl"
        comp = "Chroma"

    # print("Pre Q net path:", net_Q_path)
    # Net = load_pretrain_model(Net, net_Q_path)
    Net = nn.DataParallel(Net).cuda()

    out_dir = os.path.join(args.outDir, args.jobID)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log_dir = os.path.join(out_dir, 'loss.txt')
    with open(log_dir, 'a') as f:
        s = "epoch_loss, epoch_loss, epoch_L1_loss, val_L1_loss, val_accu, test_L1_loss, test_accu\n"
        f.write(s)
    # train_loader = Load_Pre_VP_CTU_Dataset(args.inputDir, args.inputDir1, QP=args.qp, batchSize=args.batchSize, datasetID=0, PredID=0, isLuma=args.isLuma)
    train_loader = Load_Pre_VP_Dataset(args.inputDir, QP=args.qp, batchSize=args.batchSize, datasetID=0, PredID=0, isLuma=args.isLuma, num_workers=args.train_num_workers)
    val_loader = Load_Pre_VP_Dataset(args.inputDir, QP=args.qp, batchSize=args.batchSize, datasetID=1, PredID=0, isLuma=args.isLuma, num_workers=args.val_num_workers)
    test_loader = Load_Pre_VP_Dataset(args.inputDir, QP=args.qp, batchSize=args.batchSize, datasetID=2, PredID=0, isLuma=args.isLuma, num_workers=args.test_num_workers)
    optimizer = torch.optim.Adam(Net.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    print('Start Training ...')
    print("L1 loss", "Accuracy")

    for epoch in range(args.epoch):
        adjust_learning_rate(args.lr, optimizer, epoch, args.dr)
        loss_list = []
        L1_loss_list = []
        for step, data in enumerate(train_loader):
            input_batch, qt_label_batch = data
            input_batch = input_batch.cuda()
            qt_label_batch = qt_label_batch.cuda()

            # torch.set_grad_enabled(True)
            optimizer.zero_grad()
            qt_out_batch = Net(input_batch)

            # if step > 500 and not args.classification:
            #     qt_out_batch = torch.clip(qt_out_batch, max=4)

            if args.classification:
                if args.focal_loss:
                    weight_list = [1 / max((torch.sum(torch.round(qt_label_batch) == i).item() / float(qt_label_batch.numel())), args.min_ratio) for i in range(5)]
                    focal_loss = MultiClassFocalLossWithAlpha(alpha=weight_list, gamma=args.focal_gamma)
                    loss = focal_loss(qt_out_batch.reshape(-1, 5), qt_label_batch.reshape(-1).long())
                else:
                    loss = F.cross_entropy(qt_out_batch.reshape(-1, 5), qt_label_batch.reshape(-1).long())

                # if math.isnan(loss.data):
                #     print("loss of step %d is nan." % step)
                #     continue

                loss.backward()
                # nn.utils.clip_grad_norm_(Net.parameters(), max_norm=20, norm_type=2)
                optimizer.step()
                qt_out_batch = torch.softmax(qt_out_batch, dim=-1)
                L1_loss = L1_Loss(torch.argmax(qt_out_batch, dim=-1).float(), qt_label_batch.squeeze(1).float())
            else:
                # zero_weight = 1 / max((torch.sum(torch.round(qt_label_batch) == 0).item() / float(qt_out_batch.numel())), 0.1)
                # non_zero_weight = 1 / max((torch.sum(torch.round(qt_label_batch) != 0).item() / float(qt_out_batch.numel())), 0.1)
                # weight_mask = (qt_label_batch == 0) * zero_weight + (qt_label_batch != 0 ) * non_zero_weight

                weight_mask = 1
                loss = L1_Loss(qt_out_batch * weight_mask, qt_label_batch * weight_mask)
                # loss = L1_Loss(qt_out_batch, qt_label_batch)

                # if math.isnan(loss.data):
                #     print("loss of step %d is nan." % step)
                #     continue

                loss.backward()
                # nn.utils.clip_grad_norm_(Net.parameters(), max_norm=20, norm_type=2)
                optimizer.step()

                # # 计算模型的最大梯度
                # max_gradient = 0.0
                # for param in Net.parameters():
                #     if param.grad is not None:
                #         param_gradient = param.grad.data.abs().max()
                #         max_gradient = max(max_gradient, param_gradient)
                # print("max_gradient: %.5e"%max_gradient)

                L1_loss = L1_Loss(qt_out_batch, qt_label_batch)

            loss_list.append(loss.item())
            L1_loss_list.append(L1_loss.item())
            if (step + 1) % 100 == 0:
                # print(zero_weight, non_zero_weight)
                # print(qt_out_batch.max().item(), qt_out_batch.min().item())
                if args.classification:
                    zero_rate = torch.sum(torch.argmax(qt_out_batch, dim=-1) == 0).item() / float(qt_label_batch.numel())
                else:
                    zero_rate = torch.sum(torch.round(qt_out_batch) == 0).item() / float(qt_label_batch.numel())
                zero_label = torch.sum(torch.round(qt_label_batch) == 0).item() / float(qt_label_batch.numel())
                print("epoch: %d step: %d [loss] %.6f [L1 loss] %.6f [Zero_pred] %.6f [Zero_label] %.6f " % (epoch, step, loss.item(), L1_loss.item(), zero_rate, zero_label))

        epoch_loss = np.mean(loss_list)
        epoch_L1_loss = np.mean(L1_loss_list)
        val_out_info_list = pre_validation(val_loader, Net, 0, args=args)  # validation set loss
        test_out_info_list = pre_validation(test_loader, Net, 0, args=args)  # test set loss

        print('***********************************************************************'
              '***********************************************************************')
        print("Epoch: %d  Loss: %.6f  L1 Loss %.6f" % (epoch, epoch_loss, epoch_L1_loss))
        print("Val: Loss: %.6f  Acc: %.6f  Zero: %.6f" % (val_out_info_list[0], val_out_info_list[1], val_out_info_list[2]))
        print("Test: Loss: %.6f  Acc: %.6f  Zero: %.6f" % (test_out_info_list[0], test_out_info_list[1], val_out_info_list[2]))
        print('***********************************************************************'
              '***********************************************************************')
        with open(log_dir, 'a') as f:
            for s in [epoch_loss, epoch_loss, epoch_L1_loss, val_out_info_list[0], val_out_info_list[1], test_out_info_list[0], test_out_info_list[1]]:
                f.write(str(s))
                f.write(',')
            f.write('\n')

        if (epoch + 1) % 10 == 0:
            torch.save(Net.state_dict(), os.path.join(out_dir, comp + "_Q_" + str(args.qp) + "_epoch_%d_%.4f_%.4f_%.4f.pkl" % (epoch, epoch_loss, val_out_info_list[0], test_out_info_list[0])))

def pre_train_BD(args):
    if args.isLuma:
        Net = model.Luma_MSBD_Net(classification=args.classification)
        net_B_path = "/code/Debug/Models/" + "Pre_Luma_B_QP22.pkl"
        # /ghome/fengal/VVC_Fast_Partition_DP/Models/
        comp = "Luma"
    else:
        Net = model.Chroma_MSBD_Net()
        net_B_path = "/code/Debug/Models/" + "Pre_Chroma_B_QP22.pkl"
        comp = "Chroma"

    # net_D_path = "/ghome/fengal/VVC_Fast_Partition_DP/Models/Luma" + "_D_QP" + str(args.qp) + ".pkl"
    # print("Pre B net path:", net_B_path)
    # Net = load_pretrain_model(Net, net_B_path)
    Net = nn.DataParallel(Net).cuda()

    out_dir = os.path.join(args.outDir, args.jobID)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log_dir = os.path.join(out_dir, 'loss.txt')
    with open(log_dir, 'a') as f:
        s = "epoch_loss, epoch_b0_L1_loss, epoch_b1_L1_loss, epoch_b2_L1_loss, val_b2_accu, test_b2_accu\n"
        f.write(s)
    # Net_QB.load_state_dict(torch.load('/ghome/fengal/VVC_Fast_Partition_DP/PreModel/luma_qp32_109.pkl'))

    train_loader = Load_Pre_VP_Dataset(args.inputDir, QP=args.qp, batchSize=args.batchSize, datasetID=0, PredID=2, isLuma=args.isLuma)
    val_loader = Load_Pre_VP_Dataset(args.inputDir, QP=args.qp, batchSize=args.batchSize, datasetID=1, PredID=2, isLuma=args.isLuma)
    test_loader = Load_Pre_VP_Dataset(args.inputDir, QP=args.qp, batchSize=args.batchSize, datasetID=2, PredID=2, isLuma=args.isLuma)
    optimizer = torch.optim.Adam(Net.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    print('Start Training ...')
    print("b0 L1 loss", "b1 L1 loss", "b2 L1 loss", "d0 L1 loss", "d1 L1 loss", "d2 L1 loss")
    for epoch in range(args.epoch):
        adjust_learning_rate(args.lr, optimizer, epoch, args.dr)
        loss_list = []
        b0_L1_loss_list, b1_L1_loss_list, b2_L1_loss_list = [], [], []
        d0_L1_loss_list, d1_L1_loss_list, d2_L1_loss_list = [], [], []
        for step, data in enumerate(train_loader):
            input_batch, qt_label_batch, bt_label_batch, dire_label_batch_reg = data
            input_batch = input_batch.cuda()
            qt_label_batch = qt_label_batch.cuda()
            bt_label_batch = bt_label_batch.cuda()
            dire_label_batch_reg = dire_label_batch_reg.cuda()

            # torch.set_grad_enabled(True)
            optimizer.zero_grad()
            if args.classification:
                b_out_batch0, b_out_batch1, b_out_batch2, d_out_batch0, d_out_batch1, d_out_batch2 = Net(input_batch, qt_label_batch)
                loss = loss_func_MSBD_cls(b_out_batch0, b_out_batch1, b_out_batch2, d_out_batch0, d_out_batch1, d_out_batch2, bt_label_batch, dire_label_batch_reg, args.isLuma)
            else:
                bd_out_batch0, bd_out_batch1, bd_out_batch2 = Net(input_batch, qt_label_batch)
                loss = loss_func_MSBD(bd_out_batch0, bd_out_batch1, bd_out_batch2, bt_label_batch, dire_label_batch_reg, args.isLuma)
            loss.backward()
            optimizer.step()

            if args.classification:
                b_0 = torch.argmax(torch.softmax(b_out_batch0, dim=-1), dim=-1)
                b_1 = torch.argmax(torch.softmax(b_out_batch1, dim=-1), dim=-1) + b_0
                b_2 = torch.argmax(torch.softmax(b_out_batch2, dim=-1), dim=-1) + b_1
                b0_L1_loss = L1_Loss(b_0.float(), bt_label_batch[:, 0, :, :].float())
                b1_L1_loss = L1_Loss(b_1.float(), bt_label_batch[:, 1, :, :].float())
                b2_L1_loss = L1_Loss(b_2.float(), bt_label_batch[:, 2, :, :].float())

                d_0 = torch.argmax(torch.softmax(d_out_batch0, dim=-1), dim=-1) - 1
                d_1 = torch.argmax(torch.softmax(d_out_batch1, dim=-1), dim=-1) - 1
                d_2 = torch.argmax(torch.softmax(d_out_batch2, dim=-1), dim=-1) - 1
                d0_L1_loss = L1_Loss(d_0.float(), dire_label_batch_reg[:, 0, :, :].float())
                d1_L1_loss = L1_Loss(d_1.float(), dire_label_batch_reg[:, 1, :, :].float())
                d2_L1_loss = L1_Loss(d_2.float(), dire_label_batch_reg[:, 2, :, :].float())
            else:
                b0_L1_loss = L1_Loss(bd_out_batch0[:, 0:1, :, :], bt_label_batch[:, 0:1, :, :])
                b1_L1_loss = L1_Loss(bd_out_batch1[:, 0:1, :, :], bt_label_batch[:, 1:2, :, :])
                b2_L1_loss = L1_Loss(bd_out_batch2[:, 0:1, :, :], bt_label_batch[:, 2:3, :, :])
                d0_L1_loss = L1_Loss(bd_out_batch0[:, 1:2, :, :], dire_label_batch_reg[:, 0:1, :, :])
                d1_L1_loss = L1_Loss(bd_out_batch1[:, 1:2, :, :], dire_label_batch_reg[:, 1:2, :, :])
                d2_L1_loss = L1_Loss(bd_out_batch2[:, 1:2, :, :], dire_label_batch_reg[:, 2:3, :, :])

            loss_list.append(loss.item())
            b0_L1_loss_list.append(b0_L1_loss.item())
            b1_L1_loss_list.append(b1_L1_loss.item())
            b2_L1_loss_list.append(b2_L1_loss.item())
            d0_L1_loss_list.append(d0_L1_loss.item())
            d1_L1_loss_list.append(d1_L1_loss.item())
            d2_L1_loss_list.append(d2_L1_loss.item())
            if step % 1000 == 0:
                if args.classification:
                    zero_rate_b, zero_rate_d = [], []
                    for b in [b_0, b_1 - b_0, b_2 - b_1]:
                        zero_rate_b.append(100 * torch.sum(b == 0).item() / float(b.numel() ))
                    for d in [d_0, d_1, d_2]:
                        zero_rate_d.append(100 * torch.sum(d == 0).item() / float(d.numel() ))
                    zero_rate_b_label, zero_rate_d_label = [], []
                    for b in [bt_label_batch[:, 0, :, :], bt_label_batch[:, 1, :, :] - bt_label_batch[:, 0, :, :], bt_label_batch[:, 2, :, :] - bt_label_batch[:, 1, :, :]]:
                        zero_rate_b_label.append(100 * torch.sum(b == 0).item() / float(b.numel() ))
                    for d in [dire_label_batch_reg[:, 0, :, :], dire_label_batch_reg[:, 1, :, :] - dire_label_batch_reg[:, 0, :, :], dire_label_batch_reg[:, 2, :, :] - dire_label_batch_reg[:, 1, :, :]]:
                        zero_rate_d_label.append(100 * torch.sum(d == 0).item() / float(d.numel() ))
                    print("epoch: %d step: %d [loss] %.6f [b] %.6f %.6f %.6f [d] %.6f %.6f %.6f, [b_zero] %.2f %.2f %.2f [b_label_zero] %.2f %.2f %.2f [d_zero] %.2f %.2f %.2f [d_label_zero] %.2f %.2f %.2f" \
                        % (epoch, step, loss.item(), b0_L1_loss.item(), b1_L1_loss.item(), b2_L1_loss.item(), d0_L1_loss.item(), d1_L1_loss.item(), d2_L1_loss.item(),
                            zero_rate_b[0], zero_rate_b[1] , zero_rate_b[2] , zero_rate_b_label[0] , zero_rate_b_label[1] , zero_rate_b_label[2] ,
                                zero_rate_d[0] , zero_rate_d[1] , zero_rate_d[2] , zero_rate_d_label[0] , zero_rate_d_label[1] , zero_rate_d_label[2] ))
                else:
                    print("epoch: %d step: %d [loss] %.6f [b] %.6f %.6f %.6f [d] %.6f %.6f %.6f" % (epoch, step, loss.item(), b0_L1_loss.item(), b1_L1_loss.item(), b2_L1_loss.item(), d0_L1_loss.item(), d1_L1_loss.item(), d2_L1_loss.item()))
                # test_out_info_list = pre_validation(test_loader, Net, 1, args.qp, args=args)

        epoch_loss = np.mean(loss_list)
        epoch_b0_L1_loss = np.mean(b0_L1_loss_list)
        epoch_b1_L1_loss = np.mean(b1_L1_loss_list)
        epoch_b2_L1_loss = np.mean(b2_L1_loss_list)
        epoch_d0_L1_loss = np.mean(d0_L1_loss_list)
        epoch_d1_L1_loss = np.mean(d1_L1_loss_list)
        epoch_d2_L1_loss = np.mean(d2_L1_loss_list)
        val_out_info_list = pre_validation(val_loader, Net, 1, args.qp, args=args)  # validation set loss
        test_out_info_list = pre_validation(test_loader, Net, 1, args.qp, args=args)  # test set loss

        print('*****************************************************************'
              '*****************************************************************')
        # print("Epoch:", epoch, " Loss:", epoch_loss)
        print("Epoch: %d  Loss: %.6f %.6f %.6f" % (epoch, epoch_loss, val_out_info_list[12], test_out_info_list[12]))
        print("Train L1: [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f" % (epoch_b0_L1_loss, epoch_b1_L1_loss, epoch_b2_L1_loss, epoch_d0_L1_loss, epoch_d1_L1_loss, epoch_d2_L1_loss))
        # print("Val Loss:", val_out_info_list[12])
        print("Val L1: [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f" % (val_out_info_list[0], val_out_info_list[1], val_out_info_list[2], val_out_info_list[3], val_out_info_list[4], val_out_info_list[5]))
        print("Val Accu: [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f" % (val_out_info_list[6], val_out_info_list[7], val_out_info_list[8], val_out_info_list[9], val_out_info_list[10], val_out_info_list[11]))
        # print("Test Loss:", test_out_info_list[12])
        print("Test L1: [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f" % (test_out_info_list[0], test_out_info_list[1], test_out_info_list[2], test_out_info_list[3], test_out_info_list[4], test_out_info_list[5]))
        print("Test Accu: [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f" % (test_out_info_list[6], test_out_info_list[7], test_out_info_list[8], test_out_info_list[9], test_out_info_list[10], test_out_info_list[11]))

        print('******************************************************************'
              '******************************************************************')
        # save training loss
        with open(log_dir, 'a') as f:
            for s in [epoch_loss, epoch_b0_L1_loss, epoch_b1_L1_loss, epoch_b2_L1_loss, epoch_d0_L1_loss, epoch_d1_L1_loss, epoch_d2_L1_loss]:
                f.write(str(s))
                f.write(',')
            f.write('\n')

        if (epoch + 1) % 10 == 0:
            torch.save(Net.state_dict(), os.path.join(out_dir, comp + "_BD_" + str(args.qp) + "_epoch_%d_%.4f_%.4f_%.4f.pkl" % (epoch, epoch_loss, val_out_info_list[12], test_out_info_list[12])))


def train_QBD(args):
    if args.isLuma:
        if 'SA' in args.model_type:
            Net_Q = model.Luma_Q_Net(classification=args.classification, c_ratio = args.C_ratio, sparse_threshold=qt_sparse_threshold)
            Net_BD = model.Luma_MSBD_Net(classification=args.classification, c_ratio = args.C_ratio, sparse_threshold=mt_sparse_threshold)
        else:
            Net_Q = model.Luma_Q_Net()
            Net_BD = model.Luma_MSBD_Net()
        comp = "Luma"
    else:
        if 'SA' in args.model_type:
            Net_Q = model.Chroma_Q_Net(classification=args.classification, c_ratio = args.C_ratio)
            Net_BD = model.Chroma_MSBD_Net(classification=args.classification, c_ratio = args.C_ratio)
        else:
            Net_Q = model.Chroma_Q_Net()
            Net_BD = model.Chroma_MSBD_Net()
        comp = "Chroma"

    # Net = nn.DataParallel(Net).cuda()
    print("Pre Q net path:", args.net_Q_path)
    print("Pre BD net path:", args.net_BD_path)
    if args.net_Q_path is not None:
        Net_Q = load_pretrain_model(Net_Q, args.net_Q_path)
    if args.net_BD_path is not None:
        Net_BD = load_pretrain_model(Net_BD, args.net_BD_path)
    Net_Q = nn.DataParallel(Net_Q).cuda()
    Net_BD = nn.DataParallel(Net_BD).cuda()
    # Net_Q = Net_Q.cuda()
    # Net_BD = Net_BD.cuda()
    out_dir = os.path.join(args.outDir, args.jobID)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log_dir = os.path.join(out_dir, 'loss.txt')
    with open(log_dir, 'a') as f:
        s = "epoch_loss, epoch_b0_L1_loss, epoch_b1_L1_loss, epoch_b2_L1_loss, val_b2_accu, test_b2_accu\n"
        f.write(s)

    train_loader = Load_Pre_VP_Dataset(args.inputDir, QP=args.qp, batchSize=args.batchSize, datasetID=0, PredID=2, isLuma=args.isLuma, num_workers=args.train_num_workers)
    val_loader = Load_Pre_VP_Dataset(args.inputDir, QP=args.qp, batchSize=args.batchSize, datasetID=1, PredID=2, isLuma=args.isLuma, num_workers=args.val_num_workers)
    test_loader = Load_Pre_VP_Dataset(args.inputDir, QP=args.qp, batchSize=args.batchSize, datasetID=2, PredID=2, isLuma=args.isLuma, num_workers=args.test_num_workers)
    optimizer = torch.optim.Adam(itertools.chain(Net_Q.parameters(), Net_BD.parameters()), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    print('Start Training ...')
    print("q L1 loss", "b0 L1 loss", "b1 L1 loss", "b2 L1 loss", "d0 L1 loss", "d1 L1 loss", "d2 L1 loss")

    # # 预训练模型预测准确率
    # test_out_info_list = validation_QBD(test_loader, Net_Q, Net_BD, args.qp, args=args)
    # print("Test Accu: [Q] %.6f [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f" % (test_out_info_list[7], test_out_info_list[8], test_out_info_list[9], test_out_info_list[10], test_out_info_list[11], test_out_info_list[12], test_out_info_list[13]))
    # if args.post_test:
    #     print("Post Accu: [Q] %.6f [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f " % (test_out_info_list[15], test_out_info_list[16], test_out_info_list[17], test_out_info_list[18], test_out_info_list[19], test_out_info_list[20], test_out_info_list[21]))

    for epoch in range(args.epoch):
        Net_Q.train()
        Net_BD.train()
        adjust_learning_rate(args.lr, optimizer, epoch, args.dr)
        loss_list = []
        q_L1_loss_list = []
        b0_L1_loss_list, b1_L1_loss_list, b2_L1_loss_list = [], [], []
        d0_L1_loss_list, d1_L1_loss_list, d2_L1_loss_list = [], [], []
        # # --------------------------- 统计zero-rate--------------------------------
        # qt_ratio, md0_ratio, md1_ratio, md2_ratio = [], [], [], []
        # # --------------------------- end ----------------------------------------
        for step, data in enumerate(train_loader):
            Net_Q.train()
            Net_BD.train()
            input_batch, qt_label_batch, bt_label_batch, dire_label_batch_reg = data
            # # -------------------------- 统计zero-rate, 确定ratio --------------------------- #
            # qt_ratio.append([torch.sum(qt_label_batch == i) / qt_label_batch.numel() for i in range(4)])
            # md0_ratio.append([torch.sum(bt_label_batch[:, 0] == i) / bt_label_batch[:,0].numel() for i in range(3)])
            # md1_ratio.append([torch.sum((bt_label_batch[:, 1] - bt_label_batch[:, 0]) == i) / bt_label_batch[:,1].numel() for i in range(3)])
            # md2_ratio.append([torch.sum((bt_label_batch[:, 2] - bt_label_batch[:, 1]) == i) / bt_label_batch[:,2].numel() for i in range(3)])
            # continue
            # # ---------------------------------end-------------------------------------------#
            input_batch = input_batch.cuda()
            qt_label_batch = qt_label_batch.cuda()
            bt_label_batch = bt_label_batch.cuda()
            dire_label_batch_reg = dire_label_batch_reg.cuda()

            # torch.set_grad_enabled(True)

            if args.classification:
                qt_out_batch, qt_token_decisions, qt_pred_score_list = Net_Q(input_batch)
                qt_out_batch_argmax = (F.gumbel_softmax(qt_out_batch, tau=0.1, hard=True, dim=-1) * torch.tensor([0,1,2,3,4]).reshape(1,1,1,-1).to(qt_out_batch.device)).sum(-1).unsqueeze(1).float()
                b_out_batch0, b_out_batch1, b_out_batch2, d_out_batch0, d_out_batch1, d_out_batch2, depth_decisions, dire_decisions, depth_pred_score_list, dire_pred_score_list = Net_BD(input_batch, qt_out_batch_argmax)
                loss = 0
                # if args.focal_loss:
                #     weight_list = [1 / max((torch.sum(torch.round(qt_label_batch) == i).item() / float(qt_label_batch.numel())), 0.25) for i in range(5)]
                #     sum_weight_list = sum(weight_list)
                #     weight_list = [ele / sum_weight_list for ele in weight_list]
                #     focal_loss = MultiClassFocalLossWithAlpha(alpha=weight_list, gamma=args.focal_gamma)
                #     loss = focal_loss(qt_out_batch.reshape(-1, 5), qt_label_batch.reshape(-1).long()) * 6
                # else:
                # weights = torch.tensor([1.0, 1.0, 1.0, 2.0, 0])
                loss = F.cross_entropy(qt_out_batch.reshape(-1, 5), qt_label_batch.reshape(-1).long())

                loss += loss_func_MSBD_cls(b_out_batch0, b_out_batch1, b_out_batch2, d_out_batch0, d_out_batch1, d_out_batch2, bt_label_batch, dire_label_batch_reg, args.isLuma, qt_out_batch=qt_out_batch, qt_label_batch=qt_label_batch)

                # ratio loss
                pred_loss, num_list = 0, 0
                for decisions in [qt_token_decisions]:
                    for pred_score in decisions:
                        pred_loss = pred_loss + ((pred_score.mean(dim=1) - qt_sparse_threshold[0][0]) ** 2).mean()
                        num_list += 1

                for decisions in [ depth_decisions, dire_decisions]:
                    for sub_decisions_id, sub_decisions in enumerate(decisions):
                        for pred_score in sub_decisions:
                            pred_loss = pred_loss + ((pred_score.mean(dim=1) - mt_sparse_threshold[sub_decisions_id][0]) ** 2).mean()
                            num_list += 1

                pred_loss = pred_loss / num_list
                loss += pred_loss

                aux_loss, num_list = 0, 0
                # pseudo loss
                if args.aux_loss:
                    # 根据块划分标签制作令牌稀疏化标签
                    for pred_score in qt_pred_score_list:
                        qt_token_label = rearrange(F.interpolate(qt_label_batch, scale_factor=int(math.sqrt(pred_score.shape[1]) // 8), mode='nearest'), 'b c h w -> b (h w) c').clip(min=0, max=1) == 0
                        aux_loss += F.cross_entropy(pred_score.reshape(-1, 2), qt_token_label.reshape(-1).long())
                        num_list += 1
                    # bt label
                    for bt_depth in range(3):
                        for pred_score in depth_pred_score_list[bt_depth]:
                            bt_token_label = rearrange(F.interpolate(dire_label_batch_reg[:,bt_depth:bt_depth+1], scale_factor=int(math.sqrt(pred_score.shape[1]) // 16), mode='nearest'), 'b c h w -> b (h w) c') == 0
                            aux_loss += F.cross_entropy(pred_score.reshape(-1, 2), bt_token_label.reshape(-1).long())
                            num_list += 1
                        for pred_score in dire_pred_score_list[bt_depth]:
                            bt_token_label = rearrange(F.interpolate(dire_label_batch_reg[:,bt_depth:bt_depth+1], scale_factor=int(math.sqrt(pred_score.shape[1]) // 16), mode='nearest'), 'b c h w -> b (h w) c') == 0
                            aux_loss += F.cross_entropy(pred_score.reshape(-1, 2), bt_token_label.reshape(-1).long())
                            num_list += 1
                    aux_loss = aux_loss / num_list
                loss += aux_loss

            else:
                qt_out_batch = Net_Q(input_batch)
                bd_out_batch0, bd_out_batch1, bd_out_batch2 = Net_BD(input_batch, qt_out_batch)
                # loss = weight_loss_func_MSB(bt_out_batch0, bt_out_batch1, bt_out_batch2, bt_label_batch, bt_label_batch1, bt_label_batch2)
                loss = loss_func_QBD(qt_out_batch, bd_out_batch0, bd_out_batch1, bd_out_batch2, qt_label_batch, bt_label_batch, dire_label_batch_reg, args.isLuma)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if args.classification:
                qt_out_batch = torch.softmax(qt_out_batch, dim=-1)
                q_L1_loss = L1_Loss(torch.argmax(qt_out_batch, dim=-1).float(), qt_label_batch.squeeze(1).float())
                if args.depth_label:
                    b_0 = torch.argmax(torch.softmax(b_out_batch0, dim=-1), dim=-1) + F.interpolate(torch.argmax(qt_out_batch, dim=-1).float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1)
                    b_1 = torch.argmax(torch.softmax(b_out_batch1, dim=-1), dim=-1) + b_0
                    b_2 = torch.argmax(torch.softmax(b_out_batch2, dim=-1), dim=-1) + b_1
                    b0_L1_loss = L1_Loss(b_0.float(), bt_label_batch[:, 0, :, :].float() + F.interpolate(qt_label_batch, scale_factor=2, mode='nearest').squeeze(1))
                    b1_L1_loss = L1_Loss(b_1.float(), bt_label_batch[:, 1, :, :].float() + F.interpolate(qt_label_batch, scale_factor=2, mode='nearest').squeeze(1))
                    b2_L1_loss = L1_Loss(b_2.float(), bt_label_batch[:, 2, :, :].float() + F.interpolate(qt_label_batch, scale_factor=2, mode='nearest').squeeze(1))
                else:
                    b_0 = torch.argmax(torch.softmax(b_out_batch0, dim=-1), dim=-1)
                    b_1 = torch.argmax(torch.softmax(b_out_batch1, dim=-1), dim=-1) + b_0
                    b_2 = torch.argmax(torch.softmax(b_out_batch2, dim=-1), dim=-1) + b_1
                    b0_L1_loss = L1_Loss(b_0.float(), bt_label_batch[:, 0, :, :].float())
                    b1_L1_loss = L1_Loss(b_1.float(), bt_label_batch[:, 1, :, :].float())
                    b2_L1_loss = L1_Loss(b_2.float(), bt_label_batch[:, 2, :, :].float())

                d_0 = torch.argmax(torch.softmax(d_out_batch0, dim=-1), dim=-1) - 1
                d_1 = torch.argmax(torch.softmax(d_out_batch1, dim=-1), dim=-1) - 1
                d_2 = torch.argmax(torch.softmax(d_out_batch2, dim=-1), dim=-1) - 1
                d0_L1_loss = L1_Loss(d_0.float(), dire_label_batch_reg[:, 0, :, :].float())
                d1_L1_loss = L1_Loss(d_1.float(), dire_label_batch_reg[:, 1, :, :].float())
                d2_L1_loss = L1_Loss(d_2.float(), dire_label_batch_reg[:, 2, :, :].float())
            else:
                q_L1_loss = L1_Loss(qt_out_batch, qt_label_batch)
                b0_L1_loss = L1_Loss(bd_out_batch0[:, 0:1, :, :], bt_label_batch[:, 0:1, :, :])
                b1_L1_loss = L1_Loss(bd_out_batch1[:, 0:1, :, :], bt_label_batch[:, 1:2, :, :])
                b2_L1_loss = L1_Loss(bd_out_batch2[:, 0:1, :, :], bt_label_batch[:, 2:3, :, :])
                d0_L1_loss = L1_Loss(bd_out_batch0[:, 1:2, :, :], dire_label_batch_reg[:, 0:1, :, :])
                d1_L1_loss = L1_Loss(bd_out_batch1[:, 1:2, :, :], dire_label_batch_reg[:, 1:2, :, :])
                d2_L1_loss = L1_Loss(bd_out_batch2[:, 1:2, :, :], dire_label_batch_reg[:, 2:3, :, :])

            loss_list.append(loss.item())
            q_L1_loss_list.append(q_L1_loss.item())
            b0_L1_loss_list.append(b0_L1_loss.item())
            b1_L1_loss_list.append(b1_L1_loss.item())
            b2_L1_loss_list.append(b2_L1_loss.item())
            d0_L1_loss_list.append(d0_L1_loss.item())
            d1_L1_loss_list.append(d1_L1_loss.item())
            d2_L1_loss_list.append(d2_L1_loss.item())


            if step % 1000 == 0:
                print("epoch: %d step: %d [loss] %.6f [q] %.6f [b] %.6f %.6f %.6f [d] %.6f %.6f %.6f" % (epoch, step, loss.item(), q_L1_loss.item(), b0_L1_loss.item(), b1_L1_loss.item(), b2_L1_loss.item(), d0_L1_loss.item(), d1_L1_loss.item(), d2_L1_loss.item()))

        epoch_loss = np.mean(loss_list)
        epoch_q_L1_loss = np.mean(q_L1_loss_list)
        epoch_b0_L1_loss = np.mean(b0_L1_loss_list)
        epoch_b1_L1_loss = np.mean(b1_L1_loss_list)
        epoch_b2_L1_loss = np.mean(b2_L1_loss_list)
        epoch_d0_L1_loss = np.mean(d0_L1_loss_list)
        epoch_d1_L1_loss = np.mean(d1_L1_loss_list)
        epoch_d2_L1_loss = np.mean(d2_L1_loss_list)

        args.post_test = False
        Net_Q.eval()
        Net_BD.eval()
        val_out_info_list = validation_QBD(val_loader, Net_Q, Net_BD, args.qp, args=args)  # validation set loss
        test_out_info_list = validation_QBD(test_loader, Net_Q, Net_BD, args.qp, args=args)  # test set loss


        print('*****************************************************************'
              '*****************************************************************')
        # print("Epoch:", epoch, " Loss:", epoch_loss)
        print("Epoch: %d  Loss: %.6f %.6f %.6f" % (epoch, epoch_loss, val_out_info_list[14], test_out_info_list[14]))
        print("Train L1: [Q] %.6f [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f" % (epoch_q_L1_loss, epoch_b0_L1_loss, epoch_b1_L1_loss, epoch_b2_L1_loss, epoch_d0_L1_loss, epoch_d1_L1_loss, epoch_d2_L1_loss))

        print("Val L1: [Q] %.6f [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f" % (val_out_info_list[0], val_out_info_list[1], val_out_info_list[2], val_out_info_list[3], val_out_info_list[4], val_out_info_list[5], val_out_info_list[6]))
        print("Val Accu: [Q] %.6f [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f" % (val_out_info_list[7], val_out_info_list[8], val_out_info_list[9], val_out_info_list[10], val_out_info_list[11], val_out_info_list[12], val_out_info_list[13]))

        print("Test L1: [Q] %.6f [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f" % (test_out_info_list[0], test_out_info_list[1], test_out_info_list[2], test_out_info_list[3], test_out_info_list[4], test_out_info_list[5], test_out_info_list[6]))
        print("Test Accu: [Q] %.6f [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f" % (test_out_info_list[7], test_out_info_list[8], test_out_info_list[9], test_out_info_list[10], test_out_info_list[11], test_out_info_list[12], test_out_info_list[13]))
        if args.post_test:
            print("Post Accu: [Q] %.6f [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f " % (test_out_info_list[15], test_out_info_list[16], test_out_info_list[17], test_out_info_list[18], test_out_info_list[19], test_out_info_list[20], test_out_info_list[21]))

        print('******************************************************************'
              '******************************************************************')
        # save training loss
        with open(log_dir, 'a') as f:
            for s in [epoch_loss, epoch_q_L1_loss, epoch_b0_L1_loss, epoch_b1_L1_loss, epoch_b2_L1_loss, epoch_d0_L1_loss, epoch_d1_L1_loss, epoch_d2_L1_loss]:
                f.write(str(s))
                f.write(',')
            f.write('\n')

        if (epoch + 1) % 1 == 0:
            torch.save(Net_Q.state_dict(), os.path.join(out_dir, comp + "_Q_" + str(args.qp) + "_epoch_%d_%.4f_%.4f_%.4f_%.4f.pkl" % (epoch, epoch_q_L1_loss, epoch_loss, val_out_info_list[12], test_out_info_list[14])))
            torch.save(Net_BD.state_dict(), os.path.join(out_dir, comp + "_BD_" + str(args.qp) + "_epoch_%d_%.4f_%.4f_%.4f_%.4f.pkl" % (epoch, epoch_b2_L1_loss, epoch_loss, val_out_info_list[12], test_out_info_list[14])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobID', type=str, default='0000')
    parser.add_argument("--isLuma", dest='isLuma', action="store_true")
    parser.add_argument('--inputDir', type=str, default='/input/')
    parser.add_argument('--inputDir1', type=str, default='/input/')
    parser.add_argument('--outDir', type=str, default='/output/')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--dr', default=20, type=int, help='decay rate of lr')
    parser.add_argument('--qp', default=22, type=int, help='quantization step')
    parser.add_argument('--epoch', default=100, type=int, help='number of total epoch')
    parser.add_argument('--batchSize', default=200, type=int, help='batch size')
    parser.add_argument('--lamb0', default=1.0, type=float, help='weight of loss function')
    parser.add_argument('--lamb1', default=1.0, type=float, help='weight of loss function')
    parser.add_argument('--lamb2', default=1.0, type=float, help='weight of loss function')
    parser.add_argument('--predID', default=0, type=int, help='[0 1 2] [qt bt direction]')
    parser.add_argument('--train_num_workers', default=8, type=int, help='num_workers for training dataloader')
    parser.add_argument('--val_num_workers', default=6, type=int, help='num workers for validation dataloader')
    parser.add_argument('--test_num_workers', default=6, type=int, help='num workers for testing dataloader')
    parser.add_argument('--model_type', type=str, default='type of model backbone, CNN, SA, SACNN')

    parser.add_argument('--net_Q_path', type=str, default=None, help='type of model backbone, CNN, SA, SACNN')
    parser.add_argument('--net_BD_path', type=str, default=None, help='type of model backbone, CNN, SA, SACNN')

    parser.add_argument("--classification", dest='classification', action="store_true")
    parser.add_argument("--focal_loss", dest='focal_loss', action="store_true")
    parser.add_argument('--min_ratio', default=0.1, type=float, help='min ratio of focal_loss')
    parser.add_argument('--focal_gamma', default=1.0, type=float, help='gamma of focal loss')
    parser.add_argument('--post_test', dest='post_test', action="store_true")
    parser.add_argument('--depth_label', dest='depth_label', action="store_true")  # 使用depth-based label而不是分层的label
    parser.add_argument("--aux_loss", dest='aux_loss', action="store_true") # 使用块划分标签制作token sparsification labels

    parser.add_argument('--C_ratio', default=1.0, type=float, help='ratio of channels')  # 指定当前ratio

    parser.add_argument('--lambq', default=1.0, type=float, help='weight of loss function')
    parser.add_argument('--lambb0', default=0.8, type=float, help='weight of loss function')
    parser.add_argument('--lambb1', default=1.0, type=float, help='weight of loss function')
    parser.add_argument('--lambb2', default=1.2, type=float, help='weight of loss function')
    parser.add_argument('--lambd0', default=1.0, type=float, help='weight of loss function')
    parser.add_argument('--lambd1', default=1.0, type=float, help='weight of loss function')
    parser.add_argument('--lambd2', default=1.0, type=float, help='weight of loss function')
    parser.add_argument('--lambresb0', default=0.5, type=float, help='weight of loss function')
    parser.add_argument('--lambresb1', default=0.5, type=float, help='weight of loss function')
    parser.add_argument('--lambresb2', default=0.5, type=float, help='weight of loss function')
    args = parser.parse_args()

    if 'CNN' in args.model_type:
        # import Model_QBD as model
        pass
    elif 'DyLight_SA' in args.model_type:
        # import Model_QBD_SA_sDy as model
        pass
    elif 'Light_SA' in args.model_type:
        import Model_QBD_SA_s as model
    elif 'SA' in args.model_type:
        # import Model_QBD_SA as model
        pass


    # 在critical channel number, 我们针对QP使用不同的pruning ratio
    qt_ratio = 1 - (48.285 * np.log(args.qp) - 138.98) / 100
    mt_0_ratio = 1 - (53.682 * np.log(args.qp) - 136.1) / 100
    mt_1_ratio = 1 - (46.113 * np.log(args.qp) - 87.609) / 100
    mt_2_ratio = 1 - (22.268 * np.log(args.qp) + 10.959) / 100
    print("ratio: ", qt_ratio, mt_0_ratio, mt_1_ratio, mt_2_ratio)

    qt_sparse_threshold = [[qt_ratio], [qt_ratio], [qt_ratio, qt_ratio]]
    mt_sparse_threshold=[[mt_0_ratio, mt_0_ratio], [mt_1_ratio, mt_1_ratio], [mt_2_ratio, mt_2_ratio]]

    paras_print(args)
    if args.predID == 0:
        pre_train_Q(args)
    elif args.predID == 1:
        pre_train_BD(args)
    elif args.predID == 2:
        train_QBD(args)
    else:
        print("Unknown predID!!!")
