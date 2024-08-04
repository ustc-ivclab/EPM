'''
Function:
  Network inference + Post processing

Main functions:
  * inference_VVC_seqs(args)

Author: Aolin Feng
'''

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
import time
from einops import rearrange


from Metrics import inference_pre_QBD, post_process, seq_post_process

work_on_999 = False

SAVE_MID_RESULT = False
POST_PROCESS = True
frm_global = 300
# frm_global = None
SSRatio = 80

def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_pretrain_model(current_model, pretrain_model):
    source_dict = torch.load(pretrain_model)
    if "state_dict" in source_dict.keys():
        source_dict = remove_prefix(source_dict['state_dict'], 'module.')
    else:
        source_dict = remove_prefix(source_dict, 'module.')
    # dest_dict = current_model.state_dict()
    # trained_dict = {k: v for k, v in source_dict.items() if
    #                 k in dest_dict and source_dict[k].shape == dest_dict[k].shape}
    # dest_dict.update(trained_dict)
    # current_model.load_state_dict(dest_dict)
    current_model.load_state_dict(source_dict)
    # for k, v in trained_dict.items():
    #     print(k)
    return current_model

def load_sequences_info():
    num = 26
    seqs_info_path = r"VVC_Test_Sequences.txt" # VVC_Test_Sequences
    seqs_info_fp = open(seqs_info_path, 'r')
    data = []
    for line in seqs_info_fp:
        if "end!!!!" in line:
            break
        data.append(line.rstrip('\n').split(','))
    seqs_info_fp.close()
    data = np.array(data)
    print(data.shape)
    seqs_name = data[:num, 0]
    seqs_path_name = data[:num, 1]
    seqs_width = data[:num, 2].astype(np.int64)  # enough bits for calculating h*w
    seqs_height = data[:num, 3].astype(np.int64)
    seqs_frmnum = data[:num, 4].astype(np.int64)

    sub_frmnum_list, block_num_list = [], []
    for i in range(num):
        SubSampleRatio = 30
        if i >= 79:
            SubSampleRatio = 1
        SubSampleRatio = SSRatio
        if frm_global is None:
            sub_frmnum = (seqs_frmnum[i] + SubSampleRatio - 1) // SubSampleRatio
        else:
            sub_frmnum = (300 + SubSampleRatio - 1) // SubSampleRatio

        sub_frmnum_list.append(sub_frmnum)
        block_num = (seqs_width[i] // 64) * (seqs_height[i] // 64) * sub_frmnum
        block_num_list.append(block_num)

    return seqs_name, seqs_path_name, seqs_width, seqs_height, seqs_frmnum, sub_frmnum_list, block_num_list

def import_yuv420(file_path, width, height, frm_num, SubSampleRatio=1, is10bit=False):
    fp = open(file_path,'rb')
    pixnum = width * height
    subnumfrm = (frm_num + SubSampleRatio - 1) // SubSampleRatio # actual frame number after downsampling
    if is10bit:
        data_type = np.uint16
    else:
        data_type = np.uint8
    y_temp = np.zeros(pixnum*subnumfrm, dtype=data_type)
    u_temp = np.zeros(pixnum*subnumfrm // 4, dtype=data_type)
    v_temp = np.zeros(pixnum*subnumfrm // 4, dtype=data_type)
    for i in range(0, frm_num, SubSampleRatio):
        if is10bit:
            fp.seek(i * pixnum * 3, 0)
        else:
            fp.seek(i * pixnum * 3 // 2, 0)
        subi = i // SubSampleRatio
        y_temp[subi*pixnum : (subi+1)*pixnum] = np.fromfile(fp, dtype=data_type, count=pixnum, sep='')
        u_temp[subi*pixnum//4 : (subi+1)*pixnum//4] = np.fromfile(fp, dtype=data_type, count=pixnum//4, sep='')
        v_temp[subi*pixnum//4 : (subi+1)*pixnum//4] = np.fromfile(fp, dtype=data_type, count=pixnum//4, sep='')
    fp.close()
    y = y_temp.reshape((subnumfrm, height, width))
    u = u_temp.reshape((subnumfrm, height//2, width//2))
    v = v_temp.reshape((subnumfrm, height//2, width//2))
    return y, u, v  # return frm_num * H * W

def output_block_yuv(file_path, width, height, block_size, in_overlap, numfrm, SubSampleRatio, is10bit=False, save_path=None):
    y, u, v = import_yuv420(file_path, width, height, numfrm, SubSampleRatio, is10bit=is10bit)
    if is10bit:
        y = (np.round(y / 4)).clip(0, 255).astype(np.uint8)
        u = (np.round(u / 4)).clip(0, 255).astype(np.uint8)
        v = (np.round(v / 4)).clip(0, 255).astype(np.uint8)
    block_num_in_width = width // block_size
    block_num_in_height = height // block_size
    # print(block_num_in_width, block_num_in_height)
    for id, comp in enumerate([y, u, v]):
        if id == 0:
            overlap = in_overlap
            comp_block_size = block_size
        else:
            overlap = int(in_overlap / 2)
            comp_block_size = block_size // 2
        pad_comp = np.zeros((comp.shape[0], comp.shape[1]+overlap, comp.shape[2]+overlap), dtype=np.uint8)
        pad_comp[:, overlap:, overlap:] = comp
        subnumfrm = comp.shape[0]

        block_list = []
        for f_num in range(subnumfrm):
            for i in range(block_num_in_height):
                for j in range(block_num_in_width):
                    block_list.append(pad_comp
                         [f_num, i * comp_block_size:(i + 1) * comp_block_size + overlap, j * comp_block_size:(j + 1) * comp_block_size + overlap])
        if id == 0:
            block_y = np.array(block_list)
        elif id == 1:
            block_u = np.array(block_list)
        else:
            block_v = np.array(block_list)

    if save_path is not None:
        out_fp = open(save_path, "wb")
        for i in range(block_y.shape[0]):
            out_fp.write(block_y[i].reshape(-1))
            out_fp.write(block_u[i].reshape(-1))
            out_fp.write(block_v[i].reshape(-1))
        out_fp.close()

    # print('shape of block_y', block_y.shape)
    # print('shape of block_u', block_u.shape)
    # print('shape of block_v', block_v.shape)
    # del block_y, block_u, block_v
    return block_y, block_u, block_v  # num_block * block_size * block_size

def yuv444_to_rgb(yuv444_data):
    # 提取Y、U、V分量
    Y = yuv444_data[..., 0]
    U = yuv444_data[..., 1]
    V = yuv444_data[..., 2]
    
    # 初始化RGB数组
    rgb_data = np.empty_like(yuv444_data)
    
    # 进行转换
    rgb_data[..., 0] = Y + 1.402 * (V - 128)  # R
    rgb_data[..., 1] = Y - 0.344136 * (U - 128) - 0.714136 * (V - 128)  # G
    rgb_data[..., 2] = Y + 1.772 * (U - 128)  # B
    
    # 裁剪以确保值在 [0, 255] 范围内
    rgb_data = np.clip(rgb_data, 0, 255)
    
    return rgb_data.astype(np.uint8)


@torch.no_grad()
def inference_VVC_seqs(args):
    save_dir = os.path.join(args.outDir, args.jobID, "PartitionMat")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("saving_dir: ", save_dir)

    seqs_block_time = np.zeros(22)
    seqs_net_time = np.zeros((22, 4, 2))
    seqs_post_time = np.zeros((22, 4, 2))

    seqs_name, seqs_path_name, seqs_width, seqs_height, seqs_frmnum, sub_frmnum_list, block_num_list = load_sequences_info()

    if work_on_999:
        seq_cfg_dir = '/ghome/fengxm/PMP_plus/cfg/per-sequence'
    else:
        seq_cfg_dir = r".\cfg\per-sequence"
        # for seq_id in range(args.startSeqID, args.startSeqID + args.seqNum):
    for seq_id in range(args.startSeqID, 26):
        # ********************************** Load Sequence Information *************************************
        if seq_id != 18:
            continue
        seq_name = seqs_name[seq_id]
        seq_path_name = seqs_path_name[seq_id].rstrip(".yuv")
        width = seqs_width[seq_id]
        height = seqs_height[seq_id]
        if frm_global is None:
            numfrm = seqs_frmnum[seq_id]
        else:
            numfrm = frm_global
        sub_numfrm = sub_frmnum_list[seq_id]
        block_num = block_num_list[seq_id]
        is10bit = False
        seq_cfg_path = os.path.join(seq_cfg_dir, seq_name + ".cfg")
        seq_cfg_fp = open(seq_cfg_path)
        for line in seq_cfg_fp:
            if "InputFile" in line:
                line = line.rstrip("\n").split('#')[0]  # remove annotation
                line = line.replace(" ", "")   # remove space
                seq_path = line.split(":", 1)[1]  # sequence path

            elif "InputBitDepth" in line:
                line = line.rstrip("\n").split('#')[0]  # remove annotation
                line = line.replace(" ", "")   # remove space
                bit_depth = line.split(":", 1)[1]
                if bit_depth == "10":
                    is10bit = True
        print(seq_name)
        # ********************************** Load Input Blocks *************************************
        start_time = time.time()
        if not work_on_999:
            seq_path = os.path.join("E:\\VVC_test", seq_path)
        else:
            seq_path = os.path.join("/gdata/fengxm/VVC_test_sequences", seq_path)
        block_y, block_u, block_v = output_block_yuv(seq_path, width, height, block_size=64, in_overlap=4,
                                                     numfrm=numfrm, SubSampleRatio=SSRatio, is10bit=is10bit)
        seqs_block_time[seq_id-args.startSeqID] = time.time() - start_time

        for comp_id, comp in enumerate(["Luma",]):
        # for comp_id, comp in enumerate(["Chroma", "Luma"]):
            input_batch = torch.FloatTensor(np.expand_dims(block_y, 1))
            if comp == "Chroma":
                input_batch = F.max_pool2d(input_batch, 2)
                input_batch1 = torch.FloatTensor(np.expand_dims(block_u, 1))
                input_batch2 = torch.FloatTensor(np.expand_dims(block_v, 1))
                input_batch = torch.cat([input_batch, input_batch1, input_batch2], 1)
                del input_batch1, input_batch2
            # print('input_batch.shape:', input_batch.shape)

            # print("Creating inference data loader...")
            dataset = TensorDataset(input_batch)
            QB_test_loader = DataLoader(dataset=dataset, num_workers=2, batch_size=args.batchSize, pin_memory=True, shuffle=False)

            # for qp in [22, 27, 32, 37]:
            for qp in [32]:
                # ********************************** Load Models *************************************
                qt_lamb1, qt_lamb2, lamba_params = None, None, None
                if args.model_type == 'SA':
                    if comp == 'Luma':
                        if qp == 22:
                            qt_lamb1, qt_lamb2 = 6, 1
                            lamb1, lamb4, lamb5 = 0.9701971247147136, 0.9864117362694592, 0.004648172041245801
                            lamb2, lamb3 = 0.72422494300545, 1.609634154207796
                        elif qp == 27:
                            qt_lamb1, qt_lamb2 = 14, 7
                            lamb1, lamb4, lamb5 = 0.7869711099685378, 0.8736076147247209, 0.09166046603120614
                            lamb2, lamb3 = 0.010908057782407554, 1.9818478115597669
                        elif qp == 32:
                            qt_lamb1, qt_lamb2 = 5, 1
                            lamb1, lamb4, lamb5 = 0.9826025517493511, 0.9570677286719542, 0.026347191707286766
                            lamb2, lamb3 = 0.6791854995327337, 1.9922147750278119
                        elif qp == 37:
                            qt_lamb1, qt_lamb2 = 15, 3
                            lamb1, lamb4, lamb5 = 0.8547507816937049, 0.9971921164402894, 0.0022283757223981113
                            lamb2, lamb3 = 0.7228900116403871, 1.0198461789777444
                    elif comp == 'Chroma':
                        if qp == 22:
                            qt_lamb1, qt_lamb2 = 7, 1
                            lamb1, lamb4, lamb5 = 0.926624, 0.9756405, 0.0070945
                            lamb2, lamb3 = 0.8674367998007769, 0.7877632498203448
                        elif qp == 27:
                            qt_lamb1, qt_lamb2 = 9, 1
                            lamb1, lamb4, lamb5 = 0.987343666, 0.99018836, 0.003687703
                            lamb2, lamb3 = 0.9622627145140151, 1.3789512554989878
                        elif qp == 32:
                            qt_lamb1, qt_lamb2 = 7, 1
                            lamb1, lamb4, lamb5 = 0.986950569, 0.9955934, 0.0012378
                            lamb2, lamb3 = 0.37754878756467486, 1.7982093226584765
                        elif qp == 37:
                            qt_lamb1, qt_lamb2 = 8, 1
                            lamb1, lamb4, lamb5 = 0.73668227, 0.97768294, 0.0221267
                            lamb2, lamb3 = 0.9880989796111084, 1.9948425728902635
                    else:
                        raise Exception('invalid format')
                    lamba_params = {'lamb1': lamb1, 'lamb2':lamb2, 'lamb3':lamb3, 'lamb4':lamb4, 'lamb5':lamb5}
                    
                start_time = time.time()
                if comp == 'Luma':
                    if 'LightSA' in args.model_type:
                        Net_Q = model.Luma_Q_Net(classification=True, c_ratio = args.C_ratio)
                        Net_BD = model.Luma_MSBD_Net(classification=True, c_ratio = args.C_ratio)
                    elif 'DySA' in args.model_type:
                        # C1
                        qt_ratio = 1 - (48.285 * np.log(qp) - 138.98) / 100
                        mt_0_ratio = 1 - (53.682 * np.log(qp) - 136.1) / 100
                        mt_1_ratio = 1 - (46.113 * np.log(qp) - 87.609) / 100
                        mt_2_ratio = 1 - (22.268 * np.log(qp) + 10.959) / 100
                        # 更低的ratio  C2
                        qt_ratio, mt_0_ratio, mt_1_ratio, mt_2_ratio = max(0.1, qt_ratio - 0.4), max(0.1, mt_0_ratio - 0.1), max(0.1, mt_1_ratio - 0.1), max(0.1, mt_2_ratio - 0.1)
                        # # ratio=1 C0
                        # qt_ratio, mt_0_ratio, mt_1_ratio, mt_2_ratio = 1.0, 1.0, 1.0, 1.0
                        qt_sparse_threshold = [[qt_ratio], [qt_ratio], [qt_ratio, qt_ratio]]
                        mt_sparse_threshold=[[mt_0_ratio, mt_0_ratio], [mt_1_ratio, mt_1_ratio], [mt_2_ratio, mt_2_ratio]]
                        print("-----------> ratio: ", qt_ratio, mt_0_ratio, mt_1_ratio, mt_2_ratio)
                        Net_Q = model.Luma_Q_Net(classification=True, c_ratio = args.C_ratio, sparse_threshold=qt_sparse_threshold)
                        Net_BD = model.Luma_MSBD_Net(classification=True, c_ratio = args.C_ratio, sparse_threshold=mt_sparse_threshold)
                    elif 'CNN' in args.model_type:
                        Net_Q = model.Luma_Q_Net(c_ratio=args.C_ratio, classification=True,)
                        Net_BD = model.Luma_MSBD_Net(c_ratio=args.C_ratio)
                    else:
                        raise Exception('invalid model type.')
                    comp = "Luma"
                else:
                    if 'LightSA' in args.model_type:
                        Net_Q = model.Chroma_Q_Net(classification=True, c_ratio = args.C_ratio)
                        Net_BD = model.Chroma_MSBD_Net(classification=True, c_ratio = args.C_ratio)
                    else:
                        Net_Q = model.Chroma_Q_Net()
                        Net_BD = model.Chroma_MSBD_Net()
                    comp = "Chroma"

                # net_Q_path = "./pretrained/" + args.model_type + '/' + comp + "_Q_" + str(qp) + ".pkl"
                # net_BD_path = "./pretrained/"+ args.model_type + '/' + comp + "_BD_" + str(qp) + ".pkl"
                
                net_Q_path = os.path.join(args.checkpoints_dir, args.model_type, comp + "_Q_" + str(qp) + ".pkl")
                net_BD_path = os.path.join(args.checkpoints_dir, args.model_type, comp + "_BD_" + str(qp) + ".pkl")
                Net_Q = load_pretrain_model(Net_Q, net_Q_path)
                Net_BD = load_pretrain_model(Net_BD, net_BD_path)
                Net_Q = nn.DataParallel(Net_Q).cuda()
                Net_BD = nn.DataParallel(Net_BD).cuda()
                # ********************************** Network Inference *************************************
                # qt_out_batch, bt_out_batch, dire_out_batch_reg = inference_pre_QBD(QB_test_loader, Net_Q.eval(), Net_BD.eval(), classification=True if 'SA' in args.model_type else False ,output_decisions=False)

                # output decisions
                qt_out_batch, bt_out_batch, dire_out_batch_reg, total_decisions = inference_pre_QBD(QB_test_loader, Net_Q.train(), Net_BD.train(), classification=True if 'SA' in args.model_type else False, output_decisions=True)
                def gen_visualization(image, decisions, token_size_list = [2, 4, 8, 8], alpha=0.2):
                    # keep_indices = get_keep_indices(decisions)
                    image = np.asarray(image)
                    stages = [image]
                    for i, single_decision in enumerate(decisions):
                        image_tokens = rearrange(image, '(h h_i) (w w_i) c -> h w h_i w_i c', h_i=token_size_list[i], w_i=token_size_list[i])
                        H,W,_,_,_ = image_tokens.shape
                        image_tokens = rearrange(image_tokens, 'h w h_i w_i c -> (h w) h_i w_i c')
                        indices = [i for i in range(image_tokens.shape[0]) if i not in single_decision]
                        tokens = image_tokens.copy()
                        tokens[indices] = alpha * tokens[indices] + (1 - alpha) * 255
                        stages.append(rearrange(tokens, '(h w) h_i w_i c -> (h h_i) (w w_i) c',h=H,w=W))                  
                    viz = np.concatenate(stages, axis=1)
                    return viz
                # plot decisions
                y, u, v = import_yuv420(seq_path, width, height, numfrm, SubSampleRatio=SSRatio, is10bit=is10bit)
                u = np.repeat(np.repeat(u, 2, axis=1), 2, axis=2)
                v = np.repeat(np.repeat(v, 2, axis=1), 2, axis=2)
                yuv_data = np.stack([y,u,v], axis=3)
                import cv2
                import matplotlib.pyplot as plt
                rgb_image = cv2.cvtColor(yuv_data[0], cv2.COLOR_YUV2RGB)
                rgb_image = rgb_image[:rgb_image.shape[0] // 64 * 64, :rgb_image.shape[1] // 64 * 64]
                ctu_h_num, ctu_w_num = rgb_image.shape[0] // 64, rgb_image.shape[1] // 64
                # ctu_id = 5
                # 
                # ctu_h_id, ctu_w_id = ctu_id // ctu_w_num, ctu_id % ctu_w_num
                # decisions = [total_decisions[0][0][0][ctu_id], total_decisions[0][1][0][ctu_id], total_decisions[0][2][0][ctu_id], total_decisions[0][3][0][ctu_id],]
                # rgb_block = rgb_image[64 * ctu_h_id:64 * ctu_h_id + 64, 64 * ctu_w_id:64 * ctu_w_id + 64]
                # viz = gen_visualization(rgb_block, decisions)
                # from scipy.ndimage import zoom
                # plt.figure(figsize=(15, 5))
                # plt.imsave('sparseC2.png', viz, dpi=500)
                
                masked_image = np.zeros((5, rgb_image.shape[0], rgb_image.shape[1], rgb_image.shape[2]))
                for ctu_id in range(ctu_h_num * ctu_w_num):
                    ctu_h_num, ctu_w_num = rgb_image.shape[0] // 64, rgb_image.shape[1] // 64
                    ctu_h_id, ctu_w_id = ctu_id // ctu_w_num, ctu_id % ctu_w_num
                    decisions = [total_decisions[0][0][0][ctu_id], total_decisions[0][1][0][ctu_id], total_decisions[0][2][0][ctu_id], total_decisions[0][3][0][ctu_id],]
                    rgb_block = rgb_image[64 * ctu_h_id:64 * ctu_h_id + 64, 64 * ctu_w_id:64 * ctu_w_id + 64]
                    viz = gen_visualization(rgb_block, decisions)
                    masked_image[:, 64 * ctu_h_id:64 * ctu_h_id + 64, 64 * ctu_w_id:64 * ctu_w_id + 64] = rearrange(viz, 'h (w_i w) c -> w_i h w c', w_i=5)
                # plt.figure(figsize=(15, 5))
                plt.imsave('sparseC2.png', rearrange(masked_image[:,:192, :192], 'w_i h w c -> h (w_i w) c').astype(np.uint8), dpi=500)
                
                # qt_out_batch, bt_out_batch, dire_out_batch_reg = inference_pre_QBD(QB_test_loader, Net_Q, Net_BD, classification=True if 'SA' in args.model_type else False)
                seqs_net_time[seq_id-args.startSeqID, (qp-22)//5, comp_id] = time.time() - start_time

                # ********************************** Post Process ************************************
                start_time = time.time()
                qt_out_batch = torch.FloatTensor(qt_out_batch).cuda()  # b*1*8*8
                bt_out_batch = bt_out_batch.cpu().numpy()
                dire_out_batch_reg = dire_out_batch_reg.cpu().numpy()
                
                # bt_out_batch[:, 1:2, :, :] = bt_out_batch[:, 1:2, :, :] + bt_out_batch[:, 0:1, :, :]
                # bt_out_batch[:, 2:3, :, :] = bt_out_batch[:, 2:3, :, :] + bt_out_batch[:, 1:2, :, :]

                save_path = os.path.join(save_dir, seq_path_name + "_" + comp + "_QP" + str(qp) + "_PartitionMat.txt")
                print("Save:", save_path)
                from Metrics_origin import seq_post_process
                seq_post_process(qt_out_batch.cuda(), bt_out_batch, dire_out_batch_reg, comp, sub_numfrm, width, height, save_path)
                # bt_out_batch = bt_out_batch.cpu().numpy()
                # dire_out_batch_reg = dire_out_batch_reg.cpu().numpy()
                # seq_post_process(qt_out_batch.cpu(), bt_out_batch, dire_out_batch_reg, comp, sub_numfrm, width, height, save_path, qt_lamb1=qt_lamb1, qt_lamb2=qt_lamb2, lamb_params=lamba_params)

                seqs_post_time[seq_id-args.startSeqID, (qp-22)//5, comp_id] = time.time() - start_time
    # ********************************** Log Time Information ************************************
    sta_log_path = os.path.join(args.outDir, args.jobID,
                                "Time_Sta_" + str(args.startSeqID) + "_" + str(args.startSeqID + args.seqNum) + ".txt")
    sta_log_fp = open(sta_log_path, "w")
    for seq_id in range(args.seqNum):
        for qp_id in range(4):
            for s in [str(seqs_block_time[seq_id]),
                      str(seqs_net_time[seq_id, qp_id, 0]), str(seqs_net_time[seq_id, qp_id, 1]),
                      str(seqs_post_time[seq_id, qp_id, 0]), str(seqs_post_time[seq_id, qp_id, 1])]:
                sta_log_fp.write(s)
                sta_log_fp.write(',')
            sta_log_fp.write('\n')

    print("Sum time:", np.sum(seqs_block_time) + np.sum(seqs_net_time) + np.sum(seqs_post_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobID', type=str, default='0000')
    parser.add_argument('--inputDir', type=str, default='/input/')
    parser.add_argument('--outDir', type=str, default='/output/')
    # parser.add_argument('--log_dir', type=str, default='/output/log')
    parser.add_argument('--batchSize', default=200, type=int, help='batch size')
    parser.add_argument('--startSeqID', default=0, type=int, help='QP start ID')
    parser.add_argument('--seqNum', default=22, type=int, help='test QP number')
    parser.add_argument('--model_type', default=None)
    parser.add_argument('--checkpoints_dir', type=str, default='/input/')
    parser.add_argument('--C_ratio', type=float, default=1.0)

    args = parser.parse_args()

    if 'CNN' in args.model_type:
        import Model_QBD as model
    elif args.model_type == 'SA':
        import Model_QBD_SA as model
    elif  'LightSA' in args.model_type:
        import Model_QBD_SA_s as model
    elif 'DySA' in args.model_type:
        import Model_QBD_SA_sDy as model
        # qt_sparse_threshold = [[0.7], [0.7], [0.7, 0.7]]
        # mt_sparse_threshold=[[0.7, 0.7], [0.5, 0.5], [0.3, 0.3]]
    else:
        raise Exception('invalid model type')

    start_time = time.time()
    inference_VVC_seqs(args)
    infe_time = time.time() - start_time
    print('Total inference time:', infe_time)
    
    """work on 999
    python Inference_QBD.py --jobID CNN --inputDir /gdata/fengxm/VVC_test_sequences/ --outDir /ghome/fengxm/PMP_plus/Output --batchSize 400 --startSeqID 0 --seqNum 22 --model_type CNN --checkpoints_dir /ghome/fengxm/PMP_plus/pretrained
    """
    
    '''
    python /code/DebugInference_QBD.py --batchSize 400 --inputDir /data/FengAolin/VTM10_Partition_LMCS0_LFNST0 --outDir /output/
    python dp_inference.py --input_dir /DataSet --batchSize 200
    startdocker -P /ghome/fengal -D /gdata/fengal -c "python /ghome/fengal/HM_Fast_Partition/dp_train.py --input_dir /gdata/fengal/HM_Fast_Partition --out_dir /output/" bit:5000/wangyc-pytorch1.0.1_cuda10.0_apex
    --startQPID 2 --qpNum 1 --batchSize 400 --inputDir E:\VVC-Fast-Partition-DP\Dataset\Input --outDir E:\VVC-Fast-Partition-DP\Output\Test
    --startQPID 2 --qpNum 1 --batchSize 400 --inputDir E:\VVC-Fast-Partition-DP\Dataset\Input --outDir E:\VVC-Fast-Partition-DP\Output\Test
    '''




