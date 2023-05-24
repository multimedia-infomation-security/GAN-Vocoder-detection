import sys

from matplotlib import pyplot as plt
from sklearn import manifold

sys.path.append('../../../')
import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from config import config
from utils.utils import audios
from utils.dataset import YunpeiDataset
from utils.utils import AverageMeter, accuracy, draw_roc
from utils.statistic import get_EER_states, get_HTER_at_thr, calculate, calculate_threshold
from sklearn.metrics import roc_auc_score, roc_curve, auc
# from models.DGFAS import DG_model
# from models.xception import DG_model
import shutil
from model.lcnn import LCNN

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

from model import frontends
device = 'cuda'
def _compute_frontend( x):
    frontend = frontends.get_frontend(["mfcc"])(x)
    # print(frontend.shape)
    if frontend.ndim < 4:
        return frontend.unsqueeze(1)  # (bs, 1, n_lfcc, frames)
    return frontend
def ceshi(test_dataloader, model, threshold):
    valid_top1 = AverageMeter()
    prob_dict = {}
    label_dict = {}
    model.eval()
    output_dict_tmp = {}
    target_dict_tmp = {}
    tsne_fea = []
    tsne_color=[]
    number = 0
    channel_label=[]
    with torch.no_grad():
        for iter, (input, target, videoID,fprint) in enumerate(test_dataloader):
            input = Variable(input).cuda()
            fprint = Variable(fprint).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            recons_data = input - fprint
            fprint = _compute_frontend(fprint)
            # recons_data = input_data - input_fprint
            recons_data = _compute_frontend(recons_data)
            fprint = model.feature(fprint)
            input_data = recons_data + fprint
            ######### forward #########
            cls_out, feature = model(input_data)
            # if(iter)
            channel_label=target
            prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
            label = target.cpu().data.numpy()
            videoID = videoID.cpu().data.numpy()
            # tsne_fea.extend(feats.detach().cpu().numpy())
            # tsne_color.extend(list(channel_label))
            # if (iter + 1) % 1000 == 0:
            #     tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
            #     Y = tsne.fit_transform(np.array(tsne_fea))
            #     plt.scatter(Y[:, 0], Y[:, 1], c=tsne_color)
            #     plt.show()
            for i in range(len(prob)):
                if (videoID[i] in prob_dict.keys()):
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                else:
                    prob_dict[videoID[i]] = []
                    label_dict[videoID[i]] = []
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]] = []
                    target_dict_tmp[videoID[i]] = []
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                    number += 1
                    if (number % 100 == 0):
                        print('**Testing** ', number, ' photos done!')
    print('**Testing** ', number, ' photos done!')
    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)
        # compute loss and acc for every video
        avg_single_video_output = sum(output_dict_tmp[key]) / len(output_dict_tmp[key])
        avg_single_video_target = sum(target_dict_tmp[key]) / len(target_dict_tmp[key])
        acc_valid = accuracy(avg_single_video_output, avg_single_video_target, topk=(1,))
        valid_top1.update(acc_valid[0])

    cur_EER_valid, threshold, FRR_list, FAR_list = get_EER_states(prob_list, label_list)
    ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
    auc_score = roc_auc_score(label_list, prob_list)
    draw_roc(FRR_list, FAR_list, auc_score)
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)
    return [valid_top1.avg, cur_EER_valid, cur_HTER_valid, auc_score, ACC_threshold, threshold]

def main():
    net = LCNN(frontend_algorithm="mfcc").cuda()
    test_data = audios(flag=2,  dataset_name=config.tgt_data)
    test_dataloader = DataLoader(YunpeiDataset(test_data, train=False), batch_size=1, shuffle=False)
    print('\n')
    print("**Testing** Get test files done!")
    # load model
    net_ = torch.load(config.best_model_path + config.tgt_best_model_name)
    net.load_state_dict(net_["state_dict"])
    threshold = net_["threshold"]
    # test model
    test_args = ceshi(test_dataloader, net, threshold)
    print('\n===========Test Info===========\n')
    print(config.tgt_data, 'Test acc: %5.4f' %(test_args[0]))
    print(config.tgt_data, 'Test EER: %5.4f' %(test_args[1]))
    print(config.tgt_data, 'Test HTER: %5.4f' %(test_args[2]))
    print(config.tgt_data, 'Test AUC: %5.4f' % (test_args[3]))
    print(config.tgt_data, 'Test ACC_threshold: %5.4f' % (test_args[4]))
    print('\n===============================\n')

if __name__ == '__main__':
    main()
