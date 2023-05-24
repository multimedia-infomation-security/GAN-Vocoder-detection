from utils.utils import AverageMeter, accuracy
from utils.statistic import get_EER_states, get_HTER_at_thr, calculate, calculate_threshold
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn import functional as F
from  loss.SuperLoss import SuperLoss
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from model import frontends
def _compute_frontend( x):
    frontend = frontends.get_frontend(["mfcc"])(x)
    # print(frontend.shape)
    if frontend.ndim < 4:
        return frontend.unsqueeze(1)  # (bs, 1, n_lfcc, frames)
    return frontend
def eval(valid_dataloader, model):
    criterion =SuperLoss(C=2, lam=1)
    valid_losses = AverageMeter()
    valid_top1 = AverageMeter()
    prob_dict = {}
    label_dict = {}
    model.eval()
    output_dict_tmp = {}
    target_dict_tmp = {}
    with torch.no_grad():
        for iter, (input, target, videoID,fprint) in enumerate(valid_dataloader):
            input = Variable(input).cuda()
            fprint = Variable(fprint).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            recons_data = input - fprint
            fprint = _compute_frontend(fprint)
            input_data = _compute_frontend(input)
            # recons_data = input_data - input_fprint
            recons_data = _compute_frontend(recons_data)
            fprint = model.feature(fprint)
            input = recons_data + fprint
            ######### forward #########
            cls_out, feature = model(input)
            prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
            label = target.cpu().data.numpy()
            # tsne_fea.extend(feats.detach().cpu().numpy())
            # tsne_color.extend(list(channel_label))
            # if (iter + 1) % 3 == 0:
            #     tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
            #     Y = tsne.fit_transform(np.array(tsne_fea))
            #     plt.scatter(Y[:, 0], Y[:, 1], c=tsne_color)
            videoID = videoID.cpu().data.numpy()
            for i in range(len(prob)):
                if(videoID[i] in prob_dict.keys()):
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
        loss = criterion(avg_single_video_output, avg_single_video_target.long())
        acc_valid = accuracy(avg_single_video_output, avg_single_video_target, topk=(1,))
        valid_losses.update(loss.item())
        valid_top1.update(acc_valid[0])
    auc_score = roc_auc_score(label_list.astype('int'), prob_list)
    cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)
    ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)
    return [valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid, auc_score, threshold, ACC_threshold*100]

