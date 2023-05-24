import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from scipy.special import lambertw
from loss.triplet_loss import TripletLoss
from loss.hard_triplet_loss import HardTripletLoss,_get_triplet_mask
from loss.AdLoss import Real_AdLoss
class SuperLoss(nn.Module):

    def __init__(self, C=2, lam=0.25, batch_size=24):
        super(SuperLoss, self).__init__()
        self.tau = math.log(C)
        self.lam = lam  # set to 1 for CIFAR10 and 0.25 for CIFAR100
        self.batch_size = batch_size

    def forward(self, logits, targets):
        l_i = F.cross_entropy(logits, targets, reduction='none').detach()
        sigma = self.sigma(l_i)
        loss = (F.cross_entropy(logits, targets, reduction='none') - self.tau) * sigma + self.lam * (
                    torch.log(sigma) ** 2)
        loss = loss.sum() / self.batch_size
        return loss

    # def tau(self,l_i):
    #     tau = torch.mean(l_i)
    #     return  tau
    def sigma(self, l_i):
        # tau = self.tau(l_i)
        x = torch.ones(l_i.size()) * (-2 / math.exp(1.))
        x = x.cuda()
        y = 0.5 * torch.max(x, (l_i - self.tau) / self.lam)
        y = y.cpu().numpy()
        sigma = np.exp(-lambertw(y))
        sigma = sigma.real.astype(np.float32)
        sigma = torch.from_numpy(sigma).cuda()
        return sigma

class SuperLoss1(nn.Module):

    def __init__(self, lam=0.25, batch_size=24):
        super(SuperLoss1, self).__init__()
        self.tau = math.log(4)
        self.lam = lam  # set to 1 for CIFAR10 and 0.25 for CIFAR100
        self.batch_size = batch_size
        self.l_p=[]
        self.l_n = []
        self.tripletloss=HardTripletLoss(margin=0.1)
        # self.ranking_loss = nn.MarginRankingLoss(margin=0.3,reduction="none")  #
    def forward(self, logits, targets):
        l_i_p,l_i_n = self.tripletloss(logits, targets)
        l_i_p =l_i_p.detach()
        l_i_n =l_i_n.detach()
        sigma_p = self.sigma_p(l_i_p)
        sigma_n = self.sigma_n(l_i_n)
        l_i_p1, l_i_n1= self.tripletloss(logits, targets)
        loss_p = (l_i_p1 - self.tau) * sigma_p + self.lam * (
                    torch.log(sigma_p) ** 2)
        loss_n = (l_i_n1 - self.tau) * sigma_n + self.lam * (
                torch.log(sigma_n) ** 2)
        # # Count number of hard triplets (where triplet_loss > 0)
        loss = loss_p +loss_n

        mask = _get_triplet_mask(targets).float()
        triplet_loss = loss * mask

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = F.relu(triplet_loss)

        # Count number of hard triplets (where triplet_loss > 0)
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        # print(hard_triplets.dtype)
        num_hard_triplets = torch.sum(hard_triplets)

        triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)
        return triplet_loss

    # def tau_p(self,l_i_p):
    #     for i in range (l_i_p.size(0)):
    #         self.l_p .append(l_i_p[i])
    #     tau_p = sum(self.l_p)/len(self.l_p)
    #     return  tau_p
    # def tau_n(self,l_i_n):
    #     for i in range (l_i_n.size(0)):
    #         self.l_n .append(l_i_n[i])
    #     tau_n = sum(self.l_n)/len(self.l_n)
    #     return  tau_n
    def sigma_p(self, l_i_p):
        # tau = self.tau_p(l_i_p)
        x = torch.ones(l_i_p.size()) * (-2 / math.exp(1.))
        x = x.cuda()
        y = 0.5 * torch.max(x, (l_i_p - self.tau) / self.lam)
        y = y.cpu().numpy()
        sigma = np.exp(-lambertw(y))
        sigma = sigma.real.astype(np.float32)
        sigma = torch.from_numpy(sigma).cuda()
        return sigma
    def sigma_n(self, l_i_n):
        # tau = self.tau_n(l_i_n)
        x = torch.ones(l_i_n.size()) * (-2 / math.exp(1.))
        x = x.cuda()
        y = 0.5 * torch.max(x, (l_i_n - self.tau) / self.lam)
        y = y.cpu().numpy()
        sigma = np.exp(-lambertw(y))
        sigma = sigma.real.astype(np.float32)
        sigma = torch.from_numpy(sigma).cuda()
        return sigma
class SuperLoss2(nn.Module):

    def __init__(self, C=2, lam=0.25, batch_size=12):
        super(SuperLoss2, self).__init__()
        self.tau = math.log(C)
        self.lam = lam  # set to 1 for CIFAR10 and 0.25 for CIFAR100
        self.batch_size = batch_size

    def forward(self, discriminator_out, shape_list):
        ad_label1_index = torch.LongTensor(shape_list[0], 1).fill_(0)
        ad_label1 = ad_label1_index.cuda()
        ad_label2_index = torch.LongTensor(shape_list[1], 1).fill_(1)
        ad_label2 = ad_label2_index.cuda()
        ad_label3_index = torch.LongTensor(shape_list[2], 1).fill_(2)
        ad_label3 = ad_label3_index.cuda()
        ad_label = torch.cat([ad_label1, ad_label2, ad_label3], dim=0).view(-1)
        l_i = F.cross_entropy(discriminator_out, ad_label, reduction='none').detach()
        sigma = self.sigma(l_i)
        loss = (F.cross_entropy(discriminator_out, ad_label, reduction='none') - self.tau) * sigma + self.lam * (
                    torch.log(sigma) ** 2)
        loss = loss.sum() / self.batch_size
        return loss
    def sigma(self, l_i):
        # tau = self.tau(l_i)
        x = torch.ones(l_i.size()) * (-2 / math.exp(1.))
        x = x.cuda()
        y = 0.5 * torch.max(x, (l_i - self.tau) / self.lam)
        y = y.cpu().numpy()
        sigma = np.exp(-lambertw(y))
        sigma = sigma.real.astype(np.float32)
        sigma = torch.from_numpy(sigma).cuda()
        return sigma
class SuperLoss3(nn.Module):

    def __init__(self, C=2, lam=0.25, batch_size=24):
        super(SuperLoss3, self).__init__()
        self.tau = math.log(C)
        self.lam = lam  # set to 1 for CIFAR10 and 0.25 for CIFAR100
        self.batch_size = batch_size

        self.tripletloss = TripletLoss(margin=0.3)
    def forward(self, logits, targets):
        l_i = self.tripletloss(logits, targets).detach()
        sigma = self.sigma(l_i)
        loss = (F.cross_entropy(logits, targets) - self.tau) * sigma + self.lam * (
                    torch.log(sigma) ** 2)
        loss = loss.sum() / self.batch_size
        return loss
    def sigma(self, l_i):
        # tau = self.tau(l_i)
        x = torch.ones(l_i.size()) * (-2 / math.exp(1.))
        x = x.cuda()
        y = 0.5 * torch.max(x, (l_i - self.tau) / self.lam)
        y = y.cpu().numpy()
        sigma = np.exp(-lambertw(y))
        sigma = sigma.real.astype(np.float32)
        sigma = torch.from_numpy(sigma).cuda()
        return sigma