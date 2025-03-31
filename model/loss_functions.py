from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from scipy.special import softmax


# def find_topk(a, k, axis=-1, largest=True, sorted=True):
#     if axis is None:
#         axis_size = a.size
#     else:
#         axis_size = a.shape[axis]
#     assert 1 <= k <= axis_size

#     a = np.asanyarray(a)
#     if largest:
#         index_array = np.argpartition(a, axis_size-k, axis=axis)
#         topk_indices = np.take(index_array, -np.arange(k)-1, axis=axis)
#     else:
#         index_array = np.argpartition(a, k-1, axis=axis)
#         topk_indices = np.take(index_array, np.arange(k), axis=axis)
#     topk_values = np.take_along_axis(a, topk_indices, axis=axis)
#     if sorted:
#         sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
#         if largest:
#             sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)
#         sorted_topk_values = np.take_along_axis(
#             topk_values, sorted_indices_in_topk, axis=axis)
#         sorted_topk_indices = np.take_along_axis(
#             topk_indices, sorted_indices_in_topk, axis=axis)
#         return sorted_topk_values, sorted_topk_indices
#     return topk_values, topk_indices

class ContrastiveLoss(nn.Module):
    def __init__(self, temp_param, eps=1e-12, reduce=True):
        super(ContrastiveLoss, self).__init__()
        self.temp_param = temp_param
        self.eps = eps
        self.reduce = reduce

    def get_score(self, query, key):
        '''
        query : (NxL) x C or N x C -> T x C  (initial latent features)
        key : M x C     (memory items)
        '''
        qs = query.size()
        ks = key.size()

        score = torch.matmul(query, torch.t(key))  # Fea x Mem^T : (TXC) X (CXM) = TxM
        score = F.softmax(score, dim=1)  # TxM

        return score

    def forward(self, queries, items):
        '''
        anchor : query
        positive : nearest memory item
        negative(hard) : second nearest memory item
        queries : N x L x C
        items : M x C
        '''
        batch_size = queries.size(0)
        d_model = queries.size(-1)

        # margin from 1.0 
        loss = torch.nn.TripletMarginLoss(margin=1.0, reduce=self.reduce)

        queries = queries.contiguous().view(-1, d_model)  # (NxL) x C >> T x C
        score = self.get_score(queries, items)  # TxM

        # gather indices of nearest and second nearest item
        _, indices = torch.topk(score, 2, dim=1)

        # 1st and 2nd nearest items (l2 normalized)
        pos = items[indices[:, 0]]  # TxC
        neg = items[indices[:, 1]]  # TxC
        anc = queries  # TxC

        spread_loss = loss(anc, pos, neg)

        if self.reduce:
            return spread_loss

        spread_loss = spread_loss.contiguous().view(batch_size, -1)  # N x L

        return spread_loss  # N x L


class GatheringLoss(nn.Module):
    def __init__(self, reduce=True):
        super(GatheringLoss, self).__init__()
        self.reduce = reduce

    def get_score(self, query, key):
        '''
        query : (NxL) x C or N x C -> T x C  (initial latent features)
        key : M x C     (memory items)
        '''
        qs = query.shape
        query = query.reshape(-1, qs[-1])
        score = torch.matmul(query, torch.t(key))  # Fea x Mem^T : (TXC) X (CXM) = TxM
        score = score.reshape(qs[0], qs[1], -1)
        score = F.softmax(score, dim=-1)  # TxM
        score=score.reshape(qs[0]*qs[1],-1)
        return score

    def forward(self, queries, items):
        '''
        queries : N x L x C
        items : M x C
        '''
        batch_size = queries.size(0)
        d_model = queries.size(-1)

        loss_mse = torch.nn.MSELoss(reduce=self.reduce)

        # queries = queries.contiguous().view(-1, d_model)  # (NxL) x C >> T x C
        score = self.get_score(queries, items)  # TxM

        _, indices = torch.topk(score, 1, dim=1)

        gathering_loss = loss_mse(queries.reshape(-1,d_model), items[indices].squeeze(1))

        if self.reduce:
            return gathering_loss

        gathering_loss = torch.sum(gathering_loss, dim=-1)  # T
        gathering_loss = gathering_loss.contiguous().view(batch_size, -1)  # N x L

        return gathering_loss





#
#

class Gathering_gat_Loss(nn.Module):
    def __init__(self, reduce=True):
        super(Gathering_gat_Loss, self).__init__()
        self.reduce = reduce

    def get_score(self, query, key):
        '''
        query : (NxL) x C or N x C -> T x C  (initial latent features)
        key : M x C     (memory items)
        '''
        qs = query.shape
        query = query.reshape(-1, qs[-1])
        score = torch.matmul(query, torch.t(key))  # Fea x Mem^T : (TXC) X (CXM) = TxM
        # score = score.reshape(qs[0], qs[1], -1)
        score = F.softmax(score, dim=-1)  # TxM
        # score=score.mean(1)
        return score

    def forward(self, queries, items):
        '''
        queries : N x L x C
        items : M x C
        '''
        batch_size = queries.size(0)
        d_model = queries.size(-1)

        loss_mse = torch.nn.MSELoss(reduce=self.reduce)

        queries = queries.contiguous().view(-1, d_model)  # (NxL) x C >> T x C
        score = self.get_score(queries, items)  # TxM

        gat_dis, indices = torch.topk(score, 1, dim=1)
        return gat_dis




class GatheringLossDim(nn.Module):
    def __init__(self, reduce=True):
        super(GatheringLossDim, self).__init__()
        self.reduce = reduce

    def get_score(self, query, key):
        '''
        query : (NxL) x C or N x C -> T x C  (initial latent features)
        key : M x C     (memory items)
        '''

        qs = query.size()
        ks = key.size()

        score = torch.matmul(query, torch.t(key))  # Fea x Mem^T : (TXC) X (CXM) = TxM
        score = F.softmax(score, dim=1)  # TxM

        return score

    def forward(self, queries, items):
        '''
        queries : N x L x C
        items : M x C
        '''
        B, K, T, N = queries.shape
        M, _ = items.shape

        loss_mse = torch.nn.MSELoss(reduce=False)

        queries = queries.contiguous().view(-1, N)  # (NxL) x C >> T x C
        score = self.get_score(queries, items)  # TxM
        score = score.reshape(B, K, T, M)
        queries = queries.reshape(B, K, T, N)

        _, indices = torch.topk(score, 1, dim=-1)  # B K T 1

        gathering_loss = loss_mse(queries, items[indices].squeeze(-2))  # B K T N

        # if self.reduce:
        #     return gathering_loss

        gathering_loss = gathering_loss.sum(-1).sum(1)  # B T

        return gathering_loss


# class GatheringLossNP():
# def __init__(self, reduce=True):
#     self.reduce = reduce

# def get_score(self, query, key):
#     '''
#     query : (NxL) x C or N x C -> T x C  (initial latent features)
#     key : M x C     (memory items)
#     '''
#     score = np.matmul(query, key.T)   # Fea x Mem^T : (TXC) X (CXM) = TxM
#     score = softmax(score, axis=1) # TxM

#     return score

# def get_loss(self, queries, items):
#     '''
#     queries : N x L x C
#     items : M x C
#     '''
#     batch_size = queries.shape[0]
#     d_model = queries.shape[-1]

#     queries = queries.reshape(-1,d_model)    # (NxL) x C >> T x C
#     score = self.get_score(queries, items)      # TxM

#     _, indices = find_topk(score, 1, axis=1)

#     gathering_loss = np.mean((queries - items[indices].squeeze(1)) ** 2, axis=-1, keepdims=True)

#     if self.reduce:
#         return gathering_loss

#     gathering_loss = np.sum(gathering_loss, axis=-1)  # T
#     gathering_loss = gathering_loss.reshape(batch_size, -1)   # N x L

#     return gathering_loss


class EntropyLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        '''
        x (attn_weights) : TxM
        '''
        loss = -1 * x * torch.log(x + self.eps)
        loss = torch.sum(loss, dim=-1)
        loss = torch.mean(loss)
        return loss


class NearestSim(nn.Module):
    def __init__(self):
        super(NearestSim, self).__init__()

    def get_score(self, query, key):
        '''
        query : (NxL) x C or N x C -> T x C  (initial latent features)
        key : M x C     (memory items)
        '''
        qs = query.size()
        ks = key.size()

        score = F.linear(query, key)  # Fea x Mem^T : (TXC) X (CXM) = TxM
        score = F.softmax(score, dim=1)  # TxM

        return score

    def forward(self, queries, items):
        '''
        anchor : query
        positive : nearest memory item
        negative(hard) : second nearest memory item
        queries : N x L x C
        items : M x C
        '''
        batch_size = queries.size(0)
        d_model = queries.size(-1)

        queries = queries.contiguous().view(-1, d_model)  # (NxL) x C >> T x C
        score = self.get_score(queries, items)  # TxM

        # gather indices of nearest and second nearest item
        _, indices = torch.topk(score, 2, dim=1)

        # 1st and 2nd nearest items (l2 normalized)
        pos = F.normalize(items[indices[:, 0]], p=2, dim=-1)  # TxC
        anc = F.normalize(queries, p=2, dim=-1)  # TxC

        similarity = -1 * torch.sum(pos * anc, dim=-1)  # T
        similarity = similarity.contiguous().view(batch_size, -1)  # N x L

        return similarity  # N x L
