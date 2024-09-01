from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.cluster import KMeans


class MemoryModule(nn.Module):
    def __init__(self, n_memory, fea_dim, shrink_thres=0.0025, device=None, memory_init_embedding=None, phase_type=None, config=None):
        super(MemoryModule, self).__init__()
        self.n_memory = n_memory
        self.fea_dim = fea_dim  # C(=d_model)
        self.shrink_thres = shrink_thres
        self.device = device
        self.phase_type = phase_type
        self.memory_init_embedding = memory_init_embedding
        self.config = config

        self.U = nn.Linear(fea_dim, fea_dim)
        self.W = nn.Linear(fea_dim, fea_dim)

        print('loading memory item with random initilzation (for first train phase)')

        self.mem = torch.rand((self.n_memory, self.fea_dim), requires_grad=False).cuda()
        # F.normalize(,dim=-1)

    def read(self, query):
        '''
        query (initial features) : (NxL) x C or N x C -> T x C
        read memory items and get new robust features, 
        while memory items(cluster centers) being fixed 
        '''
        # attn = self.get_attn_score(query, self.mem.detach())  # T x M
        query = self.mem.detach() + query
        self.mem = self.mem.detach()
        attn = None
        return {'output': query, 'attn': attn}

    def update(self, query):
        '''
        Update memory items(cluster centers)
        Fix Encoder parameters (detach)
        query (encoder output features) : (NxL) x C or N x C -> T x C
        '''
        add_mem = query.mean(0)
        
        # update gate : M x C
        update_gate = torch.sigmoid(self.U(self.mem.detach()) + self.W(add_mem))  # M x C
        self.mem = (1 - update_gate) * self.mem.detach() + update_gate * add_mem
        self.mem = self.mem.detach()


    def forward(self, query, istrain=True):
        '''
        query (encoder output features) : N x L x C or N x C
        inter_attn : B x k x N x N
        '''
        s = query.data.shape
        l = len(s)

        query = query.contiguous()
        query = query.view(-1, s[-1])  # N x L x C or N x C -> T x C

        # update memory items(cluster centers), while encoder parameters being fixed
        if istrain:
            self.update(query)
        elif self.config.test_update_mem:
            self.update(query)

        # get new robust features, while memory items(cluster centers) being fixed
        outs = self.read(query)

        read_query, attn = outs['output'], outs['attn']

        if l == 2:
            pass
        elif l == 3:
            read_query = read_query.view(s[0], s[1], 2 * s[2])
            # read_query = read_query.view(s[0], s[1], s[2])
            # attn = attn.view(s[0], s[1], self.n_memory)
        else:
            raise TypeError('Wrong input dimension')
        '''
        output : N x L x 2C or N x 2C
        attn : N x L x M or N x M
        '''
        return {'output': read_query, 'attn': attn, 'memory_init_embedding': self.mem}
