import numpy as np
# import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from math import sqrt
from .mem import MemoryModule

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class Attention_Block(nn.Module):
    def __init__(self, d_model, d_ff=None, n_heads=8, dropout=0.1, activation="relu"):
        super(Attention_Block, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = Attention_Layer(FullAttention, d_model, n_heads=n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross_label = False, x2=None, attn_mask=None, return_attn=False):
        if not cross_label:
            new_x, attn = self.attention(x, x, x,attn_mask=attn_mask)
        else:
            new_x, attn = self.attention(x, x2, x2,attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        
        # 不使用前馈网络
        if return_attn:
            return self.norm2(x), attn
        return self.norm2(x)

        # 使用前馈网络
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        if return_attn:
            return self.norm2(x + y), attn
        return self.norm2(x + y)

class Attention_Layer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(Attention_Layer, self).__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.inner_attention = attention(mask_flag=False,attention_dropout=0.1)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)
        out = self.out_projection(out)
        return out, attn

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        # return V.contiguous()
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class GraphBlock(nn.Module):
    def __init__(self, configs):
        super(GraphBlock, self).__init__()
        self.configs = configs
        # self.att4 = Attention_Block(configs.node_vec_size*2*configs.input_c*configs.input_c, d_ff=None, n_heads=configs.n_heads, dropout=configs.dropout,activation="gelu")
        self.dropout = nn.Dropout(configs.dropout)
        self.activation = F.relu
        self.norm1 = nn.LayerNorm(configs.node_vec_size)
        self.norm2 = nn.LayerNorm(configs.node_vec_size*2)
        self.conv1 = nn.Conv1d(in_channels=configs.node_vec_size, out_channels=configs.node_vec_size, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=configs.node_vec_size, out_channels=1, kernel_size=1)
        self.W = nn.Linear(configs.node_vec_size * 2, configs.node_vec_size * 2) # 如果使用半精度，nn.linear 需要设置输入数据类型（negative：torch 有太多的函数不支持半精度计算，放弃）
        self.W2 = nn.Linear(configs.node_vec_size * 2 * (configs.async_size+1), configs.node_vec_size * 2 * (configs.async_size+1))
        self.a2 = nn.Linear(configs.node_vec_size * 2* (configs.async_size+1),1)
        self.a = nn.Linear(configs.node_vec_size * 2, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.mem_module = MemoryModule(n_memory=configs.n_memory,fea_dim=configs.input_c * configs.input_c, config = configs)

    def forward(self, x, istrain = True):
        B, T, N, D = x.size()
        S0, S = self.configs.async_size,self.configs.async_size+1

        x = x.view(B * T, N, D)

        # lhy async 手写GAT
        wh  = x # T*T，N*N
        if self.configs.async_modeling:
            # 同步相关性
            wh2 = wh.repeat(1, 1, N, 1).view(B, T, N * N, D) # 遍历复制
            wh0 = wh.repeat(1, 1, 1, N).view(B, T, N * N, D) # 先复制再遍历 ！ 很重要
            e0 = torch.cat([wh0, wh2], dim=-1).reshape(B * T, 1, N * N * D*2)
            
            # 异步相关性
            es = []
            for i in range(self.configs.async_size):
                wh1 = torch.cat([wh2[:,self.configs.async_gap*(i+1):,:,:],wh2[:,:self.configs.async_gap*(i+1),:,:]],dim=1)
                ei = torch.cat([wh0, wh1], dim=-1).reshape(B * T, 1, N * N * D*2)
                es.append(ei)
            es = torch.cat(es, dim=1).reshape(B * T, self.configs.async_size, N * N * D*2)
            e3 = torch.cat([e0, es], dim=1).view(B*T, S, N * N * D*2)
            
            # 利用注意力网络融合同步异步相关性
            if self.configs.async_type =="mean":
                # e4 = e3.mean(dim=1).view(B * T, N * N * D*2) + e0.view(B * T, N * N * D*2)
                e4 = es.mean(dim=1).view(B * T, N * N * D*2) + e0.view(B * T, N * N * D*2)
            elif self.configs.async_type =="max":
                # e4 = e3.max(dim=1)[0].view(B * T, N * N * D*2) + e0.view(B * T, N * N * D*2)
                e4 = es.max(dim=1)[0].view(B * T, N * N * D*2) + e0.view(B * T, N * N * D*2)
            elif self.configs.async_type =="cross_attn":
                socre = torch.einsum("bid,bsd->bis",e0,es).softmax(-1)
                e4 = (torch.einsum("bis,bsd->bid",socre,es) + e0).view(B * T, N * N * D*2)
            elif self.configs.async_type =="line":
                e3 = e3.reshape(B*T, S, N * N, D*2).permute(0,2,1,3).reshape(B*T, N * N,S*D*2)
                e3 = self.W2(e3)
                e = self.a2(F.leaky_relu(e3)).reshape(B * T, N, N)
                
            if self.configs.async_type != "line":
                e4 = self.norm2(e4.view(B * T, N * N, D*2))
                e4 = self.W(e4).reshape(B * T, N * N, D*2) # todo: check 此处对效果是否影响
                e = self.a(F.leaky_relu(e4)).view(B * T, N, N)
        
            # todo: 尝试使用cnn 替代line
            # edge = self.W(torch.cat([wh1, wh2], dim=-1).reshape(B * T * T, N * N, -1).permute(0,2,1))
            # e = self.a(F.leaky_relu(edge)).permute(0,2,1).reshape(B * T, T, N, N)     
        else:
            wh2 = wh.repeat(1, 1, N, 1).reshape(B, T, N * N, D) # 首先实现异步，然后实现指标间配对
            wh0 = wh.repeat(1, 1, 1, N).reshape(B, T, N * N, D)
            edge = torch.cat([wh0, wh2], dim=-1)
            edge = self.W(edge)
            e = self.a(F.leaky_relu(edge)).reshape(B * T, N, N)
        # async end 
        attention = e
        attention = F.softmax(attention, dim=-1)
    
        h_prime = torch.einsum("ben,bnd->bed", attention, x)
        y = self.norm1(h_prime + x)
       

        # 可能可以注释
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1)).reshape(B, T, N).permute(0, 2, 1)
        # 可能可以注释 end
        
        graph_queries = e.reshape(B * T, N * N)
        out = y
        if self.configs.train_use_mem == True:
            outputs = self.mem_module(graph_queries,istrain)  # mem out 出来的图是浮点权重，而不是01边，需要从新
            adj, attn = outputs['output'], outputs['attn']  # Bl,NN*2
            memory_adj = self.mem_module.mem.reshape(self.configs.n_memory, -1)
        else:
            adj, attn = graph_queries.detach(), None
            memory_adj = None

        return {"res_with_dim": out,
                "res_adj": adj,
                "memory_adj": memory_adj,
                "cluster_atten": attn
                }
