import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import InputEmbedding
from .encoder import GraphBlock

class EncoderLayer(nn.Module):
    def __init__(self, attn, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.attn_layer = attn
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''
        out = self.attn_layer(x)
        x = x + self.dropout(out)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)  # N x L x C(=d_model)

# Transformer Encoder
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''
        for attn_layer in self.attn_layers:
            x = attn_layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x

class Decoder(nn.Module):
    def __init__(self, d_model, c_out, d_ff=None, activation='relu', dropout=0.1):
        super(Decoder, self).__init__()
        self.out_linear = nn.Linear(d_model, c_out)
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.decoder_layer1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)

        self.decoder_layer2 = nn.Conv1d(in_channels=d_ff, out_channels=c_out, kernel_size=1)
        self.activation = F.relu if activation == 'relu' else F.gelu
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = nn.BatchNorm1d(d_ff)

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''

        '''
        out : reconstructed output
        '''
        out = self.out_linear(x)
        return out  # N x L x c_out

class TransformerVar(nn.Module):
    # ours: shrink_thres=0.0025
    def __init__(self, config,device=None, memory_initial=False):
        super(TransformerVar, self).__init__()

        self.memory_initial = memory_initial
        self.config = config

        # Encoding
        self.embedding = InputEmbedding(input_c=config.input_c, embed_dim=config.node_vec_size, dropout=config.dropout, device=device)

        # MSGEncoder
        self.encoder2 = GraphBlock(self.config)

        # ours
        self.weak_decoder = Decoder(config.output_c, config.output_c, d_ff=config.d_ff, activation='gelu', dropout=config.dropout)

    def forward(self, x, istrain = True):
        '''
        x (input time window) : N x L x enc_in
        '''
        B, T, N = x.size()
        x = self.embedding(x)  # embeddin : N x L x C(=d_model)
        # out_single_scale = self.encoder(x)
        enc_outputs = self.encoder2(x, istrain)  # BTNK BTN K,B,n_head,N,N

        out = enc_outputs["res_with_dim"]
        adj = enc_outputs["res_adj"]
        memory_adj = enc_outputs["memory_adj"]
        attn = enc_outputs["cluster_atten"]

        out = self.weak_decoder(out.permute(0, 2, 1))

        '''
        out (reconstructed input time window) : N x L x enc_in
        enc_in == c_out
        '''
        return {"out": out,
                "memory_adj": memory_adj,
                "adj": adj,
                "attn": attn,}
