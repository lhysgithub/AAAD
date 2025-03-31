import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.pe = torch.zeros((max_len, d_model))
        self.pe.requires_grad = False

        pos = torch.arange(0, max_len).unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2)

        self.pe[:, ::2] = torch.sin((pos / (10000 ** (_2i / d_model))).float())
        self.pe[:, 1::2] = torch.cos((pos / (10000 ** (_2i / d_model))).float())[:, :d_model // 2]

        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, in_dim, d_model):
        super(TokenEmbedding, self).__init__()
        pad = 1 if torch.__version__ >= '1.5.0' else 2
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=d_model, kernel_size=3, padding=pad,
                              padding_mode='circular', bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)).transpose(1, 2)

        return x


class InputEmbedding(nn.Module):
    def __init__(self, input_c, embed_dim, device, dropout=0.0):
        super(InputEmbedding, self).__init__()
        self.device = device
        self.token_embedding = TokenEmbedding(in_dim=1, d_model=embed_dim)
        self.pos_embedding = PositionalEmbedding(d_model=input_c)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        B, L, d = x.size()
        x = self.token_embedding(x.reshape(B * L, d, 1)).reshape(B, L, d, -1) + self.pos_embedding(x).unsqueeze(-1).cuda()

        return self.dropout(x)
