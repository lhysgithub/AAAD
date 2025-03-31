import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayerOriginal(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True,n_heads=1):
        super(GraphAttentionLayerOriginal, self).__init__()
        # self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features,out_features)
        self.a = nn.Linear(2*out_features,1)
        self.conv1 = nn.Conv1d(in_channels=out_features, out_channels=out_features*4, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=out_features*4, out_channels=out_features, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu
        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = self.W(h) # h: B N d_model     adj: B N N
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        # attention = torch.where(adj > 0.1, e, zero_vec)
        attention = e
        attention = F.softmax(attention, dim=-1)
        # attention = F.dropout(attention, self.dropout)
        # h_prime = torch.matmul(attention, Wh)
        h_prime = torch.einsum("ben,bnd->bed",attention,Wh)
        y = x = self.norm1(h_prime+h)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y)

    def _prepare_attentional_mechanism_input(self, Wh):
        B,N,d_model = Wh.shape
        wh1 = Wh.repeat(1,N,1)
        wh2 = Wh.repeat(N,1,1).reshape(B,N*N,d_model)
        e = self.a(torch.cat([wh1,wh2],dim=-1)).reshape(B,N,N)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayerOriginal(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayerOriginal(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


