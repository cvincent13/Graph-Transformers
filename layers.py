import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import numpy as np


# Layers without edges
class MultiHeadAttention(nn.Module):
    def __init__(self, n_hidden, n_head, dropout):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.n_hidden = n_hidden
        self.dim_head = n_hidden//n_head

        self.Q = nn.Linear(n_hidden, self.dim_head * n_head)
        self.K = nn.Linear(n_hidden, self.dim_head * n_head)
        self.V = nn.Linear(n_hidden, self.dim_head * n_head)

        self.out_dropout = nn.Dropout(dropout)
        self.O = nn.Linear(n_hidden, n_hidden)

    def forward(self, x, attention_mask=None):
        batch_size = x.size(0)

        Q_x = self.Q(x).view(batch_size, -1, self.n_head, self.dim_head).permute((0,2,1,3))
        K_x = self.K(x).view(batch_size, -1, self.n_head, self.dim_head).permute((0,2,1,3))
        V_x = self.K(x).view(batch_size, -1, self.n_head, self.dim_head).permute((0,2,1,3))

        scores = F.softmax(attention_mask + (Q_x @ K_x.transpose(-2,-1))/np.sqrt(self.dim_head), dim=-1)

        attended_values = torch.matmul(scores, V_x)
        attended_values = attended_values.permute((0,2,1,3)).contiguous().view(batch_size, -1, self.n_hidden)

        attended_values = self.out_dropout(attended_values)
        x = self.O(attended_values)

        return x, scores
    

class TransformerLayer(nn.Module):
    def __init__(self, n_hidden, n_head, n_feedforward, dropout, norm):
        super(TransformerLayer, self).__init__()

        # Multi-head Attention
        self.multihead_attention = MultiHeadAttention(n_hidden, n_head, dropout)

        # Feed-forward network
        self.feedforward = nn.Sequential(nn.Linear(n_hidden, n_feedforward),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(n_feedforward, n_hidden))

        # Normalization layers
        if norm == 'layer':
            self.normalization1 = nn.LayerNorm(n_hidden)
            self.normalization2 = nn.LayerNorm(n_hidden)
        elif norm == 'batch':
            self.normalization1 = nn.BatchNorm1d(n_hidden)
            self.normalization2 = nn.BatchNorm1d(n_hidden)
        else:
            print('Accepted values for norm: \'layer\' and \'batch\'. Proceeding without normalization layers.')
            self.normalization1 = nn.Identity()
            self.normalization2 = nn.Identity()

    def forward(self, x, attention_mask):
        # Save h for residual connection
        x_residual1 = x

        # Self-attention
        x, scores = self.multihead_attention(x, attention_mask=attention_mask)
        
        # Residual Connection and Normalization
        if self.norm == 'batch':
            x = self.normalization1((x+x_residual1).permute(0,2,1)).permute(0,2,1)
        else:
            x = self.normalization1(x+x_residual1)

        # Save h for residual connection
        x_residual2 = x
        # Feed-forward network
        x = self.feedforward(x)
        # Residual Connection and Normalization
        if self.norm == 'batch':
            x = self.normalization1((x+x_residual2).permute(0,2,1)).permute(0,2,1)
        else:
            x = self.normalization2(x+x_residual2)

        return x


class TransformerLayerTorchAttention(nn.Module):
    def __init__(self, n_hidden, n_head, n_feedforward, dropout, norm):
        super(TransformerLayerTorchAttention, self).__init__()
        self.norm = norm
        # Multi-head Attention
        self.multihead_attention = torch.nn.MultiheadAttention(n_hidden, n_head, dropout=dropout)

        # Feed-forward network
        self.feedforward = nn.Sequential(nn.Linear(n_hidden, n_feedforward),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(n_feedforward, n_hidden))

        # Normalization layers
        if norm == 'layer':
            self.normalization1 = nn.LayerNorm(n_hidden)
            self.normalization2 = nn.LayerNorm(n_hidden)
        elif norm == 'batch':
            self.normalization1 = nn.BatchNorm1d(n_hidden)
            self.normalization2 = nn.BatchNorm1d(n_hidden)
        else:
            print('Accepted values for norm: \'layer\' and \'batch\'. Proceeding without normalization layers.')
            self.normalization1 = nn.Identity()
            self.normalization2 = nn.Identity()

    def forward(self, x, attention_mask):
        # Save h for residual connection
        x_residual1 = x

        # Multi-head Attention
        # Self-attention
        x, att_weights = self.multihead_attention(x, x, x, attn_mask=attention_mask)

        # Residual Connection and Normalization
        if self.norm == 'batch':
            x = self.normalization1((x+x_residual1).permute(0,2,1)).permute(0,2,1)
        else:
            x = self.normalization1(x+x_residual1)

        # Save h for residual connection
        x_residual2 = x
        # Feed-forward network
        x = self.feedforward(x)

        # Residual Connection and Normalization
        if self.norm == 'batch':
            x = self.normalization2((x+x_residual2).permute(0,2,1)).permute(0,2,1)
        else:
            x = self.normalization2(x+x_residual2)

        return x


# Layers with Edges
class MultiHeadAttentionEdges(nn.Module):
    def __init__(self, n_hidden, n_head, dropout):
        super(MultiHeadAttentionEdges, self).__init__()
        self.n_head = n_head
        self.n_hidden = n_hidden
        self.dim_head = n_hidden//n_head

        self.Q = nn.Linear(n_hidden, self.dim_head * n_head)
        self.K = nn.Linear(n_hidden, self.dim_head * n_head)
        self.V = nn.Linear(n_hidden, self.dim_head * n_head)
        self.E = nn.Linear(n_hidden, self.dim_head * n_head)

        self.out_dropout = nn.Dropout(dropout)
        self.Oh = nn.Linear(n_hidden, n_hidden)
        self.Oe = nn.Linear(n_hidden, n_hidden)

    def forward(self, x, e, attention_mask=None):
        batch_size = x.size(0)
        edge_size = e.size(1)

        Q_x = self.Q(x).view(batch_size, -1, self.n_head, self.dim_head).permute((0,2,1,3))
        K_x = self.K(x).view(batch_size, -1, self.n_head, self.dim_head).permute((0,2,1,3))
        V_x = self.K(x).view(batch_size, -1, self.n_head, self.dim_head).permute((0,2,1,3))
        E_e = self.E(e).view(batch_size, edge_size, edge_size, self.n_head, self.dim_head).permute((0,3,1,2,4))


        intermediate_scores = (Q_x[:,:,:, np.newaxis, :] * K_x[:,:,np.newaxis, :, :] * E_e)/np.sqrt(self.dim_head)
        scores = F.softmax(attention_mask + intermediate_scores.sum(-1), dim=-1)

        e = intermediate_scores.permute((0,2,3,1,4)).contiguous().view(batch_size, edge_size, edge_size, self.n_hidden)
        e = self.out_dropout(e)
        e = self.Oe(e)

        attended_values = torch.matmul(scores, V_x)
        attended_values = attended_values.permute((0,2,1,3)).contiguous().view(batch_size, -1, self.n_hidden)
        attended_values = self.out_dropout(attended_values)
        x = self.Oh(attended_values)

        return x, e, scores



class TransformerLayerEdges(nn.Module):
    def __init__(self, n_hidden, n_head, n_feedforward, dropout, norm):
        super(TransformerLayerEdges, self).__init__()
        self.n_hidden = n_hidden
        self.norm = norm
        # Multi-head Attention
        self.multihead_attention = MultiHeadAttentionEdges(n_hidden, n_head, dropout=dropout)

        # Feed-forward network
        self.feedforward_x = nn.Sequential(nn.Linear(n_hidden, n_feedforward),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(n_feedforward, n_hidden))
        
        self.feedforward_e = nn.Sequential(nn.Linear(n_hidden, n_feedforward),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(n_feedforward, n_hidden))

        # Normalization layers
        if norm == 'layer':
            self.normalization1_x = nn.LayerNorm(n_hidden)
            self.normalization2_x = nn.LayerNorm(n_hidden)
            self.normalization1_e = nn.LayerNorm(n_hidden)
            self.normalization2_e = nn.LayerNorm(n_hidden)
        elif norm == 'batch':
            self.normalization1_x = nn.BatchNorm1d(n_hidden)
            self.normalization2_x = nn.BatchNorm1d(n_hidden)
            self.normalization1_e = nn.BatchNorm2d(n_hidden)
            self.normalization2_e = nn.BatchNorm2d(n_hidden)

        else:
            print('Accepted values for norm: \'layer\' and \'batch\'. Proceeding without normalization layers.')
            self.normalization1_x = nn.Identity()
            self.normalization2_x = nn.Identity()
            self.normalization1_e = nn.Identity()
            self.normalization2_e = nn.Identity()

    def forward(self, x, e, attention_mask):
        batch_size = x.size(0)
        # Save x, e for residual connection
        x_residual1 = x
        e_residual1 = e

        # Multi-head Attention
        
        # Self-attention
        x, e, scores = self.multihead_attention(x, e, attention_mask=attention_mask)
        
        # Residual Connection and Normalization
        if self.norm == 'batch':
            x = self.normalization1_x((x+x_residual1).permute(0,2,1)).permute(0,2,1)
            e = self.normalization1_e((e+e_residual1).permute(0,3,1,2)).permute(0,2,3,1)
        else:
            x = self.normalization1_x(x+x_residual1)
            e = self.normalization1_e(e+e_residual1)

        # Save x, e for residual connection
        x_residual2 = x
        e_residual2 = e
        # Feed-forward network
        x = self.feedforward_x(x)
        e = self.feedforward_e(e)

        # Residual Connection and Normalization
        if self.norm == 'batch':
            x = self.normalization2_x((x+x_residual2).permute(0,2,1)).permute(0,2,1)
            e = self.normalization1_e((e+e_residual2).permute(0,3,1,2)).permute(0,2,3,1)
        else:
            x = self.normalization1_x(x+x_residual2)
            e = self.normalization1_e(e+e_residual2)

        return x, e