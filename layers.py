import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import numpy as np

# Utility function
def from_dense_batch_batch_second(x, batch_idx):
    h = []
    graph_indexes = torch.unique(batch_idx)
    for i in graph_indexes:
        graph_len = len(torch.where(batch_idx==i)[0])
        h.append(x[:graph_len, i])
    return torch.cat(h)

def from_dense_batch(x, batch_idx):
    h = []
    graph_indexes = torch.unique(batch_idx)
    for i in graph_indexes:
        graph_len = len(torch.where(batch_idx==i)[0])
        h.append(x[i, :graph_len])
    return torch.cat(h)


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
        self.n_head = n_head

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

    def forward(self, g, h, device):
        # Save h for residual connection
        h_residual1 = h

        # Multi-head Attention
        # Adjacency Matrix of the batch of graphs, in dense format (with padding if graphs do not have the same number of nodes)
        adj = torch_geometric.utils.to_dense_adj(g.edge_index, g.batch)
        # Attention mask of shape (BatchSize, N_heads, n_nodes, n_nodes)
        attention_mask_sparse = adj.to(dtype=torch.bool).unsqueeze(1).repeat(1, self.n_head, 1, 1).to(device)
        # Graph batch format -> dense sequence format with zero-padding (+ fully connected attention mask)
        x, attention_mask_fully_connected = torch_geometric.utils.to_dense_batch(h, g.batch.to(device))

        # Attention mask for addition format
        attention_mask = torch.nn.functional._canonical_mask(mask=attention_mask_sparse,
                                                            mask_name="attention_mask",
                                                            other_type=None,
                                                            other_name="",
                                                            target_type=torch.float,
                                                            check_other=False,)


        # Self-attention
        x, scores = self.multihead_attention(x, attention_mask=attention_mask)
        # Go back to batch of graph format
        h = from_dense_batch(x, g.batch.to(device))

        # Residual Connection and Normalization
        h = self.normalization1(h+h_residual1)

        # Save h for residual connection
        h_residual2 = h
        # Feed-forward network
        h = self.feedforward(h)

        # Residual Connection and Normalization
        h = self.normalization2(h+h_residual2)

        return h


class TransformerLayerTorchAttention(nn.Module):
    def __init__(self, n_hidden, n_head, n_feedforward, dropout, norm):
        super(TransformerLayerTorchAttention, self).__init__()
        self.n_head = n_head

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

    def forward(self, g, h, device):
        # Save h for residual connection
        h_residual1 = h

        # Multi-head Attention
        # Adjacency Matrix of the batch of graphs, in dense format (with padding if graphs do not have the same number of nodes)
        adj = torch_geometric.utils.to_dense_adj(g.edge_index, g.batch)
        # Attention mask of shape (BatchSize*N_heads, n_nodes, n_nodes)
        attention_mask_sparse = adj.to(dtype=torch.bool).repeat(self.n_head, 1, 1).to(device)
        # Graph batch format -> dense sequence format with zero-padding (+ fully connected attention mask)
        x, attention_mask_fully_connected = torch_geometric.utils.to_dense_batch(h, g.batch.to(device))
        # We want (sequence length, batch size, features)
        x = x.permute(1,0,2)
        # Self-attention
        x, att_weights = self.multihead_attention(x, x, x, attn_mask=attention_mask_sparse)
        # Go back to batch of graph format
        h = from_dense_batch_batch_second(x, g.batch.to(device))

        # Residual Connection and Normalization
        h = self.normalization1(h+h_residual1)

        # Save h for residual connection
        h_residual2 = h
        # Feed-forward network
        h = self.feedforward(h)

        # Residual Connection and Normalization
        h = self.normalization2(h+h_residual2)

        return h


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
        self.n_head = n_head

        # Multi-head Attention
        self.multihead_attention = MultiHeadAttentionEdges(n_hidden, n_head, dropout=dropout)

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

    def forward(self, g, h, e, device):
        # Save h for residual connection
        h_residual1 = h

        # Multi-head Attention
        # Adjacency Matrix of the batch of graphs, in dense format (with padding if graphs do not have the same number of nodes)
        adj = torch_geometric.utils.to_dense_adj(g.edge_index, g.batch)
        # Attention mask of shape (BatchSize*N_heads, n_nodes, n_nodes)
        attention_mask_sparse = adj.to(dtype=torch.bool).unsqueeze(1).repeat(1, self.n_head, 1, 1).to(device)
        # Graph batch format -> dense sequence format with zero-padding (+ fully connected attention mask)
        x, attention_mask_fully_connected = torch_geometric.utils.to_dense_batch(h, g.batch.to(device))

        # Attention mask for addition format
        attention_mask = torch.nn.functional._canonical_mask(mask=attention_mask_sparse,
                                                            mask_name="attention_mask",
                                                            other_type=None,
                                                            other_name="",
                                                            target_type=torch.float,
                                                            check_other=False,)
        # Self-attention
        x, e, scores = self.multihead_attention(x, e, attention_mask=attention_mask)
        # Go back to batch of graph format
        h = from_dense_batch(x, g.batch.to(device))

        # Residual Connection and Normalization
        h = self.normalization1(h+h_residual1)

        # Save h for residual connection
        h_residual2 = h
        # Feed-forward network
        h = self.feedforward(h)

        # Residual Connection and Normalization
        h = self.normalization2(h+h_residual2)

        return h, e