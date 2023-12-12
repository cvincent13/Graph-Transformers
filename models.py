import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch_geometric

from layers import TransformerLayer, TransformerLayerEdges, TransformerLayerEdgesCustom


# Positional encodings
class LaplacianPositionalEncoding(nn.Module):
    def __init__(self, k, nhid, dropout=0.1):
        super(LaplacianPositionalEncoding, self).__init__()

        # Linear projection
        self.proj = nn.Linear(k, nhid)
        # Dropout for input features
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, precomputed_eigenvectors):
        encoding = self.proj(precomputed_eigenvectors)
        h = h + encoding
        h = self.dropout(h)
        return h
    

class WLPositionalEncoding(nn.Module):
    def __init__(self, k, nhid, dropout=0.1):
        super(WLPositionalEncoding, self).__init__()
        self.nhid = nhid
        # Dropout for input features
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, precomputed_eigenvectors):
        num_nodes = h.shape[0]
        encoding = torch.zeros(num_nodes, self.nhid)
        encoding = encoding.to(h.device)
        wl_code = precomputed_eigenvectors.unsqueeze(1)
        div_term = torch.exp(
                    torch.arange(0, self.nhid, 2).float() * (-np.log(10000.0) / self.nhid)
                )
        div_term = div_term.to(h.device)
        encoding[:, 0::2] = torch.sin(wl_code * div_term)
        encoding[:, 1::2] = torch.cos(wl_code * div_term)
        h = h + encoding
        h = self.dropout(h)
        return h
    
class BothPositionalEncoding(nn.Module):
    def __init__(self, k, nhid, dropout=0.1):
        super(BothPositionalEncoding, self).__init__()
        self.nhid = nhid
        # Embedding to nhid dimension
        self.proj = nn.Linear(k, nhid)
        # Dropout for input features
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, precomputed_eigenvectors):
        laplacian, wl = precomputed_eigenvectors

        lap_encoding = self.proj(laplacian)

        num_nodes = h.shape[0]
        wl_encoding = torch.zeros(num_nodes, self.nhid)
        wl_encoding = wl_encoding.to(h.device)
        wl_code = wl.unsqueeze(1)
        div_term = torch.exp(
                    torch.arange(0, self.nhid, 2).float() * (-np.log(10000.0) / self.nhid)
                )
        div_term = div_term.to(h.device)
        wl_encoding[:, 0::2] = torch.sin(wl_code * div_term)
        wl_encoding[:, 1::2] = torch.cos(wl_code * div_term)

        h = h + wl_encoding + lap_encoding
        h = self.dropout(h)
        return h
    

class NoPositionalEncoding(nn.Module):
    def __init__(self, k, nhid, dropout=0.1):
        super(NoPositionalEncoding, self).__init__()

        # Dropout for input features
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, precomputed_eigenvectors):
        h = self.dropout(h)
        return h


# Utility function
def from_dense_batch(x, batch_idx):
    h = []
    graph_indexes = torch.unique(batch_idx)
    for i in graph_indexes:
        graph_len = len(torch.where(batch_idx==i)[0])
        h.append(x[i, :graph_len])
    return torch.cat(h)

def from_dense_batch_batch_second(x, batch_idx):
    h = []
    graph_indexes = torch.unique(batch_idx)
    for i in graph_indexes:
        graph_len = len(torch.where(batch_idx==i)[0])
        h.append(x[:graph_len, i])
    return torch.cat(h)


# Graph Transformer
class GraphTransformer(nn.Module):
    def __init__(self, n_nodes_input, n_hidden, n_head, n_feedforward, n_layers, input_dropout, dropout, k, pos_encoding, norm):
        super(GraphTransformer, self).__init__()
        """
        n_nodes_input: input size for the embedding
        n_hidden: the hidden dimension of the model
        n_feedforward: dimension for the feedforward network
        n_head: the number of heads in the multiheadattention models
        n_layers: the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        dropout: the dropout value
        """
        self.n_head = n_head
        # Embedding 
        self.embedding = nn.Embedding(n_nodes_input, n_hidden)
        # Positional Encoding
        if pos_encoding == 'laplacian':
            self.pos_encoder = LaplacianPositionalEncoding(k, n_hidden, dropout=input_dropout)
        elif pos_encoding == 'wl':
            self.pos_encoder = WLPositionalEncoding(k, n_hidden, dropout=dropout)
        elif pos_encoding == 'both':
            self.pos_encoder = BothPositionalEncoding(k, n_hidden, dropout=dropout)
        else:
            self.pos_encoder = NoPositionalEncoding(k, n_hidden, dropout=input_dropout)
        #Transformer Block
        self.transformer_block = nn.ModuleList([TransformerLayer(n_hidden, n_head, n_feedforward, dropout, norm) for _ in range(n_layers-1)])
        self.transformer_block.append(TransformerLayer(n_hidden, n_head, n_feedforward, dropout, norm))
        
        self.n_hidden = n_hidden

    def forward(self, g, h, precomputed_eigenvectors=None):
        # Embedding
        h = self.embedding(h)
        # Positional Encoding
        h = self.pos_encoder(h, precomputed_eigenvectors)
        # Transformer Block
        # Adjacency Matrix of the batch of graphs, in dense format (with padding if graphs do not have the same number of nodes)
        adj = torch_geometric.utils.to_dense_adj(g.edge_index, g.batch)
        # Attention mask of shape (BatchSize, N_heads, n_nodes, n_nodes)
        attention_mask = (1.0-adj).unsqueeze(1).repeat(1, self.n_head, 1, 1)*(-100)
        # Graph batch format -> dense sequence format with zero-padding
        x, padding_mask = torch_geometric.utils.to_dense_batch(h, g.batch)
        padding_mask = padding_mask.unsqueeze(1).repeat(1,self.n_head,1).unsqueeze(-1)

        attention_scores = []
        for layer in self.transformer_block:
            x, scores = layer(x, attention_mask, padding_mask)
            attention_scores.append(scores)

        # Go back to batch of graph format
        h = from_dense_batch(x, g.batch)

        return h, attention_scores
    

class GraphTransformerEdges(nn.Module):
    def __init__(self, n_nodes_input, n_edges_input, n_hidden, n_head, n_feedforward, n_layers, input_dropout, dropout, k, pos_encoding, norm):
        super(GraphTransformerEdges, self).__init__()
        """
        n_nodes_input: input size for the embedding
        n_hidden: the hidden dimension of the model
        n_feedforward: dimension for the feedforward network
        n_head: the number of heads in the multiheadattention models
        n_layers: the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        dropout: the dropout value
        """
        self.n_head = n_head
        # Embedding 
        self.embedding = nn.Embedding(n_nodes_input, n_hidden)
        self.embedding_e = nn.Embedding(n_edges_input, n_hidden)
        # Positional Encoding
        if pos_encoding == 'laplacian':
            self.pos_encoder = LaplacianPositionalEncoding(k, n_hidden, dropout=input_dropout)
        elif pos_encoding == 'wl':
            self.pos_encoder = WLPositionalEncoding(k, n_hidden, dropout=dropout)
        elif pos_encoding == 'both':
            self.pos_encoder = BothPositionalEncoding(k, n_hidden, dropout=dropout)
        else:
            self.pos_encoder = NoPositionalEncoding(k, n_hidden, dropout=input_dropout)
        #Transformer Block
        self.transformer_block = nn.ModuleList([TransformerLayerEdges(n_hidden, n_head, n_feedforward, dropout, norm) for _ in range(n_layers-1)])
        self.transformer_block.append(TransformerLayerEdges(n_hidden, n_head, n_feedforward, dropout, norm))
        
        self.n_hidden = n_hidden

    def forward(self, g, h, precomputed_eigenvectors=None):
        # Embedding
        h = self.embedding(h)
        e = g.edge_attr
        e = self.embedding_e(e)
        e = torch_geometric.utils.to_dense_adj(g.edge_index, batch=g.batch, edge_attr=e)
        # Positional Encoding
        h = self.pos_encoder(h, precomputed_eigenvectors)

        # Transformer Block
        # Adjacency Matrix of the batch of graphs, in dense format (with padding if graphs do not have the same number of nodes)
        adj = torch_geometric.utils.to_dense_adj(g.edge_index, g.batch)
        # Attention mask of shape (BatchSize*N_heads, n_nodes, n_nodes)
        attention_mask = (1.0-adj).unsqueeze(1).repeat(1, self.n_head, 1, 1)*(-100)
        # Graph batch format -> dense sequence format with zero-padding
        x, padding_mask = torch_geometric.utils.to_dense_batch(h, g.batch)
        padding_mask = padding_mask.unsqueeze(1).repeat(1,self.n_head,1).unsqueeze(-1)
        
        attention_scores = []
        for layer in self.transformer_block:
            x, e, scores = layer(x, e, attention_mask, padding_mask)
            attention_scores.append(scores)

        # Go back to batch of graph format
        h = from_dense_batch(x, g.batch)

        return h, attention_scores
    
class GraphTransformerEdgesCustom(nn.Module):
    def __init__(self, n_nodes_input, n_edges_input, n_hidden, n_head, n_feedforward, n_layers, input_dropout, dropout, k, pos_encoding, norm):
        super(GraphTransformerEdgesCustom, self).__init__()
        """
        n_nodes_input: input size for the embedding
        n_hidden: the hidden dimension of the model
        n_feedforward: dimension for the feedforward network
        n_head: the number of heads in the multiheadattention models
        n_layers: the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        dropout: the dropout value
        """
        self.n_head = n_head
        # Embedding 
        self.embedding = nn.Embedding(n_nodes_input, n_hidden)
        self.embedding_e = nn.Embedding(n_edges_input, n_hidden)
        # Positional Encoding
        if pos_encoding == 'laplacian':
            self.pos_encoder = LaplacianPositionalEncoding(k, n_hidden, dropout=input_dropout)
        elif pos_encoding == 'wl':
            self.pos_encoder = WLPositionalEncoding(k, n_hidden, dropout=dropout)
        elif pos_encoding == 'both':
            self.pos_encoder = BothPositionalEncoding(k, n_hidden, dropout=dropout)
        else:
            self.pos_encoder = NoPositionalEncoding(k, n_hidden, dropout=input_dropout)
        #Transformer Block
        self.transformer_block = nn.ModuleList([TransformerLayerEdgesCustom(n_hidden, n_head, n_feedforward, dropout, norm) for _ in range(n_layers-1)])
        self.transformer_block.append(TransformerLayerEdgesCustom(n_hidden, n_head, n_feedforward, dropout, norm))
        
        self.n_hidden = n_hidden

    def forward(self, g, h, precomputed_eigenvectors=None):
        # Embedding
        h = self.embedding(h)
        e = g.edge_attr
        e = self.embedding_e(e)
        e = torch_geometric.utils.to_dense_adj(g.edge_index, batch=g.batch, edge_attr=e)
        # Positional Encoding
        h = self.pos_encoder(h, precomputed_eigenvectors)

        # Transformer Block
        # Adjacency Matrix of the batch of graphs, in dense format (with padding if graphs do not have the same number of nodes)
        adj = torch_geometric.utils.to_dense_adj(g.edge_index, g.batch)
        # Attention mask of shape (BatchSize*N_heads, n_nodes, n_nodes)
        attention_mask = (1.0-adj).unsqueeze(1).repeat(1, self.n_head, 1, 1)*(-100)
        # Graph batch format -> dense sequence format with zero-padding
        x, padding_mask = torch_geometric.utils.to_dense_batch(h, g.batch)
        padding_mask = padding_mask.unsqueeze(1).repeat(1,self.n_head,1).unsqueeze(-1)
        
        attention_scores = []
        for layer in self.transformer_block:
            x, scores = layer(x, e, attention_mask, padding_mask)
            attention_scores.append(scores)

        # Go back to batch of graph format
        h = from_dense_batch(x, g.batch)

        return h, attention_scores


# Prediction Heads
class ClassificationHead(nn.Module):
    def __init__(self, n_hidden, n_classes):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(n_hidden, n_hidden//2)
        self.fc2 = nn.Linear(n_hidden//2, n_hidden//4)
        self.outfc = nn.Linear(n_hidden//4, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.outfc(x)
        return y
    

class GraphRegressionHead(nn.Module):
    def __init__(self, n_hidden):
        super(GraphRegressionHead, self).__init__()
        self.fc1 = nn.Linear(n_hidden, n_hidden//2)
        self.fc2 = nn.Linear(n_hidden//2, n_hidden//4)
        self.outfc = nn.Linear(n_hidden//4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.outfc(x).squeeze()
        return y
    

# Models
# Node Level
class NodeClassificationGraphTransformer(nn.Module):
    def __init__(self, n_nodes_input, n_hidden, n_head, n_feedforward, n_layers, n_classes, input_dropout=0.1, dropout=0.5, 
    k=None, pos_encoding='laplacian', norm='layer'):
        super(NodeClassificationGraphTransformer, self).__init__()
        self.n_classes = n_classes
        self.graph_transformer = GraphTransformer(n_nodes_input, n_hidden, n_head, n_feedforward, n_layers, input_dropout, dropout, k, pos_encoding, norm)
        self.classification_head = ClassificationHead(n_hidden, n_classes)

    def forward(self, g, h, precomputed_eigenvectors=None):
        h, attention_scores = self.graph_transformer(g, h, precomputed_eigenvectors=precomputed_eigenvectors)
        h = self.classification_head(h)
        return h, attention_scores
    

# Graph Level
class GraphRegressionGraphTransformer(nn.Module):
    def __init__(self, n_nodes_input, n_hidden=80, n_head=8, n_feedforward=160, n_layers=10, input_dropout=0.1, dropout=0.5, 
    k=None, pos_encoding='laplacian', norm='layer', readout='mean'):
        super(GraphRegressionGraphTransformer, self).__init__()
        self.graph_transformer = GraphTransformer(n_nodes_input, n_hidden, n_head, n_feedforward, n_layers, input_dropout, dropout, k, pos_encoding, norm)
        self.regression_head = GraphRegressionHead(n_hidden)

        if readout == "sum":
            self.readout = torch_geometric.nn.aggr.SumAggregation()
        elif readout == "max":
            self.readout = torch_geometric.nn.aggr.MaxAggregation()
        elif readout == "mean":
            self.readout = torch_geometric.nn.aggr.MeanAggregation()
        else:
            self.readout = readout

    def forward(self, g, h, precomputed_eigenvectors=None):
        h, attention_scores = self.graph_transformer(g, h, precomputed_eigenvectors=precomputed_eigenvectors)

        h_graph = self.readout(h, g.batch)
        
        h_graph = self.regression_head(h_graph)
        return h_graph, attention_scores
    

class GraphRegressionGraphTransformerEdges(nn.Module):
    def __init__(self, n_nodes_input, n_edges_input, n_hidden=80, n_head=8, n_feedforward=160, n_layers=10, input_dropout=0.1, dropout=0.5, 
    k=None, pos_encoding='laplacian', norm='layer', readout='mean'):
        super(GraphRegressionGraphTransformerEdges, self).__init__()
        self.graph_transformer = GraphTransformerEdges(n_nodes_input, n_edges_input, n_hidden, n_head, n_feedforward, n_layers, input_dropout, dropout, k, pos_encoding, norm)
        self.regression_head = GraphRegressionHead(n_hidden)

        if readout == "sum":
            self.readout = torch_geometric.nn.aggr.SumAggregation()
        elif readout == "max":
            self.readout = torch_geometric.nn.aggr.MaxAggregation()
        elif readout == "mean":
            self.readout = torch_geometric.nn.aggr.MeanAggregation()
        else:
            self.readout = readout

    def forward(self, g, h, precomputed_eigenvectors=None):
        h, attention_scores = self.graph_transformer(g, h, precomputed_eigenvectors=precomputed_eigenvectors)

        h_graph = self.readout(h, g.batch)
        
        h_graph = self.regression_head(h_graph)
        return h_graph, attention_scores
    

class GraphRegressionGraphTransformerEdgesCustom(nn.Module):
    def __init__(self, n_nodes_input, n_edges_input, n_hidden=80, n_head=8, n_feedforward=160, n_layers=10, input_dropout=0.1, dropout=0.5, 
    k=None, pos_encoding='laplacian', norm='layer', readout='mean'):
        super(GraphRegressionGraphTransformerEdgesCustom, self).__init__()
        self.graph_transformer = GraphTransformerEdgesCustom(n_nodes_input, n_edges_input, n_hidden, n_head, n_feedforward, n_layers, input_dropout, dropout, k, pos_encoding, norm)
        self.regression_head = GraphRegressionHead(n_hidden)

        if readout == "sum":
            self.readout = torch_geometric.nn.aggr.SumAggregation()
        elif readout == "max":
            self.readout = torch_geometric.nn.aggr.MaxAggregation()
        elif readout == "mean":
            self.readout = torch_geometric.nn.aggr.MeanAggregation()
        else:
            self.readout = readout

    def forward(self, g, h, precomputed_eigenvectors=None):
        h, attention_scores = self.graph_transformer(g, h, precomputed_eigenvectors=precomputed_eigenvectors)

        h_graph = self.readout(h, g.batch)
        
        h_graph = self.regression_head(h_graph)
        return h_graph, attention_scores
    

class GraphRepresentationGraphTransformer(nn.Module):
    def __init__(self, n_nodes_input, n_hidden=80, n_head=8, n_feedforward=160, n_layers=10, input_dropout=0.1, dropout=0.5, 
    k=None, pos_encoding='laplacian', norm='layer', readout='mean'):
        super(GraphRepresentationGraphTransformer, self).__init__()
        self.graph_transformer = GraphTransformer(n_nodes_input, n_hidden, n_head, n_feedforward, n_layers, input_dropout, dropout, k, pos_encoding, norm)

        if readout == "sum":
            self.readout = torch_geometric.nn.aggr.SumAggregation()
        elif readout == "max":
            self.readout = torch_geometric.nn.aggr.MaxAggregation()
        elif readout == "mean":
            self.readout = torch_geometric.nn.aggr.MeanAggregation()
        else:
            self.readout = readout

    def forward(self, g, h, precomputed_eigenvectors=None):
        h, attention_scores = self.graph_transformer(g, h, precomputed_eigenvectors=precomputed_eigenvectors)

        h_graph = self.readout(h, g.batch)
        
        return h_graph, attention_scores
    

class GraphRepresentationGraphTransformerEdges(nn.Module):
    def __init__(self, n_nodes_input, n_edges_input, n_hidden=80, n_head=8, n_feedforward=160, n_layers=10, input_dropout=0.1, dropout=0.5, 
    k=None, pos_encoding='laplacian', norm='layer', readout='mean'):
        super(GraphRepresentationGraphTransformerEdges, self).__init__()
        self.graph_transformer = GraphTransformerEdges(n_nodes_input, n_edges_input, n_hidden, n_head, n_feedforward, n_layers, input_dropout, dropout, k, pos_encoding, norm)

        if readout == "sum":
            self.readout = torch_geometric.nn.aggr.SumAggregation()
        elif readout == "max":
            self.readout = torch_geometric.nn.aggr.MaxAggregation()
        elif readout == "mean":
            self.readout = torch_geometric.nn.aggr.MeanAggregation()
        else:
            self.readout = readout

    def forward(self, g, h, precomputed_eigenvectors=None):
        h, attention_scores = self.graph_transformer(g, h, precomputed_eigenvectors=precomputed_eigenvectors)

        h_graph = self.readout(h, g.batch)
        
        return h_graph, attention_scores