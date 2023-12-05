import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric

from layers import TransformerLayer, TransformerLayerEdges


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

        # Embedding to nhid dimension
        self.proj = nn.Embedding(k, nhid)
        # Dropout for input features
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, precomputed_eigenvectors):
        encoding = self.proj(precomputed_eigenvectors)
        h = h + encoding
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


# Graph Transformer
class GraphTransformer(nn.Module):
    def __init__(self, n_nodes_input, n_hidden, n_head, n_feedforward, n_layers, input_dropout, dropout, k, norm):
        super(GraphTransformer, self).__init__()
        """
        n_nodes_input: input size for the embedding
        n_hidden: the hidden dimension of the model
        n_feedforward: dimension for the feedforward network
        n_head: the number of heads in the multiheadattention models
        n_layers: the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        dropout: the dropout value
        """
        # Embedding 
        self.embedding = nn.Embedding(n_nodes_input, n_hidden)
        # Positional Encoding
        self.pos_encoder = LaplacianPositionalEncoding(k, n_hidden, dropout=input_dropout)
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
        for layer in self.transformer_block:
            h = layer(g, h)

        return h
    

class GraphTransformerEdges(nn.Module):
    def __init__(self, n_nodes_input, n_edges_input, n_hidden, n_head, n_feedforward, n_layers, input_dropout, dropout, k, norm):
        super(GraphTransformerEdges, self).__init__()
        """
        n_nodes_input: input size for the embedding
        n_hidden: the hidden dimension of the model
        n_feedforward: dimension for the feedforward network
        n_head: the number of heads in the multiheadattention models
        n_layers: the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        dropout: the dropout value
        """
        # Embedding 
        self.embedding = nn.Embedding(n_nodes_input, n_hidden)
        self.embedding_e = nn.Embedding(n_edges_input, n_hidden)
        # Positional Encoding
        self.pos_encoder = LaplacianPositionalEncoding(k, n_hidden, dropout=input_dropout)
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
        for layer in self.transformer_block:
            h, e = layer(g, h, e)

        return h


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
class NodeClassificationGraphTransformer(nn.Module):
    def __init__(self, n_nodes_input, n_hidden, n_head, n_feedforward, n_layers, n_classes, input_dropout=0.1, dropout=0.5, k=None, norm='layer'):
        super(NodeClassificationGraphTransformer, self).__init__()
        self.graph_transformer = GraphTransformer(n_nodes_input, n_hidden, n_head, n_feedforward, n_layers, input_dropout, dropout, k, norm)
        self.classification_head = ClassificationHead(n_hidden, n_classes)

    def forward(self, g, h, precomputed_eigenvectors=None):
        h = self.graph_transformer(g, h, precomputed_eigenvectors=precomputed_eigenvectors)
        h = self.classification_head(h)
        return h
    

class GraphRegressionGraphTransformer(nn.Module):
    def __init__(self, n_nodes_input, n_hidden=80, n_head=8, n_feedforward=160, n_layers=10, input_dropout=0.1, dropout=0.5, k=None, norm='layer', readout='mean'):
        super(GraphRegressionGraphTransformer, self).__init__()
        self.graph_transformer = GraphTransformer(n_nodes_input, n_hidden, n_head, n_feedforward, n_layers, input_dropout, dropout, k, norm)
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
        h = self.graph_transformer(g, h, precomputed_eigenvectors=precomputed_eigenvectors)

        h_graph = self.readout(h, g.batch)
        
        h_graph = self.regression_head(h_graph)
        return h_graph
    

class GraphRegressionGraphTransformerEdges(nn.Module):
    def __init__(self, n_nodes_input, n_edges_input, n_hidden=80, n_head=8, n_feedforward=160, n_layers=10, input_dropout=0.1, dropout=0.5, k=None, norm='layer', readout='mean'):
        super(GraphRegressionGraphTransformerEdges, self).__init__()
        self.graph_transformer = GraphTransformerEdges(n_nodes_input, n_edges_input, n_hidden, n_head, n_feedforward, n_layers, input_dropout, dropout, k, norm)
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
        h = self.graph_transformer(g, h, precomputed_eigenvectors=precomputed_eigenvectors)

        h_graph = self.readout(h, g.batch)
        
        h_graph = self.regression_head(h_graph)
        return h_graph