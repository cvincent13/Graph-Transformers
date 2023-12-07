from scipy import sparse as sp
import torch
import torch_geometric
import torch_geometric.transforms as T
import hashlib
import numpy as np
import networkx as nx

def laplacian_pos_encoding(graph, pos_enc_dim):
    laplacian_edge_index, laplacian_edge_attr = torch_geometric.utils.get_laplacian(graph.edge_index, graph.edge_attr, normalization='rw')
    laplacian_sparse = torch_geometric.utils.to_scipy_sparse_matrix(laplacian_edge_index, laplacian_edge_attr)

    EigVal, EigVec = sp.linalg.eigs(laplacian_sparse, k=pos_enc_dim+1, which='SR', tol=1e-2)
    EigVec = EigVec[:, EigVal.argsort()]
    EigVec = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()

    graph.laplacian_eigs = EigVec
    return graph

class AddLaplacianPosEncoding(T.BaseTransform):
    def __init__(self, pos_enc_dim):
        self.pos_enc_dim = pos_enc_dim
    def __call__(self, graph):
        return laplacian_pos_encoding(graph, self.pos_enc_dim)
    


def wl_positional_encoding(g):
    """
        WL-based absolute positional embedding 
        adapted from 
        
        "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
        Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
        https://github.com/jwzhanggy/Graph-Bert
    """
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    adj = torch_geometric.utils.to_dense_adj(g.edge_index, g.batch)[0]
    edge_list = torch.nonzero(adj != 0, as_tuple=False).numpy()
    node_list = np.arange(g.num_nodes)

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
            u1, u2 = pair
            if u1 not in node_neighbor_dict:
                node_neighbor_dict[u1] = {}
            if u2 not in node_neighbor_dict:
                node_neighbor_dict[u2] = {}
            node_neighbor_dict[u1][u2] = 1
            node_neighbor_dict[u2][u1] = 1


    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1
        
    g.wl_encoding = torch.LongTensor(list(node_color_dict.values()))
    return g

class AddWLPosEncoding(T.BaseTransform):
    def __call__(self, graph):
        return wl_positional_encoding(graph)
    

class MakeFullGraph(T.BaseTransform):
    def __call__(self, graph):
        # Full graph with same number of nodes as G
        full_g = torch_geometric.utils.from_networkx(nx.complete_graph(graph.num_nodes))
        # Copy node data of G
        full_g.x = graph.x
        # All edges have a 1 as feature
        full_g.edge_attr = torch.ones(full_g.num_edges)
        # Copy other fields of G
        full_g.y = graph.y
        full_g.laplacian_eigs = graph.laplacian_eigs
        full_g.wl_encoding = graph.wl_encoding

        return full_g
    

class SBMDataset(T.BaseTransform):
    def __call__(self, graph):
        h = graph.x
        h = h.argmax(dim=-1)
        graph.x = h
        return graph
    