from scipy import sparse as sp
import torch
import torch_geometric
import torch_geometric.transforms as T

def laplacian_pos_encoding(graph, pos_enc_dim):
    laplacian_edge_index, laplacian_edge_attr = torch_geometric.utils.get_laplacian(graph.edge_index, graph.edge_attr, normalization='rw')
    laplacian_sparse = torch_geometric.utils.to_scipy_sparse_matrix(laplacian_edge_index, laplacian_edge_attr)

    EigVal, EigVec = sp.linalg.eigs(laplacian_sparse, k=pos_enc_dim+1, which='SR', tol=1e-2)
    EigVec = EigVec[:, EigVal.argsort()]
    EigVec = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()

    graph.x = torch.cat([graph.x, EigVec], dim=1)
    return graph

class AddLaplacianPosEncoding(T.BaseTransform):
    def __init__(self, pos_enc_dim):
        self.pos_enc_dim = pos_enc_dim
    def __call__(self, graph):
        return laplacian_pos_encoding(graph, self.pos_enc_dim)