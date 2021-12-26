import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda')

class LaplacianLoss(nn.Module):
    def __init__(self, vertices_number, faces, average=False):
        super(LaplacianLoss, self).__init__()
        self.nv = vertices_number
        # when faces are represented as a tensor
        # self.nf = faces.size(0)
        # when faces are represented as an array
        self.nf = faces.shape[0]
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = -1
        laplacian[faces[:, 1], faces[:, 0]] = -1
        laplacian[faces[:, 1], faces[:, 2]] = -1
        laplacian[faces[:, 2], faces[:, 1]] = -1
        laplacian[faces[:, 2], faces[:, 0]] = -1
        laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(self.nv):
            if laplacian[i, i] != 0:
                laplacian[i, :] /= laplacian[i, i]

        self.register_buffer('laplacian', torch.from_numpy(laplacian).to(device))

    def forward(self, x):
        batch_size = x.size(0)
        # x = x.cpu()
        x = torch.matmul(self.laplacian, x)
        # x = x.to(device)
        dims = tuple(range(x.ndimension())[1:])
        x = x.pow(2).sum(dims)
        x = x / self.nv
        if self.average:
            return x.sum() / batch_size
        else:
            return x