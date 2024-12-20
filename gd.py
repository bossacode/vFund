import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n: int):
        """_summary_

        Args:
            n (int): Number of data.
        """
        super().__init__()
        self.w = nn.Parameter(torch.ones(n) / n, requires_grad=True)
        self.k = nn.Parameter(torch.tensor(n), requires_grad=True)

    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): Tensor containg n time series data of length t. Shape: (n, t).
        """
        k = int(torch.round(self.k.detach()))
        w_k, idx = torch.topk(self.w, k)
        w_k = torch.softmax(w_k)                # shape: (k, )
        x_k = x[idx]                            # shape: (k, t)
        return torch.matmul(w_k, x_k).squeeze() # shape: (t, )