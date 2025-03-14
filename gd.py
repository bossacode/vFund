import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, use_k=False):
        """_summary_

        Args:
            use_k (bool, optional): Whether to use a subset of assets rather than the entire universe. Defaults to False.
        """
        super().__init__()
        self.w = nn.UninitializedParameter()
        if use_k:
            self.k = nn.UninitializedParameter()

    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): Log return of assets in the universe, with each asset corresponding to a single column. Shape: (t, n).

        Returns:
            (torch.Tensor): Predicted log return of portfolio. Shape: (t, ).
        """
        if self.has_uninitialized_params():
            self.initialize_params(x)           # parameter initialization

        if hasattr(self, "k"):
            # use k assets
            k = round(self.k.item())
            print("Number of assets used: ", k)
            w_k, idx = torch.topk(self.w, k)
            w_k = F.softmax(w_k, dim=0)         # shape: (k, )
            x_k = x[:, idx]                     # shape: (t, k)
            y_pred = torch.matmul(x_k, w_k)     # shape: (t, )
        else:
            # use all assets
            w = F.softmax(self.w, dim=0)        # shape: (n, )
            y_pred = torch.matmul(x, w)         # shape: (t, )
        return y_pred

    def has_uninitialized_params(self):
        return any([isinstance(param, nn.UninitializedParameter) for param in self.parameters()])

    def initialize_params(self, x):
        """Parameter initialization.

        Args:
            x (torch.Tensor): Log return of universe, with each asset corresponding to a single column. Shape: (t, n).
        """
        n = x.shape[1]
        w = torch.ones(n, device=self.w.device) * (-torch.inf)
        ind = (~x.isnan()).all(dim=0)   # boolean indicator of columns that have no missing values
        w[ind] = 1/torch.sum(ind)       # initialize with equal weight at selected indices, otherwise -inf weight
        self.w = nn.Parameter(w)        # shape (n, )
        if hasattr(self, "k"):
            self.k = nn.Parameter(torch.tensor(n, dtype=torch.float32, device=self.w.device))