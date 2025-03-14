import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import gudhi as gd
import numpy as np


class TakensEmbedding:
    def __init__(self, time_delay, dimension, stride):
        """_summary_

        Args:
            time_delay (int): _description_
            dimension (int): _description_
            stride (int): _description_
        """
        self.time_delay=time_delay
        self.dimension=dimension
        self.stride=stride

    def __call__(self, x):
        """Transforms a time series into a point cloud using sliding window embedding.

        Args:
            x (torch.Tensor): Tensor containing a single time series. Shape: (t, ).

        Returns:
            (torch.Tensor): Shape: (n, d)
        """
        n = x.shape[0]
        point_cloud = []
        window_start_idx = 0
        while window_start_idx + (self.dimension-1)*self.time_delay < n:
            point = x[[window_start_idx + i*self.time_delay for i in range(self.dimension)]]
            point_cloud.append(point)
            window_start_idx += self.stride
        return torch.stack(point_cloud)

##############################################################################################
# functions and classes for DTM

def make_grid(lims=[[-0.5, 0.5], [-0.5, 0.5]], size=[28, 28]):
    """Creates a 2D grid.

    Args:
        lims (list, optional): Domain of x & y axis. Defaults to [[-0.5, 0.5], [-0.5, 0.5]].
        size (list, optional): Number of discretized points for x & y axis. Defaults to [28, 28].

    Returns:
        grid (torch.Tensor): Grid coordinates. Shape: (H*W, 2).
    """
    assert len(size) == 2 and len(lims) == len(size)
    x_seq = torch.linspace(start=lims[0][0], end=lims[0][1], steps=size[0])
    y_seq = torch.linspace(start=lims[1][1], end=lims[1][0], steps=size[1])
    x_coord, y_coord = torch.meshgrid(x_seq, y_seq, indexing="xy")
    grid = torch.concat([x_coord.reshape(-1, 1), y_coord.reshape(-1, 1)], dim=1)
    return grid


def pc2grid_dist(X, grid, r=2):
    """Calculate distance between all points in point cloud and all grid cooridnates.

    Args:
        X (torch.Tensor): A point cloud. Shape: (N, D).
        grid (torch.Tensor): Grid coordinates. Shape: (H*W, D).
        r (int, optional): r-norm. Defaults to 2.

    Returns:
        dist (torch.Tensor): Distance between all points and grid coordinates. Shape: (H*W, N).
    """
    assert X.shape[-1] == grid.shape[-1]
    X = X.unsqueeze(-2)     # shape: (N, 1, D)
    Y = grid.unsqueeze(0)   # shape: (1, H*W, D)
    if r == 2:
        dist = torch.sqrt(torch.sum((X - Y)**2, dim=-1))    # shape: (N, H*W)
    elif r == 1:
        dist = torch.sum(torch.abs(X - Y), dim=-1)
    else:
        dist = torch.pow(torch.sum((X - Y)**r, dim=-1), 1/r)
    return dist.T


def dtm_using_knn(knn_dist, bound, r=2):
    """DTM using KNN.

    Args:
        knn_dist (torch.Tensor): Distance to k-nearest points for each grid coordinate. Shape: (H*W, k).
        bound (float): Weight bound that corresponds to m0*sum({Wi: i=1...n}).
        r (int, optional): r-norm. Defaults to 2.

    Returns:
        dtm_val (torch.Tensor): DTM value. Shape (B, H*W).
    """
    cum_knn_weight = torch.math.ceil(bound)
    if r == 2:
        r_dist = knn_dist.square()
        cum_dist = torch.cumsum(r_dist, dim=-1)                             # shape: (H*W, k)
        dtm_val = cum_dist[:, -1] + r_dist[:, -1]*(bound - cum_knn_weight)  # shape: (H*W, )
        dtm_val = torch.sqrt(dtm_val/bound)
    elif r == 1:
        r_dist = knn_dist
        cum_dist = torch.cumsum(r_dist, dim=-1)
        dtm_val = cum_dist[:, -1] + r_dist[:, -1]*(bound - cum_knn_weight)
        dtm_val = dtm_val/bound
    else:
        r_dist = knn_dist.pow(r)
        cum_dist = torch.cumsum(r_dist, dim=-1)
        dtm_val = cum_dist[:, -1] + r_dist[:, -1]*(bound - cum_knn_weight)
        dtm_val = torch.pow(dtm_val/bound, 1/r)
    return dtm_val


class DTMLayer(nn.Module):
    def __init__(self, m0=0.01, lims=[[-0.1, 0.1], [-0.1, 0.1]], size=[40, 40], r=2):
        """
        Args:
            m0 (float, optional): Parameter between 0 and 1 that controls locality. Defaults to 0.05.
            lims (list, optional): Domain of x & y axis. Defaults to [[0.0125, 0.9875], [0.0125, 0.9875]].
            size (list, optional): Number of discretized points for x & y axis. Defaults to [40, 40].
            r (int, optional): r-norm. Defaults to 2.
        """
        super().__init__()
        self.grid = make_grid(lims, size)   # shape: (H*W, 2)
        self.m0 = m0
        self.size = size
        self.r = r
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): A point cloud. Shape: (B, N, D).

        Returns:
            dtm_val (torch.Tensor): DTM value. Shape: (H, W).
        """
        bound = self.m0 * x.shape[-2]
        dist = pc2grid_dist(x, self.grid)                           # shape: (H*W, N)
        # knn, find k s.t. k-1 < bound <= k
        k = torch.math.ceil(bound)
        knn_dist, knn_index = dist.topk(k, largest=False, dim=-1)   # shape: (H*W, k)
        # dtm
        dtm_val = dtm_using_knn(knn_dist, bound, self.r)            # shape: (H*W, )
        return dtm_val.view(*self.size)                             # shape: (H, W)

##############################################################################################


##############################################################################################
# PLLay

class CubicalPLFunction(Function):
    @staticmethod
    def forward(ctx, x, func):
        """_summary_

        Args:
            x (torch.Tensor): Shape: (H, W).
            func (function or method): Function that calculates PL and gradient if needed.
            steps (int): Number of discretized points.
            K_max (int): How many landscapes to use per dimension.
            dimensions (list): Homology dimensions to consider.

        Returns:
            torch.Tensor: Shape: (len_dim, K_max, steps).
        """
        backprop = x.requires_grad
        device = x.device
        pl, grad = func(x.cpu().numpy(), backprop)
        if backprop:
            ctx.save_for_backward(torch.from_numpy(grad).to(device).to(torch.float32))
            ctx.input_size = x.shape
        return torch.from_numpy(pl).to(device).to(torch.float32)
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """_summary_

        Args:
            grad_output (torch.Tensor): Gradient w.r.t. to output. Shape: (len_dim, K_max, steps).

        Returns:
            _type_: _description_
        """
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_local, = ctx.saved_tensors                                                 # shape: (len_dim, K_max, steps, H*W)
            grad_input = torch.einsum('...ijk,...ijkl->...l', [grad_output, grad_local])    # shape: (H*W, )
            grad_input = grad_input.view(*ctx.input_size)                                   # shape: (H, W)
        return grad_input, None


class CubicalPL(nn.Module):
    def __init__(self, constr="V", sublevel=True, interval=[0.02, 0.15], steps=32, K_max=2, dimensions=[0, 1]):
        """_summary_

        Args:
            constr (str, optional): Construction used to build cubical complex. One of V or T, corresponding to V-construction and T-construction, respectively. Defaults to V.
            sublevel (bool, optional): Use sublevel set filtration. If False, superlevel set filtration will be used by imposing a sublevel set filtration on negative data. Defaults to True.
            interval (list, optional): Minimum and maximum value of interval to consider. Defaults to [0.02, 0.28].
            steps (int, optional): Number of discretized points. Defaults to 32.
            K_max (int, optional): How many landscapes to use per dimension. Defaults to 2.
            dimensions (list, optional): Homology dimensions to consider. Defaults to [0, 1].
        """
        super().__init__()
        assert constr == "V" or constr == "T", "Construction should be one of V or T."
        self.constr = constr
        self.sublevel = sublevel
        self.t_min, self.t_max = interval
        self.steps = steps
        self.tseq = np.linspace(*interval, steps)
        self.K_max = K_max
        self.dimensions = dimensions
    
    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): Shape: (H, W).

        Returns:
            (torch.Tensor): Shape: (len_dim, K_max, steps).
        """
        x = x if self.sublevel else -x
        return CubicalPLFunction.apply(x, self._cal_pl)

    def _cal_pl(self, x, backprop):
        """_summary_

        Args:
            x (numpy.ndarray): Shape: (H, W).
            backprop (bool): Whether or not the input requires gradient calculation.

        Returns:
            _type_: _description_
        """
        if self.constr == "V":
            cub_cpx = gd.CubicalComplex(vertices=x.T)               # transpose data bc. gudhi uses column-major order
            pd = cub_cpx.persistence(homology_coeff_field=2)        # persistence diagram, list of (dimension, (birth, death))
            location = cub_cpx.vertices_of_persistence_pairs()      # pixel indexes corresponding to birth and death, list containing 2 lists of numpy arrays
        else:
            cub_cpx = gd.CubicalComplex(top_dimensional_cells=x.T)
            pd = cub_cpx.persistence(homology_coeff_field=2)
            location = cub_cpx.cofaces_of_persistence_pairs()

        if location[0]:                                                         # homology feature exists other than 0-dim homology that persists forever
            location_vstack = [np.vstack(location[0]), np.vstack(location[1])]  # first element is birth and death locations for homology features, second element is birth location of 0-dim homology that persists forever
        else:
            location_vstack = [np.zeros((0, 2), dtype=np.int32), np.vstack(location[1])]
        birth_location = np.concatenate((location_vstack[0][:, 0], location_vstack[1][:, 0])).astype(np.int32)
        death_location = location_vstack[0][:, 1].astype(np.int32)

        len_dim = len(self.dimensions)
        len_pd = len(pd)

        pl = np.zeros((len_dim, self.K_max, self.steps))    # shape: (len_dim, K_max, steps)
        if backprop:
            pl_diff_b = np.zeros((len_dim, self.K_max, self.steps, len_pd))
            pl_diff_d = np.zeros((len_dim, self.K_max, self.steps, len_pd))

        for i_dim, dim in enumerate(self.dimensions):
            pd_dim = [pair for pair in pd if pair[0] == dim]
            pd_dim_ids = np.array([i for i, pair in enumerate(pd) if pair[0] == dim])
            len_pd_dim = len(pd_dim)    # number of "dim"-dimensional homology features

            # calculate persistence landscape
            land = np.zeros((max(len_pd_dim, self.K_max), self.steps))
            for d in range(len_pd_dim):
                for t in range(self.steps):
                    land[d, t] = max(min(self.tseq[t] - pd_dim[d][1][0], pd_dim[d][1][1] - self.tseq[t]), 0)
            pl[i_dim] = -np.sort(-land, axis=0)[:self.K_max]
            
            # calculation of gradient only for inputs that require gradient
            if backprop:
                # derivative of landscape functions with regard to persistence diagram: dPL/dPD
                land_idx = np.argsort(-land, axis=0)[:self.K_max]
                # (t > birth) & (t < (birth + death)/2)
                land_diff_b = np.zeros((len_pd_dim, self.steps))
                for d in range(len_pd_dim):
                    land_diff_b[d, :] = np.where((self.tseq > pd_dim[d][1][0]) & (2 * self.tseq < pd_dim[d][1][0] + pd_dim[d][1][1]), -1., 0.)
                # (t < death) & (t > (birth + death)/2)
                land_diff_d = np.zeros((len_pd_dim, self.steps))
                for d in range(len_pd_dim):
                    land_diff_d[d, :] = np.where((self.tseq < pd_dim[d][1][1]) & (2 * self.tseq > pd_dim[d][1][0] + pd_dim[d][1][1]), 1., 0.)

                for d in range(len_pd_dim):
                    pl_diff_b[i_dim, :, :, pd_dim_ids[d]] = np.where(d == land_idx, np.repeat(np.expand_dims(land_diff_b[d, :], axis=0), self.K_max, axis=0), 0)
                for d in range(len_pd_dim):
                    pl_diff_d[i_dim, :, :, pd_dim_ids[d]] = np.where(d == land_idx, np.repeat(np.expand_dims(land_diff_d[d, :], axis=0), self.K_max, axis=0), 0)
        
        # calculation of gradient only for inputs that require gradient
        if backprop:
            # derivative of persistence diagram with regard to input: dPD/dX
            pd_diff_b = np.zeros((len_pd, x.shape[0]*x.shape[1]))
            for i in range(len(birth_location)):
                pd_diff_b[i, birth_location[i]] = 1
            pd_diff_d = np.zeros((len_pd, x.shape[0]*x.shape[1]))
            for i in range(len(death_location)):
                pd_diff_d[i, death_location[i]] = 1	

            if location[0]:
                dimension = np.concatenate((np.hstack([np.repeat(ldim, len(location[0][ldim])) for ldim in range(len(location[0]))]),
                                            np.hstack([np.repeat(ldim, len(location[1][ldim])) for ldim in range(len(location[1]))])))
            else:
                dimension = np.hstack([np.repeat(ldim, len(location[1][ldim])) for ldim in range(len(location[1]))])
            if len(death_location) > 0:
                persistence = np.concatenate((x.reshape(-1)[death_location], np.repeat(np.inf, len(np.vstack(location[1]))))) - x.reshape(-1)[birth_location]
            else:
                persistence = np.repeat(np.inf, len(np.vstack(location[1])))
            order = np.lexsort((-persistence, -dimension))
            grad = np.matmul(pl_diff_b, pd_diff_b[order, :]) + np.matmul(pl_diff_d, pd_diff_d[order, :])  # shape: (len_dim, K_max, steps, H*W)
        else:
            grad = None
        return pl, grad
    
##############################################################################################