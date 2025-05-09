import torch
import torch.nn as nn
from utils.tda import TakensEmbedding, DTMLayer, CubicalPL


class LossFn:
    def __init__(self, cfg):
        if cfg.loss_fn == "L1":
            self.loss_fn = nn.L1Loss()
        elif cfg.loss_fn == "L2":
            self.loss_fn = nn.MSELoss()
        else:
            raise KeyError('"loss_fn" must be one of "L1" or "L2".')
    
    def __call__(self, y_pred, y):
        loss = self.loss_fn(y_pred, y)
        return loss


# loss function including MIP regularization
# class MipLossFn(LossFn):
#     def __init__(self, cfg, model):
#         """_summary_

#         Args:
#             cfg (_type_): _description_
#             model (_type_): _description_

#         Raises:
#             KeyError: _description_
#         """
#         super().__init__(cfg)
#         self.cfg = cfg
#         self.model = model
    
#     def __call__(self, y_pred, y):
#         loss = self.loss_fn(y_pred, y)
#         # MIP
#         reg = torch.tensor(0) if round(self.model.k.item()) == self.cfg.k else (self.model.k - self.cfg.k)**2
#         print(f"{self.cfg.loss_fn} Loss:", loss.item(), "Regularization:", self.cfg.lamda * reg.item())
#         return loss + (self.cfg.lamda * reg)


# loss function with TDA regularization
class TdaLossFn(LossFn):
    def __init__(self, cfg, pl_target):
        super().__init__(cfg)
        self.cfg = cfg
        self.embed = TakensEmbedding(time_delay=cfg.time_delay, dimension=cfg.dimension, stride=cfg.stride)
        self.dtm = DTMLayer(cfg.m0, cfg.lims, cfg.size)
        self.pllay = CubicalPL(cfg.constr, cfg.sublevel, cfg.interval, cfg.steps, cfg.K_max, cfg.dimensions)
        self.pl_target = pl_target
    
    def __call__(self, y_pred, y):
        loss = self.loss_fn(y_pred, y)
        # TDA
        pl_pred = self.pllay(self.dtm(self.embed(y_pred)))
        tda_loss = self.loss_fn(pl_pred, self.pl_target)
        print(f"{self.cfg.loss_fn} Loss:", loss.item(), "TDA Loss:", self.cfg.alpha * tda_loss.item())
        return loss + (self.cfg.alpha * tda_loss)


# loss function with TDA regularization and scaling to complement transaction cost
class TdaScaledLossFn(LossFn):
    def __init__(self, cfg, model, pl_target, w_prev):
        super().__init__(cfg)
        self.cfg = cfg
        self.model = model
        self.embed = TakensEmbedding(time_delay=cfg.time_delay, dimension=cfg.dimension, stride=cfg.stride)
        self.dtm = DTMLayer(cfg.m0, cfg.lims, cfg.size)
        self.pllay = CubicalPL(cfg.constr, cfg.sublevel, cfg.interval, cfg.steps, cfg.K_max, cfg.dimensions)
        self.pl_target = pl_target
        self.w_prev = w_prev
    
    def __call__(self, y_pred, y):
        w = torch.softmax(self.model.w, dim=0)
        buy = torch.where(w > self.w_prev, 1., 0.)  # indicator representing whether each stock was bought(=1) or sold/no transaction(=0)
        transaction_prop = torch.sum((self.cfg.fcb*buy - self.cfg.fcs*(1-buy)) * (w - self.w_prev)) # proportion of transaction cost in total value of our porfolio, i.e., G_t / C_t
        ##########################################################################################################
        # use double of actual transaction cost
        # transaction_prop = torch.sum((2*self.cfg.fcb*buy - 2*self.cfg.fcs*(1-buy)) * (w - self.w_prev))  # proportion of transaction cost in total value of our porfolio, i.e., G_t / C_t
        ##########################################################################################################
        scale = torch.pow(1/(1-transaction_prop), 1/self.cfg.pred_window_size)   # scale to overestimate y
        loss = self.loss_fn(y_pred, torch.log(scale) + y)
        # TDA
        pl_pred = self.pllay(self.dtm(self.embed(y_pred)))
        tda_loss = self.loss_fn(pl_pred, self.pl_target)
        print(f"{self.cfg.loss_fn} Loss:", loss.item(), "TDA Loss:", self.cfg.alpha * tda_loss.item())
        return loss + (self.cfg.alpha * tda_loss)