import torch
import torch.nn as nn
from utils.tda import TakensEmbedding, DTMLayer, CubicalPL


class LossFn:
    def __init__(self, mode):
        """_summary_

        Args:
            mode (str): _description_

        Raises:
            KeyError: _description_
        """
        if mode == "L1":
            self.loss_fn = nn.L1Loss()
        elif mode == "L2":
            self.loss_fn = nn.MSELoss()
        else:
            raise KeyError('"mode" must be one of "L1" or "L2".')
    
    def __call__(self, y_pred, y):
        loss = self.loss_fn(y_pred, y)
        return loss


# loss function including MIP regularization
class MipLossFn(LossFn):
    def __init__(self, mode, cfg, model):
        """_summary_

        Args:
            mode (str): _description_
            cfg (_type_): _description_
            model (_type_): _description_

        Raises:
            KeyError: _description_
        """
        super().__init__(mode)
        self.mode = mode
        self.cfg = cfg
        self.model = model
    
    def __call__(self, y_pred, y):
        loss = self.loss_fn(y_pred, y)
        # MIP
        reg = torch.tensor(0) if round(self.model.k.item()) == self.cfg.k else (self.model.k - self.cfg.k)**2
        print(f"{self.mode} Loss:", loss.item(), "Regularization:", self.cfg.lamda * reg.item())
        return loss + (self.cfg.lamda * reg)


# loss function including TDA Loss
class TdaLossFn(LossFn):
    def __init__(self, mode, cfg, pl_target):
        """_summary_

        Args:
            mode (str): _description_
            cfg (_type_): _description_
            pl_target (_type_): _description_

        Raises:
            KeyError: _description_
        """
        super().__init__(mode)
        self.mode = mode
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
        print(f"{self.mode} Loss:", loss.item(), "TDA Loss:", self.cfg.alpha * tda_loss.item())
        return loss + (self.cfg.alpha * tda_loss)


# class TdaMipLossFn(MipLoss):
#     def __init__(self, mode, cfg, model=None):
#         super().__init__(mode, cfg, model)
#         self.embed = TakensEmbedding(
#             time_delay=cfg.time_delay,
#             dimension=cfg.dimension,
#             stride=cfg.stride
#             )
#         ############################################################
#         # change parameters here
#         self.dtm = DTMLayer()
#         self.pl = CubicalPL()
#         ############################################################
    
#     def __call__(self, y_pred, y):
#         loss = self.loss_fn(y_pred, y)

#         # TDA 
#         y_pred_embed = self.embed(y_pred)
#         y_pred_dtm = self.dtm(y_pred_embed)
#         pl_pred = self.pl(y_pred_dtm)
        
#         y_embed = self.embed(y)
#         y_dtm = self.dtm(y_embed)
#         pl = self.pl(y_dtm)

#         tda_loss = self.loss_fn(pl_pred, pl)    ######## need to give weight to this as well

#         # regularization on number of assets
#         if self.model is not None:
#             regularization = torch.tensor(0) if round(self.model.k.item()) == self.cfg.k else self.cfg.lamda * (self.model.k - self.cfg.k)**2
#             print(f"{self.mode} Loss:", loss.item(), "TDA Loss:", tda_loss.item(), "Regularization:", regularization.item())
#             return loss + tda_loss + regularization
#         else:
#             return loss + tda_loss