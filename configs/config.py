import torch
from dataclasses import dataclass, field, asdict
from typing import List

@dataclass
class Cfg:
    # training parameters
    train_window_size:int = 52
    pred_window_size:int = 8
    window_shift:int = 8
    lr:float = 0.1
    factor:float = 0.1      # factor by which the learning rate will be reduced
    sch_patience:int = 5    # learning rate scheduler patience
    es_patience:int = 15    # early stopping patience
    threshold:float = 1e-7  # threshold for measuring the new optimum, to only focus on significant changes.
    mode:str = "L2"         # loss function to use
    device:str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # model initialization and regularization parameters
    use_k:bool = False
    # k:int = 10             # number of assets to use
    # lamda:float = 0.1      # weight on regularization

    def dict(self):
        return asdict(self)


@dataclass
class TDACfg:
    # training parameters
    train_window_size:int = 52
    pred_window_size:int = 8
    window_shift:int = 8
    lr:float = 0.1
    factor:float = 0.1
    sch_patience:int = 5
    es_patience:int = 15
    threshold:float = 1e-7
    mode:str = "L2"
    device:str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # model initialization and regularization parameters
    use_k:bool = False
    # k:int = 10             # number of assets to use
    # lamda:float = 0.1      # weight on regularization
    
    # TDA parameters
    # Takens Embedding
    time_delay:int = 1
    dimension:int = 2
    stride:int = 1
    # # DTM
    m0:float = 0.01
    lims:List = field(default_factory=lambda: [[-0.1, 0.1], [-0.1, 0.1]])
    size:List = field(default_factory=lambda: [40, 40])

    # PLLay
    constr:str = "V"
    sublevel:bool = True
    interval:List = field(default_factory=lambda: [0., 0.02])
    steps:int = 64
    K_max:int = 2
    dimensions:List = field(default_factory=lambda: [0, 1])

    alpha:float = 100   # weight on TDA loss

    def dict(self):
        return asdict(self)