import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from copy import deepcopy
from gd import Model
from utils.loss_fn import LossFn, TdaLossFn, TdaScaledLossFn
from utils.tda import TakensEmbedding, DTMLayer, CubicalPL


class EarlyStopping:
    def __init__(self, patience, threshold):
        """_summary_

        Args:
            patience (int): _description_
            threshold (float): _description_
        """
        assert patience > 0, "Patience must be an integer greater than 0"
        self.patience = patience
        self.threshold = threshold
        self.count = 0
        self.best_loss, self.best_epoch = float("inf"), None

    def stop_training(self, loss, epoch):
        stop, improve = True, True
        diff = self.best_loss - loss
        if diff > self.threshold:   # improvement needs to be above threshold 
            self.count = 0
            self.best_loss, self.best_epoch = loss, epoch
            return not stop, improve
        else:
            self.count += 1
            if self.count > self.patience:  # stop training if no improvement for patience + 1 epochs
                print("-"*30)
                print(f"Best Epoch: {self.best_epoch}")
                print(f"Best Loss: {self.best_loss:>10f}")
                print("-"*30)
                return stop, not improve
            return not stop, not improve


def train(X, y, model, loss_fn, optim, scheduler, early_stopping, log=False):
    i_epoch = 1
    model.train()
    while True:
        y_fitted = model(X)
        loss = loss_fn(y_fitted, y)
        print(f"Training loss: {loss.item():>10f} [Epoch {i_epoch:>3d}]")

        loss.backward()
        optim.step()
        optim.zero_grad()
        scheduler.step(loss)
        
        # early stopping
        stop, improve = early_stopping.stop_training(loss, i_epoch)
        if stop:
            if log:
                wandb.log({"best_training_loss":early_stopping.best_loss})  # log only the best training loss
            break
        elif improve:
            best_state_dict = deepcopy(model.state_dict())  # weights of best model
            y_best_fitted = deepcopy(y_fitted.detach())
        i_epoch += 1
    return y_best_fitted, best_state_dict


def eval(X, y, model, loss_fn, log=False):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
    print(f"Test loss: {loss:>8f} \n\n")
    if log:
        wandb.log({"test_loss":loss})
    return y_pred


def run(X, y, cfg, log=False):
    X, y = X.to(cfg.device), y.to(cfg.device)
    if cfg.window_shift is None:
        cfg.window_shift = cfg.pred_window_size
    train_window_start = 0
    i_window = 1
    nav_pred_hist = [1.0]               # initial net asset value is set to 1.0
    w_hist = [torch.zeros(X.shape[1])]  # all weights are considered to be 0 before first rebalancing
    while train_window_start + cfg.train_window_size + cfg.pred_window_size <= X.shape[0]:
        print(f"Window {i_window}".center(30))
        print("-"*30)

        train_window_end = train_window_start + cfg.train_window_size
        pred_window_end = train_window_end + cfg.pred_window_size

        # set data
        x_train, y_train = X[train_window_start:train_window_end], y[train_window_start:train_window_end]
        x_test, y_test = X[train_window_end:pred_window_end], y[train_window_end:pred_window_end]
        
        # set model
        model = Model(use_k=False).to(cfg.device)
        model(x_train)  # perform dry run to initialize weights
        
        # set loss function, optimizer, lr scheduler, and early stopping
        loss_fn = LossFn(cfg)
        optim = Adam(model.parameters(), cfg.lr)
        scheduler = ReduceLROnPlateau(optim, mode="min", factor=cfg.factor, patience=cfg.sch_patience, threshold=cfg.threshold, threshold_mode="abs")
        early_stopping = EarlyStopping(patience=cfg.es_patience, threshold=cfg.threshold)
        
        # train and save weights
        y_best_fitted, best_state_dict = train(x_train.nan_to_num(0), y_train, model, loss_fn, optim, scheduler, early_stopping, log)
        w = torch.softmax(best_state_dict["w"], dim=0)
        w_hist.append(w)

        # test
        model.load_state_dict(best_state_dict)
        y_pred = eval(x_test.nan_to_num(0), y_test, model, loss_fn, log)

        # calculate transaction cost
        w_prev = w_hist[-2]
        buy = torch.where(w > w_prev, 1., 0.)   # indicator representing whether each stock was bought(=1) or sold/no transaction(=0)
        current_nav = nav_pred_hist[-1]
        transaction_cost = current_nav * torch.sum((cfg.fcb*buy - cfg.fcs*(1-buy)) * (w - w_prev))

        # track net asset value
        current_nav -= transaction_cost     # net asset value after rebalancing
        nav_pred = current_nav * y_pred.exp().cumprod(dim=0)
        nav_pred_hist.extend(nav_pred.cpu().tolist())

        # shift window
        train_window_start += cfg.window_shift
        i_window += 1
    return torch.tensor(nav_pred_hist), torch.stack(w_hist, dim=0).cpu()


def run_tda(X, y, cfg, overestimate=False, log=False):
    X, y = X.to(cfg.device), y.to(cfg.device)
    if cfg.window_shift is None:
        cfg.window_shift = cfg.pred_window_size
    train_window_start = 0
    i_window = 1
    nav_pred_hist = [1.0]               # initial net asset value is set to 1.0
    w_hist = [torch.zeros(X.shape[1])]  # all weights are considered to be 0 before first rebalancing
    # TDA layers
    embed = TakensEmbedding(time_delay=cfg.time_delay, dimension=cfg.dimension, stride=cfg.stride)
    dtm = DTMLayer(cfg.m0, cfg.lims, cfg.size)
    pllay = CubicalPL(cfg.constr, cfg.sublevel, cfg.interval, cfg.steps, cfg.K_max, cfg.dimensions)
    while train_window_start + cfg.train_window_size + cfg.pred_window_size <= X.shape[0]:
        print(f"Window {i_window}".center(30))
        print("-"*30)

        train_window_end = train_window_start + cfg.train_window_size
        pred_window_end = train_window_end + cfg.pred_window_size

        # set data
        x_train, y_train = X[train_window_start:train_window_end], y[train_window_start:train_window_end]
        x_test, y_test = X[train_window_end:pred_window_end], y[train_window_end:pred_window_end]
        pl_train = pllay(dtm(embed(y_train)))
        
        # set model
        model = Model(use_k=False).to(cfg.device)
        model(x_train)  # perform dry run to initialize weights
        
        # set loss function, optimizer, lr scheduler, and early stopping
        w_prev = w_hist[-1]
        train_loss_fn = TdaScaledLossFn(cfg, model, pl_train, w_prev) if overestimate else  TdaLossFn(cfg, pl_train)
        test_loss_fn = LossFn(cfg)
        optim = Adam(model.parameters(), cfg.lr)
        scheduler = ReduceLROnPlateau(optim, mode="min", factor=cfg.factor, patience=cfg.sch_patience, threshold=cfg.threshold, threshold_mode="abs")
        early_stopping = EarlyStopping(patience=cfg.es_patience, threshold=cfg.threshold)
        
        # train and save weights
        y_best_fitted, best_state_dict = train(x_train.nan_to_num(0), y_train, model, train_loss_fn, optim, scheduler, early_stopping, log)
        w = torch.softmax(best_state_dict["w"], dim=0)
        w_hist.append(w)
        
        # test
        model.load_state_dict(best_state_dict)
        y_pred = eval(x_test.nan_to_num(0), y_test, model, test_loss_fn, log)

        # calculate transaction cost
        buy = torch.where(w > w_prev, 1., 0.)   # indicator representing whether each stock was bought(=1) or sold/no transaction(=0)
        current_nav = nav_pred_hist[-1]
        transaction_cost = current_nav * torch.sum((cfg.fcb*buy - cfg.fcs*(1-buy)) * (w - w_prev))

        # track net asset value
        current_nav -= transaction_cost     # net asset value after rebalancing
        nav_pred = current_nav * y_pred.exp().cumprod(dim=0)
        nav_pred_hist.extend(nav_pred.cpu().tolist())

        # shift window
        train_window_start += cfg.window_shift
        i_window += 1
    return torch.tensor(nav_pred_hist), torch.stack(w_hist, dim=0).cpu()