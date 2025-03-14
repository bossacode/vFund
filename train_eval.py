import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from gd import Model
from utils.loss_fn import LossFn, MipLossFn, TdaLossFn
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


def train(X, y, model, loss_fn, optim1, optim2, scheduler, early_stopping, log=False):
    i_epoch = 1
    model.train()
    while True:
        y_fitted = model(X)
        loss = loss_fn(y_fitted, y)
        print(f"Training loss: {loss.item():>10f} [Epoch {i_epoch:>3d}]")

        loss.backward()
        optim1.step()
        optim1.zero_grad()
        if optim2 is not None:
            optim2.step()
            optim2.zero_grad()

        if scheduler:
            scheduler.step(loss)
        
        # early stopping
        stop, improve = early_stopping.stop_training(loss, i_epoch)
        if stop:
            if log:
                wandb.log({"best_training_loss":early_stopping.best_loss})  # log only the best training loss
            model.load_state_dict(model_state_dict)
            break
        elif improve:
            model_state_dict = model.state_dict()
            y_best_fitted = y_fitted.detach()
        i_epoch += 1
    return y_best_fitted


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
    train_hist, test_hist, w_hist = [], [], []
    while train_window_start + cfg.train_window_size + cfg.pred_window_size <= X.shape[0]:
        print(f"Window {i_window}".center(30))
        print("-"*30)

        train_window_end = train_window_start + cfg.train_window_size
        pred_window_end = train_window_end + cfg.pred_window_size

        # set data
        x_train = X[train_window_start:train_window_end]
        y_train = y[train_window_start:train_window_end]
        x_test = X[train_window_end:pred_window_end]
        y_test = y[train_window_end:pred_window_end]
        
        # set model
        model = Model(cfg.use_k).to(cfg.device)
        model(x_train)  # perform dry run to initialize weights
        
        # set loss function and optimizer
        if cfg.use_k:
            loss_fn = MipLossFn(cfg.mode, cfg, model)
            params_list = list(model.parameters())
            optim1 = Adam([params_list[0]], cfg.lr) # optimizer for w
            optim2 = SGD([params_list[1]], cfg.lr)  # optimizer for k
        else:
            loss_fn = LossFn(cfg.mode)
            optim1 = Adam(model.parameters(), cfg.lr)
            optim2 = None
        
        # set learning rate scheduler
        try:
            scheduler = ReduceLROnPlateau(optim1, mode="min", factor=cfg.factor, patience=cfg.sch_patience, threshold=cfg.threshold, threshold_mode="abs")
        except:
            scheduler = None
            print("No learning rate scheduler!\n")
        
        # set early stopping
        early_stopping = EarlyStopping(patience=cfg.es_patience, threshold=cfg.threshold)
        
        # train
        fitted = train(x_train.nan_to_num(0), y_train, model, loss_fn, optim1, optim2, scheduler, early_stopping, log)
        train_hist.append((fitted, y_train))

        # test
        pred = eval(x_test.nan_to_num(0), y_test, model, loss_fn, log)
        test_hist.append((pred, y_test))

        # save weight
        if cfg.use_k:
            w = torch.zeros_like(model.w)
            k = round(model.k.item())
            w_k, idx = torch.topk(model.w.detach(), k)
            w_k = torch.softmax(w_k, dim=0)
            w[idx] = w_k
        else:
            w = torch.softmax(model.w.detach(), dim=0)
        w_hist.append(w)

        # shift window
        train_window_start += cfg.window_shift
        i_window += 1
    return train_hist, test_hist, w_hist


def run_tda(X, y, cfg, log=False):
    X, y = X.to(cfg.device), y.to(cfg.device)
    if cfg.window_shift is None:
        cfg.window_shift = cfg.pred_window_size
    train_window_start = 0
    i_window = 1
    train_hist, test_hist, w_hist = [], [], []
    embed = TakensEmbedding(time_delay=cfg.time_delay, dimension=cfg.dimension, stride=cfg.stride)
    dtm = DTMLayer(cfg.m0, cfg.lims, cfg.size)
    pllay = CubicalPL(cfg.constr, cfg.sublevel, cfg.interval, cfg.steps, cfg.K_max, cfg.dimensions)
    while train_window_start + cfg.train_window_size + cfg.pred_window_size <= X.shape[0]:
        print(f"Window {i_window}".center(30))
        print("-"*30)

        train_window_end = train_window_start + cfg.train_window_size
        pred_window_end = train_window_end + cfg.pred_window_size

        # set data
        x_train = X[train_window_start:train_window_end]
        y_train = y[train_window_start:train_window_end]
        pl_train = pllay(dtm(embed(y_train)))
        x_test = X[train_window_end:pred_window_end]
        y_test = y[train_window_end:pred_window_end]
        # pl_test = pllay(dtm(embed(y_test)))
        
        # set model
        model = Model(cfg.use_k).to(cfg.device)
        model(x_train)  # perform dry run to initialize weights
        
        # set loss function and optimizer
        if cfg.use_k:
            raise NotImplementedError
        else:
            train_loss_fn = TdaLossFn(cfg.mode, cfg, pl_train)
            # test_loss_fn = TdaLossFn(cfg.mode, cfg, pl_test)
            test_loss_fn = LossFn(cfg.mode)
            optim1 = Adam(model.parameters(), cfg.lr)
            optim2 = None
        
        # set learning rate scheduler
        try:
            scheduler = ReduceLROnPlateau(optim1, mode="min", factor=cfg.factor, patience=cfg.sch_patience, threshold=cfg.threshold, threshold_mode="abs")
        except:
            scheduler = None
            print("No learning rate scheduler!\n")
        
        # set early stopping
        early_stopping = EarlyStopping(patience=cfg.es_patience, threshold=cfg.threshold)
        
        # train
        fitted = train(x_train.nan_to_num(0), y_train, model, train_loss_fn, optim1, optim2, scheduler, early_stopping, log)
        train_hist.append((fitted, y_train))

        # test
        pred = eval(x_test.nan_to_num(0), y_test, model, test_loss_fn, log)
        test_hist.append((pred, y_test))

        # save weight
        if cfg.use_k:
            w = torch.zeros_like(model.w)
            k = round(model.k.item())
            w_k, idx = torch.topk(model.w.detach(), k)
            w_k = torch.softmax(w_k, dim=0)
            w[idx] = w_k
        else:
            w = torch.softmax(model.w.detach(), dim=0)
        w_hist.append(w)

        # shift window
        train_window_start += cfg.window_shift
        i_window += 1
    return train_hist, test_hist, w_hist