import torch
import pandas as pd
from configs.config import QPcfg, TDAcfg
from train_eval import run, run_tda
from utils.visualize import viz_logret_nav, viz_weights
from utils.linear_reg import reg_fit
import os
import wandb

torch.manual_seed(123)

# Universe
df_return = pd.read_csv("./data/universe_ret.csv", index_col=0)
drop_cols = df_return.columns[(df_return.abs() > 0.3).any()]    # drop columns s.t. have absolute value of returns larger than 0.3
x_return = torch.from_numpy(df_return.drop(columns=drop_cols).to_numpy("float32"))
x_log_return = torch.log(x_return + 1)

# Target
y_returns = pd.read_csv("./data/target_ret.csv", index_col=0)

# Parameter Search Domain
window_params = [104, 12, 12]   # tuple of [train_window_size, pred_window_size, window_shift]
embedding_params = (            # tuple of [time_delay, stride]
    [2, 1],
    [5, 1],
    [10, 1]
    ) 
lims_params = (
    [[-0.15, 0.15], [-0.15, 0.15]],
    [[-0.2, 0.2], [-0.2, 0.2]]
    )
interval_params = (
    [0., 0.02],
    [0., 0.03]
    )


# function for transforming sequence of net asset values to sequence of log returns
def nav2logret(nav):
    log_returns = torch.log(nav[1:] / nav[:-1])
    return log_returns


if __name__=="__main__":
    for target_fund in y_returns.columns:
        y_return = torch.from_numpy(y_returns.loc[:, target_fund].to_numpy("float32"))
        y_log_return = torch.log(y_return + 1)
        train_window_size, pred_window_size, window_shift = window_params
        # QP
        cfg = QPcfg(train_window_size, pred_window_size, window_shift)
        nav_pred, w_hist = run(x_log_return, y_log_return, cfg, log=False)
        
        # compute predicted & true log return values and true net asset value for entire prediction period
        log_return_pred = nav2logret(nav_pred)
        log_return_true = y_log_return[cfg.train_window_size:(cfg.train_window_size+len(log_return_pred))]
        nav_true = torch.concat([torch.tensor([nav_pred[0]]), nav_pred[0] * log_return_true.exp().cumprod(dim=0)])  # nav_pred[0] contains initial net asset value

        # save nav result
        dir_path = f"./{target_fund}/rm/{cfg.fcb}/"
        os.makedirs(dir_path, exist_ok=True)
        df = pd.DataFrame(torch.stack([nav_pred, nav_true]).T.numpy())
        df.to_csv(dir_path + "rm.csv")

        # TDA
        for time_delay, stride in embedding_params:
            for lims in lims_params:
                for interval in interval_params:
                    
                    gamma = torch.rand(1).item()    # sample from Unif(0, 1)
                    
                    # TDA without overestimating
                    cfg = TDAcfg(train_window_size, pred_window_size, window_shift, time_delay=time_delay, stride=stride, lims=lims, interval=interval, gamma=gamma)
                    nav_pred, w_hist = run_tda(x_log_return, y_log_return, cfg, overestimate=False, log=False)

                    # compute predicted & true log return values and true net asset value for entire prediction period
                    log_return_pred = nav2logret(nav_pred)
                    log_return_true = y_log_return[cfg.train_window_size:(cfg.train_window_size+len(log_return_pred))]
                    nav_true = torch.concat([torch.tensor([nav_pred[0]]), nav_pred[0] * log_return_true.exp().cumprod(dim=0)])  # nav_pred[0] contains initial net asset value
                    
                    # save nav result
                    dir_path = f"./{target_fund}/tr/{cfg.fcb}/"
                    os.makedirs(dir_path, exist_ok=True)
                    df = pd.DataFrame(torch.stack([nav_pred, nav_true]).T.numpy())
                    df.to_csv(dir_path + f"gamma_{gamma :.3f}.csv")

                    # TDA with overestimating
                    cfg = TDAcfg(train_window_size, pred_window_size, window_shift, time_delay=time_delay, stride=stride, lims=lims, interval=interval, gamma=gamma)
                    nav_pred, w_hist = run_tda(x_log_return, y_log_return, cfg, overestimate=True, log=False)

                    # compute predicted & true log return values and true net asset value for entire prediction period
                    log_return_pred = nav2logret(nav_pred)
                    log_return_true = y_log_return[cfg.train_window_size:(cfg.train_window_size+len(log_return_pred))]
                    nav_true = torch.concat([torch.tensor([nav_pred[0]]), nav_pred[0] * log_return_true.exp().cumprod(dim=0)])  # nav_pred[0] contains initial net asset value

                    # save nav result
                    dir_path = f"./{target_fund}/tt/{cfg.fcb}/"
                    os.makedirs(dir_path, exist_ok=True)
                    df = pd.DataFrame(torch.stack([nav_pred, nav_true]).T.numpy())
                    df.to_csv(dir_path + f"gamma_{gamma :.3f}.csv")


    # log using wandb
    # project_name = "vfund"
    # for target_fund in y_returns.columns:
    #     y_return = torch.from_numpy(y_returns.loc[:, target_fund].to_numpy("float32"))
    #     y_log_return = torch.log(y_return + 1)
    #     train_window_size, pred_window_size, window_shift = window_params
    #     # QP
    #     cfg = QPcfg(train_window_size, pred_window_size, window_shift)
    #     with wandb.init(config=cfg, project=project_name, group=target_fund, job_type="QP"):
    #         cfg = wandb.config
            
    #         nav_pred, w_hist = run(x_log_return, y_log_return, cfg, log=True)
            
    #         # compute predicted & true log return values and true net asset value for entire prediction period
    #         log_return_pred = nav2logret(nav_pred)
    #         log_return_true = y_log_return[cfg.train_window_size:(cfg.train_window_size+len(log_return_pred))]
    #         nav_true = torch.concat([torch.tensor([nav_pred[0]]), nav_pred[0] * log_return_true.exp().cumprod(dim=0)])  # nav_pred[0] contains initial net asset value

    #         # log net asset values
    #         wandb.log({"nav table": wandb.Table(columns=["pred nav", "true nav"], data=[[i, j] for i,j in zip(nav_pred, nav_true)])})

    #         # visualize return and asset
    #         fig = viz_logret_nav(log_return_pred, log_return_true, nav_pred, nav_true)
    #         wandb.log({"return asset plot": wandb.Image(fig)})

    #         # visualize model weights
    #         w_fig = viz_weights(w_hist)
    #         wandb.log({"weight plot": wandb.Image(w_fig)})

    #         # mse & mean prediction loss of entire time series
    #         test_mse_loss = torch.mean((log_return_pred - log_return_true)**2).item()
    #         test_avg_loss = torch.mean(log_return_pred - log_return_true).item()
    #         wandb.log({"mse_loss":test_mse_loss, "avg_loss":test_avg_loss})

    #         # regression fit between predicted & true log return
    #         test_fit = reg_fit(log_return_pred, log_return_true)
    #         intercept, slope = test_fit.summary2().tables[1].iloc[:, 0]
    #         intercept_se, slope_se = test_fit.summary2().tables[1].iloc[:, 1]
    #         skew = float(test_fit.summary2().tables[2].iloc[2,1])
    #         wandb.log({"intercept":intercept, "slope":slope, "intercept_se":intercept_se, "slope_se":slope_se, "skew":skew})

    #     # TDA
    #     for time_delay, stride in embedding_params:
    #         for lims in lims_params:
    #             for interval in interval_params:
                    
    #                 gamma = torch.rand(1).item()    # sample from Unif(0, 1)
                    
    #                 # TDA without overestimating
    #                 cfg = TDAcfg(train_window_size, pred_window_size, window_shift, time_delay=time_delay, stride=stride, lims=lims, interval=interval, gamma=gamma)
    #                 with wandb.init(config=cfg, project=project_name, group=target_fund, job_type="TDA"):
    #                     cfg = wandb.config
                        
    #                     nav_pred, w_hist = run_tda(x_log_return, y_log_return, cfg, overestimate=False, log=True)

    #                     # compute predicted & true log return values and true net asset value for entire prediction period
    #                     log_return_pred = nav2logret(nav_pred)
    #                     log_return_true = y_log_return[cfg.train_window_size:(cfg.train_window_size+len(log_return_pred))]
    #                     nav_true = torch.concat([torch.tensor([nav_pred[0]]), nav_pred[0] * log_return_true.exp().cumprod(dim=0)])  # nav_pred[0] contains initial net asset value

    #                     # log net asset values
    #                     wandb.log({"nav table": wandb.Table(columns=["pred nav", "true nav"], data=[[i, j] for i,j in zip(nav_pred, nav_true)])})

    #                     # visualize return and asset
    #                     fig = viz_logret_nav(log_return_pred, log_return_true, nav_pred, nav_true)
    #                     wandb.log({"return asset plot": wandb.Image(fig)})

    #                     # visualize model weights
    #                     w_fig = viz_weights(w_hist)
    #                     wandb.log({"weight plot": wandb.Image(w_fig)})

    #                     # mse & mean prediction loss of entire time series
    #                     test_mse_loss = torch.mean((log_return_pred - log_return_true)**2).item()
    #                     test_avg_loss = torch.mean(log_return_pred - log_return_true).item()
    #                     wandb.log({"mse_loss":test_mse_loss, "avg_loss":test_avg_loss})

    #                     # regression fit between predicted & true log return
    #                     test_fit = reg_fit(log_return_pred, log_return_true)
    #                     intercept, slope = test_fit.summary2().tables[1].iloc[:, 0]
    #                     intercept_se, slope_se = test_fit.summary2().tables[1].iloc[:, 1]
    #                     skew = float(test_fit.summary2().tables[2].iloc[2,1])
    #                     wandb.log({"intercept":intercept, "slope":slope, "intercept_se":intercept_se, "slope_se":slope_se, "skew":skew})

    #                 # TDA with overestimating
    #                 cfg = TDAcfg(train_window_size, pred_window_size, window_shift, time_delay=time_delay, stride=stride, lims=lims, interval=interval, gamma=gamma)
    #                 with wandb.init(config=cfg, project=project_name, group=target_fund, job_type="TDA+TC"):
    #                     cfg = wandb.config
                        
    #                     nav_pred, w_hist = run_tda(x_log_return, y_log_return, cfg, overestimate=True, log=True)

    #                     # compute predicted & true log return values and true net asset value for entire prediction period
    #                     log_return_pred = nav2logret(nav_pred)
    #                     log_return_true = y_log_return[cfg.train_window_size:(cfg.train_window_size+len(log_return_pred))]
    #                     nav_true = torch.concat([torch.tensor([nav_pred[0]]), nav_pred[0] * log_return_true.exp().cumprod(dim=0)])  # nav_pred[0] contains initial net asset value

    #                     # log net asset values
    #                     wandb.log({"nav table": wandb.Table(columns=["pred nav", "true nav"], data=[[i, j] for i,j in zip(nav_pred, nav_true)])})

    #                     # visualize return and asset
    #                     fig = viz_logret_nav(log_return_pred, log_return_true, nav_pred, nav_true)
    #                     wandb.log({"return asset plot": wandb.Image(fig)})

    #                     # visualize model weights
    #                     w_fig = viz_weights(w_hist)
    #                     wandb.log({"weight plot": wandb.Image(w_fig)})

    #                     # mse & mean prediction loss of entire time series
    #                     test_mse_loss = torch.mean((log_return_pred - log_return_true)**2).item()
    #                     test_avg_loss = torch.mean(log_return_pred - log_return_true).item()
    #                     wandb.log({"mse_loss":test_mse_loss, "avg_loss":test_avg_loss})

    #                     # regression fit between predicted & true log return
    #                     test_fit = reg_fit(log_return_pred, log_return_true)
    #                     intercept, slope = test_fit.summary2().tables[1].iloc[:, 0]
    #                     intercept_se, slope_se = test_fit.summary2().tables[1].iloc[:, 1]
    #                     skew = float(test_fit.summary2().tables[2].iloc[2,1])
    #                     wandb.log({"intercept":intercept, "slope":slope, "intercept_se":intercept_se, "slope_se":slope_se, "skew":skew})