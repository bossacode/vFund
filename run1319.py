import torch
import pandas as pd
import wandb
from configs.config import Cfg, TDACfg
from train_eval import run, run_tda
from utils.visualize import viz_ret_asset
from utils.linear_reg import reg_fit


# Universe
df_return = pd.read_csv("./data/1319/WEEK_univ1319_redret.csv").iloc[1:, :]
drop_cols = df_return.columns[(df_return.abs() > 0.3).any()]    # drop columns s.t. have absolute value of returns larger than 0.3
x_return = torch.from_numpy(df_return.drop(columns=drop_cols).to_numpy("float32"))
x_log_return = torch.log(x_return + 1)

# Targets
y_returns = pd.read_csv("./data/1319/WEEK_target1319.csv", index_col=0).iloc[1:, :]

# Parameter Search Domain
window_params = ([26, 4, 4], [52, 8, 8])    # tuple of [train_window_size, pred_window_size, window_shift]
embedding_params = ([1, 1], [2, 1])         # tuple of [time_delay, stride]
lims_params = ([[-0.05, 0.05], [-0.05, 0.05]], [[-0.1, 0.1], [-0.1, 0.1]])
interval_params = ([0., 0.01], [0., 0.02], [0., 0.03], [0., 0.04])
alpha_params = (1, 10, 50)


for target_fund in y_returns.columns:
    y_return = torch.from_numpy(y_returns.loc[:, target_fund].to_numpy("float32"))
    y_log_return = torch.log(y_return + 1)
    for train_window_size, pred_window_size, window_shift in window_params:
        # QP
        cfg = Cfg(train_window_size, pred_window_size, window_shift)
        with wandb.init(config=cfg, project="1319", group=target_fund, job_type="QP"):
            cfg = wandb.config
            train_hist, test_hist, w_hist = run(x_log_return, y_log_return, cfg, log=True)
            
            # visualize return and asset
            ret_asset_fig = viz_ret_asset(train_hist.cpu(), test_hist.cpu(), cfg)
            wandb.log({"return asset plot": wandb.Image(ret_asset_fig)})

            # mse and mean loss of entire time series
            # train loss
            fitted_hist, train_target_hist = zip(*train_hist)
            tr_mse_loss = torch.mean((torch.concat(fitted_hist) - torch.concat(train_target_hist))**2).item()
            tr_avg_loss = torch.mean(torch.concat(fitted_hist) - torch.concat(train_target_hist)).item()
            wandb.log({"tr_mse_loss":tr_mse_loss, "tr_avg_loss":tr_avg_loss})
            # test loss
            pred_hist, test_target_hist = zip(*test_hist)
            test_mse_loss = torch.mean((torch.concat(pred_hist) - torch.concat(test_target_hist))**2).item()
            test_avg_loss = torch.mean(torch.concat(pred_hist) - torch.concat(test_target_hist)).item()
            wandb.log({"test_mse_loss":test_mse_loss, "test_avg_loss":test_avg_loss})

            # regression fit between ground truth and fitted/predicted value
            train_fit, test_fit = reg_fit(train_hist, test_hist)
            # train fit
            intercept, slope = train_fit.summary2().tables[1].iloc[:, 0]
            intercept_se, slope_se = train_fit.summary2().tables[1].iloc[:, 1]
            skew = float(train_fit.summary2().tables[2].iloc[2,1])
            wandb.log({"tr_intercept":intercept, "tr_slope":slope, "tr_intercept_se":intercept_se, "tr_slope_se":slope_se, "tr_skew":skew})
            # test fit
            intercept, slope = test_fit.summary2().tables[1].iloc[:, 0]
            intercept_se, slope_se = test_fit.summary2().tables[1].iloc[:, 1]
            skew = float(test_fit.summary2().tables[2].iloc[2,1])
            wandb.log({"test_intercept":intercept, "test_slope":slope, "test_intercept_se":intercept_se, "test_slope_se":slope_se, "test_skew":skew})

        # TDA
        for time_delay, stride in embedding_params:
            for lims in lims_params:
                for interval in interval_params:
                    for alpha in alpha_params:
                        tdacfg = TDACfg(train_window_size, pred_window_size, window_shift, time_delay=time_delay, stride=stride, lims=lims, interval=interval, alpha=alpha)
                        with wandb.init(config=tdacfg, project="1319", group=target_fund, job_type="TDA"):
                            tdacfg = wandb.config
                            train_hist, test_hist, w_hist = run_tda(x_log_return, y_log_return, tdacfg, log=True)

                            # visualize return and asset
                            ret_asset_fig = viz_ret_asset(train_hist.cpu(), test_hist.cpu(), cfg)
                            wandb.log({"return asset plot": wandb.Image(ret_asset_fig)})

                            # mse and mean loss of entire time series
                            # train loss
                            fitted_hist, train_target_hist = zip(*train_hist)
                            tr_mse_loss = torch.mean((torch.concat(fitted_hist) - torch.concat(train_target_hist))**2).item()
                            tr_avg_loss = torch.mean(torch.concat(fitted_hist) - torch.concat(train_target_hist)).item()
                            wandb.log({"tr_mse_loss":tr_mse_loss, "tr_avg_loss":tr_avg_loss})
                            # test loss
                            pred_hist, test_target_hist = zip(*test_hist)
                            test_mse_loss = torch.mean((torch.concat(pred_hist) - torch.concat(test_target_hist))**2).item()
                            test_avg_loss = torch.mean(torch.concat(pred_hist) - torch.concat(test_target_hist)).item()
                            wandb.log({"test_mse_loss":test_mse_loss, "test_avg_loss":test_avg_loss})

                            # regression fit between ground truth and fitted/predicted value
                            train_fit, test_fit = reg_fit(train_hist, test_hist)
                            # train fit
                            intercept, slope = train_fit.summary2().tables[1].iloc[:, 0]
                            intercept_se, slope_se = train_fit.summary2().tables[1].iloc[:, 1]
                            skew = float(train_fit.summary2().tables[2].iloc[2,1])
                            wandb.log({"tr_intercept":intercept, "tr_slope":slope, "tr_intercept_se":intercept_se, "tr_slope_se":slope_se, "tr_skew":skew})
                            # test fit
                            intercept, slope = test_fit.summary2().tables[1].iloc[:, 0]
                            intercept_se, slope_se = test_fit.summary2().tables[1].iloc[:, 1]
                            skew = float(test_fit.summary2().tables[2].iloc[2,1])
                            wandb.log({"test_intercept":intercept, "test_slope":slope, "test_intercept_se":intercept_se, "test_slope_se":slope_se, "test_skew":skew})