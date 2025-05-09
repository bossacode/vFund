import torch
import pandas as pd
import wandb
from configs.config import QPcfg, TDAcfg
from train_eval import run, run_tda
from utils.visualize import viz_logret_nav, viz_weights
from utils.linear_reg import reg_fit

torch.manual_seed(123)

# Universe
# df_return = pd.read_pickle("./data/prev/universe_2012_ret.pkl").iloc[1:, :]
df_return = pd.read_csv("./data/universe_ret.csv", index_col=0)
drop_cols = df_return.columns[(df_return.abs() > 0.3).any()]    # drop columns s.t. have absolute value of returns larger than 0.3
x_return = torch.from_numpy(df_return.drop(columns=drop_cols).to_numpy("float32"))
x_log_return = torch.log(x_return + 1)

# Target
# y_returns = pd.read_pickle("./data/prev/target_2012_ret.pkl").iloc[1:, :]
y_returns = pd.read_csv("./data/target_ret.csv", index_col=0)

# Parameter Search Domain
# window_params = ([52, 8, 8], [26, 4, 4])    # tuple of [train_window_size, pred_window_size, window_shift]
window_params = ([52, 8, 8], )      # tuple of [train_window_size, pred_window_size, window_shift]
embedding_params = (                # tuple of [time_delay, stride]
    [2, 1],
    [5, 1],
    [10, 1],
    [15, 1]
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


# hyperparameter search
project_name = "vfund"
for target_fund in y_returns.columns:
    y_return = torch.from_numpy(y_returns.loc[:, target_fund].to_numpy("float32"))
    y_log_return = torch.log(y_return + 1)
    for train_window_size, pred_window_size, window_shift in window_params:
        # QP
        cfg = QPcfg(train_window_size, pred_window_size, window_shift)
        with wandb.init(config=cfg, project=project_name, group=target_fund, job_type="QP"):
            cfg = wandb.config
            
            nav_pred, w_hist = run(x_log_return, y_log_return, cfg, log=True)
            
            # compute predicted & true log return values and true net asset value for entire prediction period
            log_return_pred = nav2logret(nav_pred)
            log_return_true = y_log_return[cfg.train_window_size:(cfg.train_window_size+len(log_return_pred))]
            nav_true = nav_pred[0] * log_return_true.exp().cumprod(dim=0)   # nav_pred[0] contains initial net asset value

            # visualize return and asset
            fig = viz_logret_nav(log_return_pred, log_return_true, nav_pred, nav_true)
            wandb.log({"return asset plot": wandb.Image(fig)})

            # visualize model weights
            w_fig = viz_weights(w_hist)
            wandb.log({"weight plot": wandb.Image(w_fig)})

            # mse & mean prediction loss of entire time series
            test_mse_loss = torch.mean((log_return_pred - log_return_true)**2).item()
            test_avg_loss = torch.mean(log_return_pred - log_return_true).item()
            wandb.log({"mse_loss":test_mse_loss, "avg_loss":test_avg_loss})

            # regression fit between predicted & true log return
            test_fit = reg_fit(log_return_pred, log_return_true)
            intercept, slope = test_fit.summary2().tables[1].iloc[:, 0]
            intercept_se, slope_se = test_fit.summary2().tables[1].iloc[:, 1]
            skew = float(test_fit.summary2().tables[2].iloc[2,1])
            wandb.log({"intercept":intercept, "slope":slope, "intercept_se":intercept_se, "slope_se":slope_se, "skew":skew})

        # TDA
        for time_delay, stride in embedding_params:
            for lims in lims_params:
                for interval in interval_params:
                    
                    gamma = 5 + (torch.rand(1) * 5).item()  # sample from Unif(5, 10)

                    cfg = TDAcfg(train_window_size, pred_window_size, window_shift, time_delay=time_delay, stride=stride, lims=lims, interval=interval, gamma=gamma)
                    # TDA without overestimating
                    with wandb.init(config=cfg, project=project_name, group=target_fund, job_type="TDA"):
                        cfg = wandb.config
                        
                        nav_pred, w_hist = run_tda(x_log_return, y_log_return, cfg, overestimate=False, log=True)

                        # compute predicted & true log return values and true net asset value for entire prediction period
                        log_return_pred = nav2logret(nav_pred)
                        log_return_true = y_log_return[cfg.train_window_size:(cfg.train_window_size+len(log_return_pred))]
                        nav_true = nav_pred[0] * log_return_true.exp().cumprod(dim=0)   # nav_pred[0] contains initial net asset value

                        # visualize return and asset
                        fig = viz_logret_nav(log_return_pred, log_return_true, nav_pred, nav_true)
                        wandb.log({"return asset plot": wandb.Image(fig)})

                        # visualize model weights
                        w_fig = viz_weights(w_hist)
                        wandb.log({"weight plot": wandb.Image(w_fig)})

                        # mse & mean prediction loss of entire time series
                        test_mse_loss = torch.mean((log_return_pred - log_return_true)**2).item()
                        test_avg_loss = torch.mean(log_return_pred - log_return_true).item()
                        wandb.log({"mse_loss":test_mse_loss, "avg_loss":test_avg_loss})

                        # regression fit between predicted & true log return
                        test_fit = reg_fit(log_return_pred, log_return_true)
                        intercept, slope = test_fit.summary2().tables[1].iloc[:, 0]
                        intercept_se, slope_se = test_fit.summary2().tables[1].iloc[:, 1]
                        skew = float(test_fit.summary2().tables[2].iloc[2,1])
                        wandb.log({"intercept":intercept, "slope":slope, "intercept_se":intercept_se, "slope_se":slope_se, "skew":skew})

                    # TDA with overestimating
                    cfg = TDAcfg(train_window_size, pred_window_size, window_shift, time_delay=time_delay, stride=stride, lims=lims, interval=interval, gamma=gamma)
                    with wandb.init(config=cfg, project=project_name, group=target_fund, job_type="TDA+TC"):
                        cfg = wandb.config
                        
                        nav_pred, w_hist = run_tda(x_log_return, y_log_return, cfg, overestimate=True, log=True)

                        # compute predicted & true log return values and true net asset value for entire prediction period
                        log_return_pred = nav2logret(nav_pred)
                        log_return_true = y_log_return[cfg.train_window_size:(cfg.train_window_size+len(log_return_pred))]
                        nav_true = nav_pred[0] * log_return_true.exp().cumprod(dim=0)   # nav_pred[0] contains initial net asset value

                        # visualize return and asset
                        fig = viz_logret_nav(log_return_pred, log_return_true, nav_pred, nav_true)
                        wandb.log({"return asset plot": wandb.Image(fig)})

                        # visualize model weights
                        w_fig = viz_weights(w_hist)
                        wandb.log({"weight plot": wandb.Image(w_fig)})

                        # mse & mean prediction loss of entire time series
                        test_mse_loss = torch.mean((log_return_pred - log_return_true)**2).item()
                        test_avg_loss = torch.mean(log_return_pred - log_return_true).item()
                        wandb.log({"mse_loss":test_mse_loss, "avg_loss":test_avg_loss})

                        # regression fit between predicted & true log return
                        test_fit = reg_fit(log_return_pred, log_return_true)
                        intercept, slope = test_fit.summary2().tables[1].iloc[:, 0]
                        intercept_se, slope_se = test_fit.summary2().tables[1].iloc[:, 1]
                        skew = float(test_fit.summary2().tables[2].iloc[2,1])
                        wandb.log({"intercept":intercept, "slope":slope, "intercept_se":intercept_se, "slope_se":slope_se, "skew":skew})


        # baseline
        # basecfg = BaselineCfg(train_window_size, pred_window_size, window_shift)
        # with wandb.init(config=basecfg, project="prev", group=target_fund, job_type="baseline"):
        #     basecfg = wandb.config
        #     train_hist, test_hist, w_hist = run_baseline(x_log_return, y_log_return, basecfg, log=True)

        #     # visualize return and asset
        #     ret_asset_fig = viz_ret_asset(train_hist, test_hist, basecfg)
        #     wandb.log({"return asset plot": wandb.Image(ret_asset_fig)})

        #     # mse and mean loss of entire time series
        #     # train loss
        #     fitted_hist, train_target_hist = zip(*train_hist)
        #     tr_mse_loss = torch.mean((torch.concat(fitted_hist) - torch.concat(train_target_hist))**2).item()
        #     tr_avg_loss = torch.mean(torch.concat(fitted_hist) - torch.concat(train_target_hist)).item()
        #     wandb.log({"tr_mse_loss":tr_mse_loss, "tr_avg_loss":tr_avg_loss})
        #     # test loss
        #     pred_hist, test_target_hist = zip(*test_hist)
        #     test_mse_loss = torch.mean((torch.concat(pred_hist) - torch.concat(test_target_hist))**2).item()
        #     test_avg_loss = torch.mean(torch.concat(pred_hist) - torch.concat(test_target_hist)).item()
        #     wandb.log({"test_mse_loss":test_mse_loss, "test_avg_loss":test_avg_loss})

        #     # regression fit between ground truth and fitted/predicted value
        #     train_fit, test_fit = reg_fit(train_hist, test_hist)
        #     # train fit
        #     intercept, slope = train_fit.summary2().tables[1].iloc[:, 0]
        #     intercept_se, slope_se = train_fit.summary2().tables[1].iloc[:, 1]
        #     skew = float(train_fit.summary2().tables[2].iloc[2,1])
        #     wandb.log({"tr_intercept":intercept, "tr_slope":slope, "tr_intercept_se":intercept_se, "tr_slope_se":slope_se, "tr_skew":skew})
        #     # test fit
        #     intercept, slope = test_fit.summary2().tables[1].iloc[:, 0]
        #     intercept_se, slope_se = test_fit.summary2().tables[1].iloc[:, 1]
        #     skew = float(test_fit.summary2().tables[2].iloc[2,1])
        #     wandb.log({"test_intercept":intercept, "test_slope":slope, "test_intercept_se":intercept_se, "test_slope_se":slope_se, "test_skew":skew})


# train given fixed hyperparams
# time_delay_list =(1, 1, 2, 1, 1, 1, 1, 1)
# lims_list = ([[-0.15, 0.15], [-0.15, 0.15]], [[-0.2, 0.2], [-0.2, 0.2]], [[-0.2, 0.2], [-0.2, 0.2]], [[-0.2, 0.2], [-0.2, 0.2]], [[-0.2, 0.2], [-0.2, 0.2]], [[-0.2, 0.2], [-0.2, 0.2]], [[-0.2, 0.2], [-0.2, 0.2]], [[-0.2, 0.2], [-0.2, 0.2]])
# interval_list = ([0., 0.03], [0., 0.03], [0., 0.02], [0., 0.03], [0., 0.03], [0., 0.02], [0., 0.03], [0., 0.03])
# alpha_list = (6.609009742736816, 6.662144660949707, 6.15727424621582, 8.043250560760498, 9.91151237487793, 4.669143438339233, 7.8944091796875, 2.1938716173171997)

# os.makedirs("./result/prev/", exist_ok=True)

# for target_fund, time_delay, lims, interval, alpha in zip(y_returns.columns, time_delay_list, lims_list, interval_list, alpha_list):
#     y_return = torch.from_numpy(y_returns.loc[:, target_fund].to_numpy("float32"))
#     y_log_return = torch.log(y_return + 1)
    
#     tdacfg = TDACfgTc(52, 8, 8, time_delay=time_delay, stride=1, lims=lims, interval=interval, alpha=alpha)
#     # Tranaction cost + scale
#     train_hist, scaled_test_hist, w_hist, scaled_asset_hist, scaled_tc_prop_hist = run_tda_tc(x_log_return, y_log_return, tdacfg, use_scale=True, log=False)
#     scaled_pred_hist, _ = zip(*scaled_test_hist)
#     # Tranaction cost + no scale
#     train_hist, test_hist, w_hist, asset_hist, tc_prop_hist = run_tda_tc(x_log_return, y_log_return, tdacfg, use_scale=False, log=False)
#     pred_hist, test_target_hist = zip(*test_hist)
#     num_windows = len(test_target_hist)

#     plt.style.use("seaborn-v0_8-dark")
#     fig, ax = plt.subplots(3, 1, figsize=(40, 40))
#     plt.subplots_adjust(wspace=0.05, hspace=0.05)

#     # asset plot
#     ax[0].plot(scaled_asset_hist[1:], "-^", label="Scaled Pred", c="red")
#     ax[0].plot(asset_hist[1:], "-o", label="Not Scaled Pred", c="green")
#     ax[0].plot(torch.concat(test_target_hist).exp().cumprod(dim=0).cpu(), "--v", label="True", c="blue")
#     ax[0].set_xticks(range(-1, num_windows*tdacfg.pred_window_size, tdacfg.pred_window_size))
#     ax[0].set_title("Test Assets reflecting Transaction Cost")
#     ax[0].legend(prop={'size': 30})
#     ax[0].grid(True)

#     # return plot
#     ax[1].plot(torch.concat(scaled_pred_hist).cpu(), "-^", label="Scaled Pred", c="red")
#     ax[1].plot(torch.concat(pred_hist).cpu(), "-o", label="Not Scaled Pred", c="green")
#     ax[1].plot(torch.concat(test_target_hist).cpu(), "--v", label="True", c="blue")
#     ax[1].set_xticks(range(-1, num_windows*tdacfg.pred_window_size, tdacfg.pred_window_size))
#     ax[1].set_title("Test Returns")
#     ax[1].legend(prop={'size': 30})
#     ax[1].grid(True)

#     # transaction cost proportion plot
#     ax[2].plot(scaled_tc_prop_hist, "-^", label="Scaled", c="red")
#     ax[2].plot(tc_prop_hist, "-o", label="Not Scaled", c="green")
#     ax[2].set_xticks(range(-1, num_windows))
#     ax[2].set_title("Proportion of Transaction Cost")
#     ax[2].legend(prop={'size': 30})
#     ax[2].grid(True)

#     plt.savefig(f"./result/prev/{target_fund}.png", bbox_inches='tight')