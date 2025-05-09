import torch
import matplotlib.pyplot as plt
import seaborn as sns


def viz_logret_nav(logret_pred, logret_true, nav_pred, nav_true):
    fig, ax = plt.subplots(2, 1, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    lw = 3.5
    # return
    ax[0].plot(logret_pred, "-o", label="Pred", c='#009E73', lw=lw)
    ax[0].plot(logret_true, "--v", label="True", c='#CC79A7', lw=lw)
    ax[0].set_title("Log Return")
    ax[0].legend(prop={'size': 60})
    # net asset value
    ax[1].plot(nav_pred, "-o", label="Pred", c='#009E73', lw=lw)
    ax[1].plot(nav_true, "--v", label="True", c='#CC79A7', lw=lw)
    ax[1].set_title("Net Asset Value")
    ax[1].legend(prop={'size': 60})
    return fig


def viz_weights(w_hist):
    fig = plt.figure(figsize=(40, 8))
    sns.heatmap(w_hist)
    plt.title("Weights")
    plt.xlabel("Assets")
    plt.ylabel("Windows")
    plt.close(fig)
    return fig


# def viz_avg_error(train_hist, test_hist):
#     # mean errors
#     train_error, test_error = [], []
#     for fitted, train_target in train_hist:
#         train_error.append(torch.mean(fitted - train_target).item())
#     for pred, test_target in test_hist:
#         test_error.append(torch.mean(pred - test_target).item())
    
#     plt.style.use("seaborn-v0_8-dark")
#     fig = plt.figure(figsize=(15,5))
#     plt.plot(train_error, "-o", label="train", c="blue")
#     plt.plot(test_error, "--v", label ="test", c="red")
#     plt.hlines(y=0, xmin=-0.5, xmax=len(train_error)-0.5, colors="black")
#     plt.title("Average Errors")
#     plt.xticks(range(len(train_hist)))
#     plt.legend(prop={'size': 15})
#     plt.grid(True)
#     plt.close(fig)
#     return fig


# def viz_ret_asset(train_hist, test_hist, cfg):
#     fitted_hist, train_target_hist = zip(*train_hist)
#     pred_hist, test_target_hist = zip(*test_hist)
#     num_windows = len(fitted_hist)

#     plt.style.use("seaborn-v0_8-dark")
#     fig, ax = plt.subplots(2, 2, figsize=(40, 30))
#     plt.subplots_adjust(wspace=0.05, hspace=0.05)
#     # pred return
#     ax[0,0].plot(torch.concat(pred_hist).cpu(), "-^", label="Pred", c="red")
#     ax[0,0].plot(torch.concat(test_target_hist).cpu(), "--v", label="True", c="blue")
#     ax[0,0].set_xticks(range(-1, num_windows*cfg.pred_window_size, cfg.pred_window_size))
#     ax[0,0].set_title("Test log return")
#     ax[0,0].legend(prop={'size': 30})
#     ax[0,0].grid(True)
#     # pred asset
#     ax[0,1].plot(torch.concat(pred_hist).exp().cumprod(dim=0).cpu(), "-^", label="Pred", c="red")
#     ax[0,1].plot(torch.concat(test_target_hist).exp().cumprod(dim=0).cpu(), "--v", label="True", c="blue")
#     ax[0,1].set_xticks(range(-1, num_windows*cfg.pred_window_size, cfg.pred_window_size))
#     ax[0,1].set_title("Test asset")
#     ax[0,1].legend(prop={'size': 30})
#     ax[0,1].grid(True)
#     #train return
#     ax[1,0].plot(torch.concat(fitted_hist).cpu(), "-^", label="Fitted", c="red")
#     ax[1,0].plot(torch.concat(train_target_hist).cpu(), "--v", label="True", c="blue")
#     ax[1,0].set_xticks(range(-1, num_windows*cfg.train_window_size, cfg.train_window_size))
#     ax[1,0].set_title("Train log return")
#     ax[1,0].legend(prop={'size': 30})
#     ax[1,0].grid(True)
#     # train asset
#     ax[1,1].plot(torch.concat(fitted_hist).exp().cumprod(dim=0).cpu(), "-^", label="Fitted", c="red")
#     ax[1,1].plot(torch.concat(train_target_hist).exp().cumprod(dim=0).cpu(), "--v", label="True", c="blue")
#     ax[1,1].set_xticks(range(-1, num_windows*cfg.train_window_size, cfg.train_window_size))
#     ax[1,1].set_title("Train asset")
#     ax[1,1].legend(prop={'size': 30})
#     ax[1,1].grid(True)
#     plt.close(fig)
#     return fig