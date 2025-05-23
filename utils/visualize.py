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
    plt.close(fig)
    return fig


def viz_weights(w_hist):
    fig = plt.figure(figsize=(40, 8))
    sns.heatmap(w_hist)
    plt.title("Weights")
    plt.xlabel("Assets")
    plt.ylabel("Windows")
    plt.close(fig)
    return fig