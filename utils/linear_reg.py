import torch
import statsmodels.api as sm


def reg_fit(train_hist, test_hist):
    fitted_hist, train_target_hist = zip(*train_hist)
    pred_hist, test_target_hist = zip(*test_hist)

    # train
    x = sm.add_constant(torch.concat(fitted_hist).numpy())
    est = sm.OLS(torch.concat(train_target_hist).numpy(), x)
    train_fit = est.fit()
    
    # test
    x = sm.add_constant(torch.concat(pred_hist).numpy())
    est = sm.OLS(torch.concat(test_target_hist).numpy(), x)
    test_fit = est.fit()

    print("Train")
    print(train_fit.summary(), "\n")
    print("-"*100, "\n")
    print("Test")
    print(test_fit.summary())
    return train_fit, test_fit