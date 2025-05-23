import statsmodels.api as sm


def reg_fit(pred, target):
    x = sm.add_constant(pred.numpy())
    est = sm.OLS(target.numpy(), x)
    test_fit = est.fit()

    print(test_fit.summary())
    return test_fit