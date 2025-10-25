import numpy as np


def rse(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def corr(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def mae(pred, true):
    return np.mean(np.abs(pred - true))


def mse(pred, true):
    return np.mean((pred - true) ** 2)


def rmse(pred, true):
    return np.sqrt(mse(pred, true))


def mape(pred, true):
    mask = (true != 0)
    if mask.sum() > 0:
        return np.mean(np.abs((pred[mask] - true[mask]) / true[mask]))
    else:
        return np.nan


def smape(pred, true):
    mask = (true != 0)
    if mask.sum() > 0:
        p = pred[mask]
        t = true[mask]
        denominator = (np.abs(p) + np.abs(t)) / 2.0
        smape = np.abs(p - t) / denominator
        smape = np.nan_to_num(smape)
        return np.mean(smape)
    else:
        return np.nan


def calc_metrics(pred, true):
    return {'mae': mae(pred, true),
            'mse': mse(pred, true),
            'rmse': rmse(pred, true),
            'mape': mape(pred, true),
            'smape': smape(pred, true)}
