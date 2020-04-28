from math import sqrt
import numpy as np
from scipy import stats

#
def rmse(y, f):
    rmse = sqrt(((y - f) ** 2))
    return rmse


def mse(y, f):
    mse = ((y - f) ** 2)
    return mse


def mae(y, f):
    return np.sum(np.abs(y - f))

  # huber loss
def huber(true, pred, delta=0.5):
    loss = np.where(np.abs(true - pred) < delta, 0.5 * ((true - pred) ** 2),
                    delta * np.abs(true - pred) - 0.5 * (delta ** 2))
    return np.sum(loss)

    # log cosh loss
def logcosh(true, pred):
    loss = np.log(np.cosh(pred - true))
    return np.sum(loss)


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci
