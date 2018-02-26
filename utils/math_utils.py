import numpy as np
import scipy.signal as signal


def discount(x, gamma):
  """
  Compute discounted sum of future values
  out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
  """
  return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def explained_variance_1d(ypred, y):
  """
  Var[ypred - y] / var[y].
  https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
  """
  assert y.ndim == 1 and ypred.ndim == 1
  vary = np.var(y)
  return np.nan if vary == 0 else 1 - np.var(y-ypred)/vary
