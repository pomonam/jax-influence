""" Utilities for testing influence functions.

Some codes are adapted from:
https://github.com/google-research/jax-influence.
"""

from functools import partial
import random

import haiku as hk
import jax
from jax.flatten_util import ravel_pytree
import jax.nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import torch
from torchvision import transforms
from torchvision.datasets import MNIST


class LinearRegression(hk.Module):

  def __init__(self, out_features, bias=False, name=None):
    super().__init__(name=name)
    self.bias = bias
    self.out_features = out_features

  def __call__(self, x):
    out = hk.Linear(self.out_features, with_bias=self.bias)(x)
    return out


class BinaryLogisticRegression(hk.Module):
  # While the functionalities are same as that of LinearRegression,
  # we use a separate module for easier interpretation.

  def __init__(self, out_features, bias=False, name=None):
    super().__init__(name=name)
    self.bias = bias
    self.out_features = out_features

  def __call__(self, x):
    out = hk.Linear(self.out_features, with_bias=self.bias)(x)
    return out


class MLP(hk.Module):

  def __init__(self, out_features, bias=False, name=None):
    super().__init__(name=name)
    self.bias = bias
    self.out_features = out_features

  def __call__(self, x):
    x = hk.Linear(100, with_bias=self.bias)(x)
    x = jax.nn.relu(x)
    out = hk.Linear(self.out_features, with_bias=self.bias)(x)
    return out


def from10to2classes(x, y, num_a, num_b):
  is_num_a, is_num_b = y == num_a, y == num_b
  x_2class = np.concatenate([x[is_num_a], x[is_num_b]])
  y_2class = np.concatenate([np.ones(is_num_a.sum()),
                             np.zeros(is_num_b.sum())]).reshape([-1, 1])
  return x_2class, y_2class


def load_binary_mnist_data(num_data=None):
  train_data = MNIST(
      "../data",
      train=True,
      download=True,
      transform=transforms.Compose([
          transforms.ToTensor(),
      ]))
  x_train, y_train = train_data.data, train_data.targets

  test_data = MNIST(
      "../data",
      train=False,
      download=True,
      transform=transforms.Compose([
          transforms.ToTensor(),
      ]))
  x_test, y_test = test_data.data, test_data.targets

  if num_data is not None:
    x_train = x_train[:num_data]
    y_train = y_train[:num_data]

  x_train, x_test = x_train.reshape([-1, 784]) / 255.0, x_test.reshape(
      [-1, 784]) / 255.0
  x_train_2class, y_train_2class = from10to2classes(x_train, y_train, 1, 3)
  x_test_2class, y_test_2class = from10to2classes(x_test, y_test, 1, 3)

  return x_train_2class, y_train_2class, x_test_2class, y_test_2class


def load_dummy_data(num_samples=50000, num_features=10, seed=0):
  state = np.random.RandomState(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  x, y = make_regression(num_samples,
                         num_features,
                         random_state=seed,
                         n_informative=num_features,
                         noise=1)
  y = np.expand_dims(y, -1)

  permutation = state.choice(np.arange(x.shape[0]), x.shape[0], replace=False)
  size_train = int(np.round(x.shape[0] * 0.8))
  index_train = permutation[0:size_train]
  index_val = permutation[size_train:]
  x_train, y_train = x[index_train, :], y[index_train, :]
  x_test, y_test = x[index_val, :], y[index_val, :]

  return x_train, y_train, x_test, y_test


def visualize_result(actual_loss_diff, estimated_loss_diff):
  r2_s = r2_score(actual_loss_diff, estimated_loss_diff)
  spearman = spearmanr(actual_loss_diff, estimated_loss_diff)[0]

  max_abs = np.max([np.abs(actual_loss_diff), np.abs(estimated_loss_diff)])
  min_, max_ = -max_abs * 1.1, max_abs * 1.1
  plt.rcParams["figure.figsize"] = 6, 5
  plt.scatter(actual_loss_diff, estimated_loss_diff, zorder=2, s=10)
  plt.title("Loss difference")
  plt.xlabel("Actual loss diff")
  plt.ylabel("Estimated loss diff")
  range_ = [min_, max_]
  plt.plot(range_, range_, "k-", alpha=0.2, zorder=1)
  text = "MAE = {:.03}\nR2 score = {:.03}\nSpearman = {:.03}".format(
      mean_absolute_error(actual_loss_diff, estimated_loss_diff),
      r2_s,
      spearman)
  plt.text(
      max_abs,
      -max_abs,
      text,
      verticalalignment="bottom",
      horizontalalignment="right")
  plt.xlim(min_, max_)
  plt.ylim(min_, max_)

  plt.show()
  return spearman
