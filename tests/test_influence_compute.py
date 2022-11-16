from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
import jax
from jax import nn
import jax.numpy as jnp
import numpy as np
from sklearn import linear_model

from jax_influence.computer import InfluenceComputer
from tests.utils import BinaryLogisticRegression
from tests.utils import load_binary_mnist_data
from tests.utils import visualize_result


def log_clip(x):
  return jnp.log(jnp.maximum(x, jnp.ones_like(x) * 1e-10))


def xe_loss(model, params, inputs, targets, outputs=None):
  if outputs is None:
    outputs = nn.sigmoid(model.apply(params, inputs))
  else:
    outputs = nn.sigmoid(outputs)
  loss = -jnp.mean(targets * log_clip(outputs) +
                   (1 - targets) * log_clip(1 - outputs))
  return loss


class TestInfluenceCompute(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.weight_decay = 0.01
    self.sample_num = 30
    self.test_idx = 5

    self.train_inputs, self.train_targets, self.test_inputs, self.test_targets = \
        load_binary_mnist_data()
    self.train_sample_num = len(self.train_inputs)
    self.c = 1.0 / (self.train_sample_num * self.weight_decay)
    sk_model = linear_model.LogisticRegression(
        C=self.c, solver="lbfgs", tol=1e-10, max_iter=1000, fit_intercept=False)
    sk_model.fit(self.train_inputs, self.train_targets.ravel())

    self.model = hk.without_apply_rng(
        hk.transform(lambda *args: BinaryLogisticRegression(1, bias=False)
                     (*args)))
    self.params = self.model.init(jax.random.PRNGKey(42), self.train_inputs)

    self.params = hk.data_structures.to_mutable_dict(self.params)
    self.params["binary_logistic_regression/linear"]["w"] = jnp.array(
        sk_model.coef_).T

    self.test_inputs_single = np.expand_dims(self.test_inputs[self.test_idx], 0)
    self.test_targets_single = np.expand_dims(self.test_targets[self.test_idx],
                                              0)
    self.test_loss = xe_loss(self.model,
                             self.params,
                             self.test_inputs_single,
                             self.test_targets_single)

    self.influence = InfluenceComputer(
        self.model,
        (self.train_inputs, self.train_targets),
        (self.test_inputs, self.test_targets),
        xe_loss,
        damping=1e-4,
        scale=100,
        recursion_depth=5000,
        repeat=10,
        wd=self.weight_decay,
    )

  def compute_influence_single(self, use_stest=True, use_cg=True, gnh=False):
    if use_stest:
      loss_diff_approx, s_test = \
          self.influence.compute_single_test_stest_influence(self.params, self.test_idx, use_cg=use_cg, gnh=gnh)
    else:
      loss_diff_approx = \
          self.influence.compute_single_test_strain_influence(self.params, self.test_idx, use_cg=use_cg, gnh=gnh)

    loss_diff_approx = jnp.negative(loss_diff_approx) / len(loss_diff_approx)
    sorted_indice = jnp.argsort(loss_diff_approx)
    sample_indice = jnp.concatenate([
        sorted_indice[-int(self.sample_num / 2):],
        sorted_indice[:int(self.sample_num / 2)]
    ])

    loss_diff_true = np.zeros(self.sample_num)
    for i, index in zip(range(self.sample_num), sample_indice):
      print("[{}/{}]".format(i + 1, self.sample_num))
      train_inputs_minus_one = jnp.delete(self.train_inputs, index, axis=0)
      train_targets_minus_one = jnp.delete(self.train_targets, index, axis=0)

      c = 1.0 / ((self.train_sample_num - 1) * self.weight_decay)
      sklearn_model_minus_one = linear_model.LogisticRegression(
          C=c, solver="lbfgs", tol=1e-10, max_iter=1000, fit_intercept=False)
      sklearn_model_minus_one.fit(train_inputs_minus_one,
                                  train_targets_minus_one.ravel())

      w_retrain = sklearn_model_minus_one.coef_
      self.params["binary_logistic_regression/linear"]["w"] = jnp.array(
          w_retrain).T

      test_loss_retrain = xe_loss(self.model,
                                  self.params,
                                  self.test_inputs_single,
                                  self.test_targets_single)
      loss_diff_true[i] = test_loss_retrain - self.test_loss

      print("Original loss       :{}".format(self.test_loss))
      print("Retrain loss        :{}".format(test_loss_retrain))
      print("True loss diff      :{}".format(loss_diff_true[i]))
      print("Estimated loss diff :{}".format(loss_diff_approx[index]))

    corr = visualize_result(loss_diff_true, loss_diff_approx[sample_indice])
    return corr

  def test_compute_influence_single_cg(self):
    corr = self.compute_influence_single(use_cg=True, gnh=False)
    print(corr)
    self.assertTrue(corr > 0.9)

  def test_compute_influence_single_gnh_cg(self):
    corr = self.compute_influence_single(use_cg=True, gnh=True)
    print(corr)
    self.assertTrue(corr > 0.9)

  def test_compute_influence_single_lissa(self):
    self.influence.scale = 10
    corr = self.compute_influence_single(use_cg=False, gnh=False)
    print(corr)
    self.assertTrue(corr > 0.9)

  def test_compute_influence_single_gnh_lissa(self):
    self.influence.scale = 10
    corr = self.compute_influence_single(use_cg=False, gnh=True)
    print(corr)
    self.assertTrue(corr > 0.9)


if __name__ == "__main__":
  absltest.main()
