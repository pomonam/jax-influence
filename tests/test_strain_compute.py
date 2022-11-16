from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
import jax
from jax.config import config
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import numpy as np
from sklearn.linear_model import LinearRegression as SkLR

from jax_influence.computer import InfluenceComputer
from jax_influence.utils import make_float64
from tests.utils import LinearRegression
from tests.utils import load_dummy_data

config.update("jax_enable_x64", True)


def l2_loss(model, params, inputs, targets, outputs=None):
  if outputs is None:
    outputs = model.apply(params, inputs)
  return 0.5 * jnp.mean((outputs - targets)**2.)


class TestSTrainCompute(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.num_features = 10
    x_train, y_train, x_test, y_test = load_dummy_data(
        num_features=self.num_features)
    self.data_size = x_train.shape[0]
    self.model = hk.without_apply_rng(
        hk.transform(lambda *args: LinearRegression(1, bias=True)(*args)))
    self.params = self.model.init(jax.random.PRNGKey(42), x_train)

    clf = SkLR()
    clf.fit(x_train, y_train)

    self.params = hk.data_structures.to_mutable_dict(self.params)
    self.params["linear_regression/linear"]["w"] = jnp.array(clf.coef_).T
    self.params["linear_regression/linear"]["b"] = jnp.array(clf.intercept_)
    self.params = hk.data_structures.to_immutable_dict(self.params)
    self.float64_params = make_float64(self.params)

    self.train_idx = 16
    self.train_input = np.expand_dims(x_train[self.train_idx], 0)
    self.train_target = np.expand_dims(y_train[self.train_idx], 0)

    params_flat, unravel = ravel_pytree(self.float64_params)
    hess = jax.hessian(
        lambda p: l2_loss(self.model, unravel(p), x_train, y_train))(
            params_flat)

    print("Difference between true Hessian and computed Hessian:")
    real_hessian = x_train.T @ x_train / x_train.shape[0]
    print(jnp.linalg.norm(real_hessian - hess[-10:, -10:]))

    inv_hess = jnp.linalg.inv(hess + 0.001 * jnp.eye(hess.shape[0]))
    flat_grads = jax.grad(lambda p: l2_loss(self.model, unravel(p), self.train_input, self.train_target))\
        (params_flat)
    self.real_ihvp = inv_hess @ flat_grads

    self.influence_computer = InfluenceComputer(
        self.model,
        (x_train, y_train),
        (x_train, y_train),
        l2_loss,
        damping=0.001,
        scale=50,
        recursion_depth=5000,
        repeat=10,
    )

  def test_compute_strain_cg(self):
    estimated_params, estimated_ihvp = self.influence_computer.compute_strain_cg(
        self.params, self.train_idx)
    flat_estimated_ihvp, unravel = ravel_pytree(estimated_ihvp)

    print("\nConjugate Gradient:")
    self.assertTrue(self.check_estimation(flat_estimated_ihvp))

  def test_compute_strain_gnh_cg(self):
    estimated_params, estimated_ihvp = \
        self.influence_computer.compute_strain_cg(self.params, self.train_idx, gnh=True)
    flat_estimated_ihvp, unravel = ravel_pytree(estimated_ihvp)

    print("\nConjugate Gradient (GNH):")
    self.assertTrue(self.check_estimation(flat_estimated_ihvp))

  def test_compute_strain_lissa(self):
    estimated_params, estimated_ihvp = self.influence_computer.compute_strain_lissa(
        self.params, self.train_idx)
    flat_estimated_ihvp, unravel = ravel_pytree(estimated_ihvp)

    print("\nLiSSA:")
    self.assertTrue(self.check_estimation(flat_estimated_ihvp, tol=0.1))

  def test_compute_strain_gnh_lissa(self):
    estimated_params, estimated_ihvp = \
        self.influence_computer.compute_strain_lissa(self.params, self.train_idx, gnh=True)
    flat_estimated_ihvp, unravel = ravel_pytree(estimated_ihvp)

    print("\nLiSSA (GNH):")
    self.assertTrue(self.check_estimation(flat_estimated_ihvp, tol=0.1))

  def check_estimation(self, estimated_ihvp, tol=1e-5):
    print(estimated_ihvp.T)
    print(self.real_ihvp.T)

    l_2_difference = jnp.linalg.norm(self.real_ihvp - estimated_ihvp)
    l_infty_difference = jnp.linalg.norm(
        self.real_ihvp - estimated_ihvp, ord=jnp.inf)
    print(f"L-2 difference: {l_2_difference}")
    print(f"L-infty difference: {l_infty_difference}")

    return jnp.allclose(self.real_ihvp, estimated_ihvp, rtol=tol, atol=tol)


if __name__ == "__main__":
  absltest.main()
