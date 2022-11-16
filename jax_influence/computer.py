""" Influence Functions implementation."""

from functools import partial

import jax
from jax import grad
from jax import jit
from jax import lax
from jax.config import config
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg as jcg
from jax.tree_util import tree_leaves
import tree

from jax_influence.utils import _add
from jax_influence.utils import _div
from jax_influence.utils import _mul
from jax_influence.utils import _sub
from jax_influence.utils import _vdot
from jax_influence.utils import gnhvp
from jax_influence.utils import hvp
from jax_influence.utils import make_float32
from jax_influence.utils import make_float64
from jax_influence.utils import tree_zeros_like


class InfluenceComputer:

  def __init__(self,
               model,
               train_dataset,
               test_dataset,
               loss_fnc,
               wd=0.,
               damping=0.,
               scale=10.0,
               recursion_depth=5000,
               repeat=10,
               seed=42):
    self.model = model
    self.train_inputs, self.train_targets = train_dataset
    self.test_inputs, self.test_targets = test_dataset
    self.loss_fnc = loss_fnc
    self.wd = wd
    self.damping = damping
    self.scale = scale
    self.recursion_depth = recursion_depth
    self.repeat = repeat
    self.seed = seed
    self.key = jax.random.PRNGKey(seed)

  @partial(jit, static_argnums=(0,))
  def get_loss_with_wd(self, params, inputs, targets):
    loss = self.loss_fnc(self.model, params, inputs, targets)
    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
    return loss + self.wd * l2_loss

  @partial(jit, static_argnums=(0,))
  def get_loss_with_wd_state(self, params, state, inputs, targets):
    loss = self.loss_fnc(self.model, params, state, inputs, targets)
    l2_params = [p for ((mod_name, _), p) in tree.flatten_with_path(params)]
    l2_loss = 0.5 * sum(
        jnp.sum(jnp.square(p)) for p in jax.tree_leaves(l2_params))
    return loss + self.wd * l2_loss

  @partial(jit, static_argnums=(0,))
  def get_loss_wo_wd(self, params, inputs, targets):
    return self.loss_fnc(self.model, params, inputs, targets)

  @partial(jit, static_argnums=(0,))
  def get_loss_wo_wd_state(self, params, state, inputs, targets):
    return self.loss_fnc(self.model, params, state, inputs, targets)

  def get_train_loss_wo_wd(self, params):
    return self.get_loss_wo_wd(params, self.train_inputs, self.train_targets)

  def get_train_loss_with_wd(self, params, state=None):
    if state is None:
      return self.get_loss_with_wd(params,
                                   self.train_inputs,
                                   self.train_targets)
    else:
      return self.get_loss_with_wd_state(params,
                                         state,
                                         self.train_inputs,
                                         self.train_targets)

  def get_test_loss_wo_wd(self, params):
    return self.get_loss_wo_wd(params, self.test_inputs, self.test_targets)

  def get_single_train_loss_with_wd(self, params, index):
    idx_train_input = jnp.expand_dims(self.train_inputs[index], 0)
    idx_train_target = jnp.expand_dims(self.train_targets[index], 0)
    return self.get_loss_with_wd(params, idx_train_input, idx_train_target)

  def get_single_train_loss_wo_wd(self, params, index, state=None):
    idx_train_input = jnp.expand_dims(self.train_inputs[index], 0)
    idx_train_target = jnp.expand_dims(self.train_targets[index], 0)
    if state is None:
      return self.get_loss_wo_wd(params, idx_train_input, idx_train_target)
    else:
      return self.get_loss_wo_wd_state(params,
                                       state,
                                       idx_train_input,
                                       idx_train_target)

  def get_multi_train_loss_wo_wd(self, params, index):
    idx_train_input = self.train_inputs[index, :]
    idx_train_target = self.train_targets[index, :]
    return self.get_loss_wo_wd(params, idx_train_input,
                               idx_train_target) * len(index)

  def get_single_test_loss_wo_wd(self, params, index):
    idx_test_input = jnp.expand_dims(self.test_inputs[index], 0)
    idx_test_target = jnp.expand_dims(self.test_targets[index], 0)
    return self.get_loss_wo_wd(params, idx_test_input, idx_test_target)

  @partial(jit, static_argnums=(0,))
  def get_grad_single_loss_with_wd(self, params, inputs, targets):
    return grad(lambda p: self.get_loss_with_wd(p, inputs, targets))(params)

  @partial(jit, static_argnums=(0,))
  def get_grad_single_loss_wo_wd(self, params, inputs, targets):
    return grad(lambda p: self.get_loss_wo_wd(p, inputs, targets))(params)

  def get_grad_single_train_loss_with_wd(self, params, train_index):
    return grad(lambda p: self.get_single_train_loss_with_wd(p, train_index))(
        params)

  def get_grad_single_train_loss_wo_wd(self, params, train_index, state=None):
    if state is None:
      return grad(lambda p: self.get_single_train_loss_wo_wd(p, train_index))(
          params)
    else:
      return grad(
          lambda p: self.get_single_train_loss_wo_wd(p, train_index, state))(
              params)

  def get_grad_multi_train_loss_wo_wd(self, params, train_index):
    return grad(lambda p: self.get_multi_train_loss_wo_wd(p, train_index))(
        params)

  def get_grad_single_test_loss_wo_wd(self, params, test_index):
    return grad(lambda p: self.get_single_test_loss_wo_wd(p, test_index))(
        params)

  def compute_strain_cg(self,
                        params,
                        train_index,
                        eps=None,
                        gnh=False,
                        group=False,
                        state=None):
    config.update("jax_enable_x64", True)

    float64_params = make_float64(params)
    if state is not None:
      state = make_float64(state)

    if group:
      v = self.get_grad_multi_train_loss_wo_wd(params, train_index)
    else:
      v = self.get_grad_single_train_loss_wo_wd(params, train_index, state)

    def hvp_fnc(x):
      x = make_float64(x)
      return _add(
          hvp(lambda p: self.get_train_loss_with_wd(p, state),
              (float64_params,), (x,)),
          jax.tree_map(lambda p: p * self.damping, x))

    def net_fnc(p):
      if state is None:
        return self.model.apply(p, self.train_inputs)
      else:
        outputs, _ = self.model.apply(p,
                                      state,
                                      jax.random.PRNGKey(42),
                                      self.train_inputs,
                                      is_training=False)
        return outputs

    def loss_fnc(y):
      if state is None:
        return self.loss_fnc(None, None, None, self.train_targets, outputs=y)
      else:
        return self.loss_fnc(
            None, None, None, None, self.train_targets, outputs=y)

    def gnhvp_fnc(x):
      x = make_float64(x)
      return _add(
          gnhvp(lambda y: loss_fnc(y),
                lambda p: net_fnc(p), (float64_params,), (x,)),
          jax.tree_map(lambda p: p * (self.damping + self.wd), x))

    cg_result = jcg(
        A=hvp_fnc if not gnh else gnhvp_fnc,
        b=make_float64(v),
        x0=make_float64(tree_zeros_like(v)),
        tol=1e-8,
        atol=1e-8)
    cg_ihvp = cg_result[0]
    if eps is None:
      cg_influence = _div(cg_ihvp, self.train_inputs.shape[0])
    else:
      cg_influence = _mul(cg_ihvp, eps)
    cg_params = _add(params, cg_influence)
    return make_float32(cg_params), make_float32(cg_ihvp)

  def compute_stest_cg(self, params, test_index, gnh=False):
    config.update("jax_enable_x64", True)

    float64_params = make_float64(params)
    v = self.get_grad_single_test_loss_wo_wd(params, test_index)

    def hvp_fnc(x):
      x = make_float64(x)
      return _add(
          hvp(lambda p: self.get_train_loss_with_wd(p), (float64_params,),
              (x,)),
          jax.tree_map(lambda p: p * self.damping, x))

    def net_fnc(p):
      return self.model.apply(p, self.train_inputs)

    def loss_fnc(y):
      return self.loss_fnc(None, None, None, self.train_targets, outputs=y)

    def gnhvp_fnc(x):
      x = make_float64(x)
      return _add(
          gnhvp(lambda y: loss_fnc(y),
                lambda p: net_fnc(p), (float64_params,), (x,)),
          jax.tree_map(lambda p: p * (self.damping + self.wd), x))

    cg_result = jcg(
        A=hvp_fnc if not gnh else gnhvp_fnc,
        b=make_float64(v),
        x0=make_float64(tree_zeros_like(v)),
        tol=1e-8,
        atol=1e-8)
    cg_ihvp = cg_result[0]
    cg_influence = _div(cg_ihvp, self.train_inputs.shape[0])
    cg_params = _add(params, cg_influence)
    return make_float32(cg_params), make_float32(cg_ihvp)

  def compute_strain_lissa(self,
                           params,
                           train_index,
                           eps=None,
                           gnh=False,
                           group=False,
                           state=None):

    def hvp_iter(fun_in):
      h_estimate, (h_input, h_target) = fun_in
      if state is None:
        hv = _add(
            hvp(lambda p: self.get_loss_with_wd(p, h_input, h_target),
                (params,), (h_estimate,)),
            jax.tree_map(lambda p: p * self.damping, h_estimate))
      else:
        hv = _add(
            hvp(
                lambda p: self.get_loss_with_wd_state(
                    p, state, h_input, h_target), (params,), (h_estimate,)),
            jax.tree_map(lambda p: p * self.damping, h_estimate))
      h_estimate = _sub(_add(v, h_estimate), _div(hv, self.scale))
      return h_estimate, None

    def net_fnc(p, inputs):
      if len(inputs.shape) == 3:
        inputs = jnp.expand_dims(inputs, 0)
      if state is None:
        return self.model.apply(p, inputs)
      else:
        outputs, _ = self.model.apply(p,
                                      state,
                                      jax.random.PRNGKey(self.seed),
                                      inputs,
                                      is_training=False)
        return outputs

    def loss_fnc(y, targets):
      return self.loss_fnc(None, None, None, targets, outputs=y)

    def gnhvp_iter(fun_in):
      h_estimate, (h_input, h_target) = fun_in
      hv = _add(
          gnhvp(lambda y: loss_fnc(y, h_target),
                lambda p: net_fnc(p, h_input), (params,), (h_estimate,)),
          jax.tree_map(lambda p: p * (self.damping + self.wd), h_estimate))
      h_estimate = _sub(_add(v, h_estimate), _div(hv, self.scale))
      return h_estimate, None

    def noop(fun_in):
      h_estimate, _ = fun_in
      return h_estimate, None

    def main_body(h_estimate, batch):
      isnan = jnp.isnan(ravel_pytree(h_estimate)[0][0])
      return lax.cond(isnan, (h_estimate, None),
                      noop, (h_estimate, batch),
                      hvp_iter if not gnh else gnhvp_iter)

    r_estimate = tree_zeros_like(params)
    if group:
      v = self.get_grad_multi_train_loss_wo_wd(params, train_index)
    else:
      v = self.get_grad_single_train_loss_wo_wd(
          params, train_index, state=state)

    for i in range(self.repeat):
      self.key, subkey = jax.random.split(self.key)
      choices = jax.random.choice(subkey,
                                  jnp.arange(len(self.train_inputs)),
                                  (self.recursion_depth,))
      h_inputs = self.train_inputs[choices]
      h_targets = self.train_targets[choices]

      if group:
        h_estimate = self.get_grad_multi_train_loss_wo_wd(params, train_index)
      else:
        h_estimate = self.get_grad_single_train_loss_wo_wd(
            params, train_index, state=state)
      h_estimate, _ = lax.scan(main_body, h_estimate, (h_inputs, h_targets))
      r_estimate = _add(r_estimate, _div(h_estimate, self.scale))

    r_estimate = _div(r_estimate, self.repeat)
    if eps is None:
      lissa_influence = _div(r_estimate, self.train_inputs.shape[0])
    else:
      lissa_influence = _mul(r_estimate, eps)
    lissa_params = _add(params, lissa_influence)
    return lissa_params, r_estimate

  def compute_stest_lissa(self, params, test_index, gnh=False):

    def hvp_iter(fun_in):
      h_estimate, (h_input, h_target) = fun_in
      hv = _add(
          hvp(lambda p: self.get_loss_with_wd(p, h_input, h_target), (params,),
              (h_estimate,)),
          jax.tree_map(lambda p: p * self.damping, h_estimate))
      h_estimate = _sub(_add(v, h_estimate), _div(hv, self.scale))
      return h_estimate, None

    def net_fnc(p, inputs):
      if len(inputs.shape) == 3:
        inputs = jnp.expand_dims(inputs, 0)
      return self.model.apply(p, inputs)

    def loss_fnc(y, targets):
      return self.loss_fnc(None, None, None, targets, outputs=y)

    def gnhvp_iter(fun_in):
      h_estimate, (h_input, h_target) = fun_in
      hv = _add(
          gnhvp(lambda y: loss_fnc(y, h_target),
                lambda p: net_fnc(p, h_input), (params,), (h_estimate,)),
          jax.tree_map(lambda p: p * (self.damping + self.wd), h_estimate))
      h_estimate = _sub(_add(v, h_estimate), _div(hv, self.scale))
      return h_estimate, None

    def noop(fun_in):
      h_estimate, _ = fun_in
      return h_estimate, None

    def main_body(h_estimate, batch):
      isnan = jnp.isnan(ravel_pytree(h_estimate)[0][0])
      return lax.cond(isnan, (h_estimate, None),
                      noop, (h_estimate, batch),
                      hvp_iter if not gnh else gnhvp_iter)

    r_estimate = tree_zeros_like(params)
    v = self.get_grad_single_test_loss_wo_wd(params, test_index)

    for i in range(self.repeat):
      self.key, subkey = jax.random.split(self.key)
      choices = jax.random.choice(subkey,
                                  jnp.arange(len(self.train_inputs)),
                                  (self.recursion_depth,))
      h_inputs = self.train_inputs[choices]
      h_targets = self.train_targets[choices]

      h_estimate = self.get_grad_single_test_loss_wo_wd(params, test_index)
      h_estimate, _ = lax.scan(main_body, h_estimate, (h_inputs, h_targets))
      r_estimate = _add(r_estimate, _div(h_estimate, self.scale))

    r_estimate = _div(r_estimate, self.repeat)
    lissa_influence = _div(r_estimate, self.train_inputs.shape[0])
    lissa_params = _add(params, lissa_influence)
    return lissa_params, r_estimate


  def compute_single_test_stest_influence(self,
                                          params,
                                          test_index,
                                          stest=None,
                                          use_cg=True,
                                          gnh=False):
    if stest is None:
      if use_cg:
        stest = self.compute_stest_cg(params, test_index, gnh=gnh)[1]
      else:
        stest = self.compute_stest_lissa(params, test_index, gnh=gnh)[1]

    test_influence = []
    train_data_size = self.train_inputs.shape[0]
    for train_index in range(train_data_size):
      train_input = jnp.expand_dims(self.train_inputs[train_index], 0)
      train_target = jnp.expand_dims(self.train_targets[train_index], 0)
      train_grads = self.get_grad_single_loss_with_wd(params,
                                                      train_input,
                                                      train_target)
      influence = _vdot(stest, train_grads)
      test_influence.append(-sum(tree_leaves(influence)))
    return jnp.array(test_influence), stest

  def compute_single_test_strain_influence(self,
                                           params,
                                           test_index,
                                           use_cg=True,
                                           gnh=False):
    # Note: this function can be extremely slow.
    test_influence = []
    train_data_size = self.train_inputs.shape[0]
    for train_index in range(train_data_size):
      if use_cg:
        strain = self.compute_strain_cg(params, train_index, gnh=gnh)[1]
      else:
        strain = self.compute_strain_lissa(params, train_index, gnh=gnh)[1]
      test_input = jnp.expand_dims(self.test_inputs[test_index], 0)
      test_target = jnp.expand_dims(self.test_targets[test_index], 0)
      test_grads = self.get_grad_single_loss_wo_wd(params,
                                                   test_input,
                                                   test_target)
      influence = _vdot(strain, test_grads)
      test_influence.append(-sum(tree_leaves(influence)))
    return test_influence
