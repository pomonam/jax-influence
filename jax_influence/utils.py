""" Utilities for influence functions computations."""

from functools import partial
import operator

import jax
from jax import grad
from jax import jacfwd
from jax import jacrev
from jax import jit
from jax import jvp
from jax import lax
from jax import vjp
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from jax.tree_util import tree_leaves
from jax.tree_util import tree_map


def make_float64(params):
  return jax.tree_map(lambda x: x.astype(jnp.float64), params)


def make_float32(params):
  return jax.tree_map(lambda x: x.astype(jnp.float32), params)


def tree_zeros_like(tree):
  return jax.tree_map(jnp.zeros_like, tree)


def _mul(scalar, tree):
  return tree_map(partial(operator.mul, scalar), tree)


def _div(tree, scalar):
  return tree_map(lambda x: operator.truediv(x, scalar), tree)


def _vdot(x, y):
  f = partial(jnp.vdot, precision=lax.Precision.HIGHEST)
  return sum(tree_leaves(tree_map(f, x, y)))


_add = partial(tree_map, operator.add)
_sub = partial(tree_map, operator.sub)
_tree_mul = partial(tree_map, operator.mul)


def flatten(params):
  return ravel_pytree(params)[0]


def leaves_to_jndarray(pytree):
  """Converts leaves of pytree to jax.numpy arrays."""
  return jax.tree_map(jnp.array, pytree)


@partial(jit, static_argnums=(0,))
def hvp(f, primals, tangents):
  return jvp(grad(f), primals, tangents)[1]


def hessian(f):
  return jit(jacfwd(jacrev(f)))


@partial(jit, static_argnums=(0, 1))
def gnhvp(g, f, primals, tangents):
  z, r_z = jvp(f, primals, tangents)
  r_gz = hvp(g, (z,), (r_z,))
  _, f_vjp = vjp(f, primals[0])
  return f_vjp(r_gz)[0]
