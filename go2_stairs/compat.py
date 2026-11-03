"""Parches de compatibilidad entre versiones de JAX y Brax.

Brax 0.14.x todavía llama a `jax.device_put_replicated`, que JAX quitó en la
v0.10. Este parche lo reemplaza por el equivalente oficial (apilar una copia
por dispositivo; `pmap` se encarga de repartirlas). No hace nada si tu JAX
todavía lo trae. Importar ANTES de usar brax.training.
"""

import jax
import jax.numpy as jp


def _device_put_replicated(x, devices):
  n = len(devices)
  return jax.tree.map(
      lambda a: jax.device_put(jp.stack([jp.asarray(a)] * n), devices[0]), x)


def apply():
  try:
    jax.device_put_replicated  # pylint: disable=pointless-statement
    ok = True
  except AttributeError:
    ok = False
  if not ok:
    jax.device_put_replicated = _device_put_replicated


apply()
