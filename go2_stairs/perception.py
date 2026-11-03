"""LiDAR L1 simulado -> escaneo de alturas alrededor del robot.

Qué ve la política
------------------
Una grilla de 17 x 11 puntos (cada 10 cm, de -0.8 a +0.8 m adelante/atrás y
-0.5 a +0.5 m a los lados), alineada con el RUMBO (yaw) del robot pero no con
su cabeceo/alabeo. Para cada punto la política recibe:

    h_rel = z_base - z_terreno - 0.30      (recortado a [-1, 1])
    valido = 1 si el LiDAR "ve" ese punto, 0 si no

0 = piso a la altura normal; NEGATIVO = algo más alto que el piso (un
escalón, una caja); POSITIVO = un hueco / escalón hacia abajo.

Por qué así
-----------
En el robot real el L1 no da una grilla: da una nube de puntos. Lo normal es
armar un mapa de elevación (p. ej. elevation_mapping / GLIM + odometría) y
muestrear ESTA MISMA grilla alrededor del robot. Para que lo aprendido en
simulación sirva allá, en el entrenamiento la grilla se arruina igual que un
mapa real:

  * Oclusión: el L1 va en la cabeza (pos/orientación del URDF de Unitree,
    link "radar"). Si la línea entre el sensor y un punto choca con el
    terreno (p. ej. detrás del borde de una caja alta) ese punto no se ve.
  * Campo de visión: hemisferio del L1 (360° x 90°) apuntando hacia abajo y
    un poco hacia adelante.
  * Ruido, puntos falsos, huecos, deriva de la odometría (desfase en z/xy/yaw
    por episodio) y episodios completamente "ciegos" (sin mapa), para que la
    política siga caminando si el mapa se cae. Eso está en env.py.

El código usa `xp` (numpy o jax.numpy), así que es EL MISMO en el
entrenamiento, en play.py y en el robot.
"""

from __future__ import annotations

import numpy as np

GRID_X = np.round(np.linspace(-0.8, 0.8, 17), 3)   # adelante (+) / atrás (-)
GRID_Y = np.round(np.linspace(-0.5, 0.5, 11), 3)   # izquierda (+) / derecha (-)
NUM_POINTS = GRID_X.size * GRID_Y.size             # 187

# LiDAR L1 en el marco del cuerpo (URDF go2_description, link "radar":
# xyz="0.28945 0 -0.046825" rpy="0 2.8782 0"). El eje del hemisferio de
# visión es el +z del sensor, que con ese cabeceo apunta abajo/adelante.
LIDAR_POS = (0.28945, 0.0, -0.046825)
_P = 2.8782
LIDAR_AXIS = (float(np.sin(_P)), 0.0, float(np.cos(_P)))  # (0.26, 0, -0.97)

NOMINAL_HEIGHT = 0.30   # = reward_config.base_height_target
CLIP = 1.0


def grid_points() -> np.ndarray:
  """(N, 2) puntos de la grilla en el marco "yaw" del robot. Orden: fila = y."""
  gx, gy = np.meshgrid(GRID_X, GRID_Y)
  return np.stack([gx.ravel(), gy.ravel()], -1).astype(np.float32)


def yaw_from_mat(xp, mat):
  return xp.arctan2(mat[1, 0], mat[0, 0])


def grid_world(xp, pts, base_xy, yaw):
  """Puntos de la grilla en coordenadas del mundo (N, 2)."""
  c, s = xp.cos(yaw), xp.sin(yaw)
  x = base_xy[0] + c * pts[:, 0] - s * pts[:, 1]
  y = base_xy[1] + s * pts[:, 0] + c * pts[:, 1]
  return xp.stack([x, y], -1)


def heights_to_obs(xp, base_z, h):
  """Altura del terreno -> lo que ve la política."""
  return xp.clip(base_z - h - NOMINAL_HEIGHT, -CLIP, CLIP)


def visibility(xp, height_fn, base_pos, base_mat, pts_world, h,
               n_samples: int = 10, margin: float = 0.015):
  """True si el L1 ve cada punto (campo de visión + oclusión por el terreno).

  height_fn: xy (..., 2) -> altura del terreno (...,).
  base_mat: matriz de rotación 3x3 del cuerpo (mundo <- cuerpo).
  """
  lp = xp.asarray(LIDAR_POS, dtype=xp.float32)
  la = xp.asarray(LIDAR_AXIS, dtype=xp.float32)
  origin = base_pos + base_mat @ lp
  axis = base_mat @ la
  # Si la superficie está MÁS ALTA que el sensor (caja de 40 cm), el L1 ve
  # la cara vertical: el mapa marca ese borde con su altura, pero lo que hay
  # más atrás encima de la caja queda oculto. Por eso se apunta como mucho a
  # la altura del sensor.
  tz = xp.minimum(h + 0.01, origin[2] - 0.03)
  target = xp.concatenate([pts_world, tz[:, None]], -1)  # (N, 3)
  d = target - origin[None, :]
  in_fov = (d @ axis) >= 0.0
  s = xp.asarray(np.linspace(0.05, 1.0, n_samples), dtype=xp.float32)
  q = origin[None, None, :] + s[None, :, None] * d[:, None, :]   # (N, S, 3)
  hq = height_fn(q[..., :2])                                       # (N, S)
  # los últimos ~7 cm del rayo son la propia celda (arista del escalón)
  horiz = xp.sqrt(xp.sum(xp.square(d[:, :2]), -1))
  near_target = (1.0 - s[None, :]) * horiz[:, None] < 0.07
  blocked = xp.any((hq > q[..., 2] + margin) & ~near_target, axis=1)
  return in_fov & ~blocked


def clean_scan(xp, height_fn, base_pos, base_mat, pts):
  """Escaneo ideal (sin ruido ni oclusión): para el crítico y depuración."""
  yaw = yaw_from_mat(xp, base_mat)
  w = grid_world(xp, pts, base_pos[:2], yaw)
  return heights_to_obs(xp, base_pos[2], height_fn(w))


def lidar_scan(xp, height_fn, base_pos, base_mat, pts, n_samples: int = 10):
  """Escaneo como lo vería el L1 sin ruido: (h_rel (N,), valido (N,) bool)."""
  yaw = yaw_from_mat(xp, base_mat)
  w = grid_world(xp, pts, base_pos[:2], yaw)
  h = height_fn(w)
  vis = visibility(xp, height_fn, base_pos, base_mat, w, h, n_samples)
  return heights_to_obs(xp, base_pos[2], h), vis


def to_policy_input(xp, h_rel, valid):
  """Lo que entra a la red: alturas (0 donde no hay dato) + máscara."""
  v = valid.astype(xp.float32)
  return xp.concatenate([xp.where(valid, h_rel, 0.0), v])
