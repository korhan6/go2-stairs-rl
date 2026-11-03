"""Generador de terreno procedural (heightfield) con escaleras.

El mapa es una grilla de "baldosas" (tiles):
  - filas  = niveles de dificultad (fila 0 = fácil, última = difícil)
  - columnas = tipo de terreno (escaleras arriba, escaleras abajo, rugoso, ...)

En las escaleras el robot aparece en el centro de la baldosa:
  * "stairs_up"   -> pirámide: el centro es la plataforma alta, al caminar
                     hacia afuera BAJA escalones.
  * "stairs_down" -> pirámide invertida (foso): el centro es el fondo, al
                     caminar hacia afuera SUBE escalones.
Como los comandos de velocidad son aleatorios (adelante, atrás, de lado,
girando) el robot practica subir y bajar en todas las direcciones.

Todo es numpy puro, así que se usa igual en el entrenamiento (MJX), en
play.py (MuJoCo normal) y para dibujar el mapa.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Tuple

import numpy as np


@dataclasses.dataclass
class TerrainConfig:
  columns: Tuple[str, ...] = (
      "rough",
      "stairs_up",
      "stairs_down",
      "stairs_up",
      "stairs_down",
      "slope",
      "discrete",
  )
  num_levels: int = 8
  tile_size: float = 5.0          # metros
  border: float = 3.0             # borde plano alrededor del mapa
  resolution: float = 0.05        # metros por celda
  platform: float = 1.2           # plataforma central de las escaleras
  # (fácil, difícil): interpolado linealmente por nivel
  step_height: Tuple[float, float] = (0.05, 0.19)
  step_width: Tuple[float, float] = (0.34, 0.26)
  rough_amp: Tuple[float, float] = (0.01, 0.05)
  slope_grad: Tuple[float, float] = (0.05, 0.40)   # pendiente (m/m)
  discrete_h: Tuple[float, float] = (0.03, 0.13)
  # "box_ring": anillo cuadrado de cajas alrededor del punto de aparición.
  # Caminando hacia afuera en CUALQUIER dirección hay que subirse a una caja
  # de altura box_h y bajarse del otro lado.
  box_h: Tuple[float, float] = (0.08, 0.40)
  ring_inner: float = 0.9         # m del centro al borde interno de la caja
  ring_width: float = 0.8         # ancho de la caja (profundidad)
  seed: int = 0


def preset(name: str, seed: int = 0) -> TerrainConfig:
  """Terrenos predefinidos.

  flat   : plano + algo rugoso. Etapa 1 (aprender a caminar suave rápido).
  stairs : mapa completo con escaleras. Etapa 2 (ciega o con LiDAR).
  climb  : escaleras de hasta 23 cm + cajas de hasta 40 cm + bloques de
           25 cm. Etapa 3, necesita la política con LiDAR (percepción).
  """
  if name == "flat":
    return TerrainConfig(
        columns=("flat", "rough", "flat", "rough"),
        num_levels=4,
        rough_amp=(0.0, 0.03),
        seed=seed,
    )
  if name == "stairs":
    return TerrainConfig(seed=seed)
  if name == "climb":
    return TerrainConfig(
        columns=("rough", "stairs_up", "stairs_down", "box_ring",
                 "box_ring", "discrete", "stairs_down", "slope"),
        num_levels=10,
        step_height=(0.05, 0.23),
        step_width=(0.34, 0.26),
        discrete_h=(0.05, 0.25),
        slope_grad=(0.05, 0.45),
        box_h=(0.08, 0.40),
        seed=seed,
    )
  if name == "climb_only":  # para evaluar / grabar videos
    return TerrainConfig(
        columns=("box_ring", "stairs_down", "stairs_up"), num_levels=10,
        step_height=(0.05, 0.23), step_width=(0.34, 0.26),
        box_h=(0.08, 0.40), seed=seed,
    )
  if name == "stairs_only":  # para evaluar / grabar videos
    return TerrainConfig(
        columns=("stairs_up", "stairs_down"), num_levels=8, seed=seed
    )
  raise ValueError(f"Terreno desconocido: {name}")


@dataclasses.dataclass
class Terrain:
  config: TerrainConfig
  heights: np.ndarray        # (ny, nx) alturas en metros (z mundo)
  x0: float                  # x de la celda (0, 0)
  y0: float                  # y de la celda (0, 0)
  resolution: float
  spawn: np.ndarray          # (num_levels * ncols, 3) puntos de aparición
  spawn_level: np.ndarray    # (N,) nivel de cada punto
  spawn_column: np.ndarray   # (N,) columna de cada punto
  half_x: float
  half_y: float

  @property
  def shape(self):
    return self.heights.shape

  def height_at(self, x, y):
    """Altura bilineal (numpy)."""
    fx = (np.asarray(x) - self.x0) / self.resolution
    fy = (np.asarray(y) - self.y0) / self.resolution
    ny, nx = self.heights.shape
    c0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
    r0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
    tx = np.clip(fx - c0, 0.0, 1.0)
    ty = np.clip(fy - r0, 0.0, 1.0)
    h = self.heights
    return (
        (1 - tx) * (1 - ty) * h[r0, c0]
        + tx * (1 - ty) * h[r0, c0 + 1]
        + (1 - tx) * ty * h[r0 + 1, c0]
        + tx * ty * h[r0 + 1, c0 + 1]
    )

  def find_spawn(self, column: str, level: int) -> np.ndarray:
    cols = self.config.columns
    for i, (lv, c) in enumerate(zip(self.spawn_level, self.spawn_column)):
      if lv == level and cols[c] == column:
        return self.spawn[i]
    raise ValueError(f"No hay baldosa {column} nivel {level}")


def _lerp(rng: Tuple[float, float], t: float) -> float:
  return rng[0] + (rng[1] - rng[0]) * t


def _tile_pyramid_stairs(n, res, cfg, t, sign):
  """Escaleras en pirámide cuadrada. sign=+1 arriba, -1 foso."""
  h = _lerp(cfg.step_height, t)
  w = _lerp(cfg.step_width, t)
  coords = (np.arange(n) + 0.5) * res - cfg.tile_size / 2.0
  xx, yy = np.meshgrid(coords, coords)
  d = np.maximum(np.abs(xx), np.abs(yy))  # distancia Chebyshev al centro
  half_p = cfg.platform / 2.0
  n_steps = int(np.floor((cfg.tile_size / 2.0 - half_p - 0.1) / w))
  k = np.floor(np.maximum(d - half_p, 0.0) / w)  # escalones desde arriba
  k = np.where(d <= half_p, 0, k + 1)
  level = np.clip(n_steps - k, 0, n_steps)
  return sign * level * h


def _tile_slope(n, res, cfg, t):
  g = _lerp(cfg.slope_grad, t)
  coords = (np.arange(n) + 0.5) * res - cfg.tile_size / 2.0
  xx, yy = np.meshgrid(coords, coords)
  d = np.maximum(np.abs(xx), np.abs(yy))
  half_p = cfg.platform / 2.0
  top = (cfg.tile_size / 2.0 - half_p) * g
  return np.clip(top - (d - half_p) * g, 0.0, top)


def _tile_rough(n, res, cfg, t, rng):
  amp = _lerp(cfg.rough_amp, t)
  block = 2  # celdas (10 cm)
  nb = n // block + 1
  coarse = rng.uniform(-amp, amp, size=(nb, nb))
  fine = np.kron(coarse, np.ones((block, block)))[:n, :n]
  return np.round(fine / 0.005) * 0.005


def _tile_box_ring(n, res, cfg, t):
  h = _lerp(cfg.box_h, t)
  coords = (np.arange(n) + 0.5) * res - cfg.tile_size / 2.0
  xx, yy = np.meshgrid(coords, coords)
  d = np.maximum(np.abs(xx), np.abs(yy))
  ring = (d >= cfg.ring_inner) & (d < cfg.ring_inner + cfg.ring_width)
  return np.where(ring, h, 0.0)


def _tile_discrete(n, res, cfg, t, rng):
  hmax = _lerp(cfg.discrete_h, t)
  out = np.zeros((n, n))
  for _ in range(18):
    sx, sy = rng.uniform(0.4, 1.0, size=2)
    cx, cy = rng.uniform(0, cfg.tile_size, size=2)
    i0 = int(max(0, (cy - sy / 2) / res))
    i1 = int(min(n, (cy + sy / 2) / res))
    j0 = int(max(0, (cx - sx / 2) / res))
    j1 = int(min(n, (cx + sx / 2) / res))
    out[i0:i1, j0:j1] = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0) * hmax
  # plataforma central plana para aparecer
  c = n // 2
  r = int(cfg.platform / 2 / res)
  out[c - r : c + r, c - r : c + r] = 0.0
  return out


def generate(cfg: TerrainConfig) -> Terrain:
  rng = np.random.default_rng(cfg.seed)
  res = cfg.resolution
  n_tile = int(round(cfg.tile_size / res))
  n_border = int(round(cfg.border / res))
  ncols = len(cfg.columns)
  nlev = cfg.num_levels
  nx = ncols * n_tile + 2 * n_border + 1
  ny = nlev * n_tile + 2 * n_border + 1
  heights = np.zeros((ny, nx), dtype=np.float64)

  half_x = (nx - 1) * res / 2.0
  half_y = (ny - 1) * res / 2.0
  x0, y0 = -half_x, -half_y

  spawn, spawn_level, spawn_col = [], [], []
  for lv in range(nlev):
    t = lv / max(nlev - 1, 1)
    for ci, kind in enumerate(cfg.columns):
      if kind == "flat":
        tile = np.zeros((n_tile, n_tile))
      elif kind == "rough":
        tile = _tile_rough(n_tile, res, cfg, t, rng)
      elif kind == "stairs_up":
        tile = _tile_pyramid_stairs(n_tile, res, cfg, t, +1.0)
      elif kind == "stairs_down":
        tile = _tile_pyramid_stairs(n_tile, res, cfg, t, -1.0)
      elif kind == "slope":
        tile = _tile_slope(n_tile, res, cfg, t)
      elif kind == "box_ring":
        tile = _tile_box_ring(n_tile, res, cfg, t)
      elif kind == "discrete":
        tile = _tile_discrete(n_tile, res, cfg, t, rng)
      else:
        raise ValueError(kind)
      r0 = n_border + lv * n_tile
      c0 = n_border + ci * n_tile
      heights[r0 : r0 + n_tile, c0 : c0 + n_tile] = tile
      cx = x0 + (c0 + n_tile / 2.0) * res
      cy = y0 + (r0 + n_tile / 2.0) * res
      spawn.append([cx, cy, 0.0])
      spawn_level.append(lv)
      spawn_col.append(ci)

  terrain = Terrain(
      config=cfg,
      heights=heights.astype(np.float32),
      x0=x0,
      y0=y0,
      resolution=res,
      spawn=np.array(spawn, dtype=np.float32),
      spawn_level=np.array(spawn_level, dtype=np.int32),
      spawn_column=np.array(spawn_col, dtype=np.int32),
      half_x=half_x,
      half_y=half_y,
  )
  terrain.spawn[:, 2] = terrain.height_at(terrain.spawn[:, 0], terrain.spawn[:, 1])
  return terrain


def hfield_arrays(terrain: Terrain) -> Dict[str, object]:
  """Datos para el <hfield> de MuJoCo.

  MuJoCo guarda la elevación normalizada en [0, 1] (fila = y, col = x) y la
  escala por size[2]. El geom se ubica en z = hmin.
  """
  h = terrain.heights.astype(np.float64)
  hmin, hmax = float(h.min()), float(h.max())
  span = max(hmax - hmin, 1e-3)
  data = ((h - hmin) / span).astype(np.float32)
  ny, nx = h.shape
  return dict(
      data=data,
      nrow=ny,
      ncol=nx,
      size=(terrain.half_x, terrain.half_y, span, 0.5),
      z_offset=hmin,
  )
