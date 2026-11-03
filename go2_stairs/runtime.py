"""Ejecuta la política exportada (policy.npz) solo con numpy.

Se usa EXACTAMENTE el mismo código en:
  * play.py      -> MuJoCo normal (sim2sim)
  * deploy_go2.py -> robot real (unitree_sdk2py)
Así, si funciona en play.py, las observaciones del robot real se arman igual.
Sin JAX: corre en la laptop, en una Jetson o en la Raspberry Pi.
"""

from __future__ import annotations

import json

import numpy as np

from go2_stairs import contact as contact_lib
from go2_stairs import perception


def _swish(x):
  return x / (1.0 + np.exp(-x))


def projected_gravity_from_quat(q_wxyz) -> np.ndarray:
  """Gravedad en el marco del cuerpo a partir del cuaternión (w, x, y, z)."""
  w, x, y, z = q_wxyz
  # R^T @ [0, 0, -1]  (tercera fila de R, negada)
  return -np.array([
      2 * (x * z - w * y),
      2 * (y * z + w * x),
      1 - 2 * (x * x + y * y),
  ], dtype=np.float32)


class PolicyRunner:
  """Arma observaciones, mantiene historial/reloj y evalúa la red."""

  def __init__(self, npz_path: str):
    z = np.load(npz_path, allow_pickle=False)
    self.meta = json.loads(str(z["meta"]))
    m = self.meta
    n_layers = m["num_layers"]
    self.layers = [(z[f"W{i}"].astype(np.float32), z[f"b{i}"].astype(np.float32))
                   for i in range(n_layers)]
    self.obs_mean = z["obs_mean"].astype(np.float32)
    self.obs_std = z["obs_std"].astype(np.float32)
    self.default_pose = np.array(m["default_pose"], dtype=np.float32)
    self.action_scale = np.array(m["action_scale"], dtype=np.float32)
    self.history_len = int(m["history_len"])
    self.frame_size = int(m["frame_size"])
    self.dt = float(m["ctrl_dt"])
    self.freq = float(m["gait_freq"])
    # "torque" (estimado con tau_est), "sensor" (foot_force) o "none"
    self.contact_source = m.get(
        "contact_source", "sensor" if m.get("actor_contacts", True) else "none")
    self.contact_threshold = float(
        m.get("contact_threshold", contact_lib.DEFAULT_THRESHOLD))
    self.use_contacts = self.contact_source != "none"
    self.last_foot_fz = np.zeros(4, dtype=np.float32)
    self.Kp = float(m["Kp"])
    self.Kd = float(m["Kd"])
    # Percepción (LiDAR -> mapa de alturas). Políticas viejas: ciegas.
    self.perceptive = bool(m.get("perception", {}).get("enable", False))
    self.scan_pts = perception.grid_points()
    self.n_scan = perception.NUM_POINTS
    self.last_scan = None  # (h_rel, valido) del último paso, para dibujar
    self.reset()

  def reset(self, phase: float = 0.0):
    self.phase = float(phase)
    self.last_action = np.zeros(12, dtype=np.float32)
    self.history = None

  def forward(self, obs: np.ndarray) -> np.ndarray:
    x = (obs - self.obs_mean) / self.obs_std
    for i, (W, b) in enumerate(self.layers):
      x = x @ W + b
      if i < len(self.layers) - 1:
        x = _swish(x)
    return np.tanh(x[:12])  # moda de la distribución tanh-normal

  def make_frame(self, gyro, gravity, command, qpos, qvel, contacts):
    gait = [np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)]
    c = np.asarray(contacts, dtype=np.float32)
    if not self.use_contacts:
      c = np.zeros(4, dtype=np.float32)
    return np.concatenate([
        np.asarray(gyro, np.float32),
        np.asarray(gravity, np.float32),
        np.asarray(command, np.float32),
        np.asarray(qpos, np.float32) - self.default_pose,
        np.asarray(qvel, np.float32),
        self.last_action,
        np.asarray(gait, np.float32),
        c,
    ]).astype(np.float32)

  def contacts_from_torque(self, qpos, tau, gravity):
    """Contacto por pata a partir del torque de los motores (tau_est)."""
    c, fz = contact_lib.estimate_contacts(
        np, np.asarray(qpos, np.float64), np.asarray(tau, np.float64),
        np.asarray(gravity, np.float64), self.contact_threshold)
    self.last_foot_fz = fz.astype(np.float32)
    return c

  def scan_from_height_fn(self, height_fn, base_pos, base_rot,
                          occlusion: bool = True):
    """Escaneo (h_rel, valido) a partir de una función de altura del mundo.

    height_fn: xy (..., 2) -> altura. En simulación es el terreno; en el
    robot real es la consulta a tu mapa de elevación (NaN = sin dato).
    base_pos: posición del cuerpo en el MISMO marco que el mapa (odometría).
    base_rot: matriz de rotación 3x3 del cuerpo en ese marco.
    """
    base_pos = np.asarray(base_pos, np.float64)
    base_rot = np.asarray(base_rot, np.float64).reshape(3, 3)
    yaw = perception.yaw_from_mat(np, base_rot)
    w = perception.grid_world(np, self.scan_pts, base_pos[:2], yaw)
    h = np.asarray(height_fn(w), np.float64)
    valid = np.isfinite(h)
    h = np.where(valid, h, 0.0)
    if occlusion:
      hf = lambda xy: np.nan_to_num(np.asarray(height_fn(xy), np.float64),
                                    nan=-10.0)
      valid &= perception.visibility(np, hf, base_pos, base_rot, w, h)
    return perception.heights_to_obs(np, base_pos[2], h), valid

  def act(self, gyro, gravity, command, qpos, qvel, tau=None, contacts=None,
          height_scan=None, scan_valid=None):
    """Todo en orden de la POLÍTICA (FL, FR, RL, RR).

    tau: torque medido de los 12 motores (tau_est en el robot real,
         actuator_force en simulación). Se usa si contact_source == "torque".
    contacts: contacto booleano por pata (solo si contact_source == "sensor").
    height_scan, scan_valid: (187,) alturas relativas y máscara (ver
         scan_from_height_fn / perception.py). Si la política usa LiDAR y no
         pasas nada, corre en modo "sin mapa" (todo inválido), que también
         se entrenó: camina, pero sin anticipar escalones.
    Devuelve (objetivos_articulares, acción). Llamar a 1/ctrl_dt Hz (50 Hz).
    """
    if self.contact_source == "torque":
      if tau is None:
        raise ValueError("Esta política estima el contacto con el torque: "
                         "pasa tau=")
      contacts = self.contacts_from_torque(qpos, tau, gravity)
    elif contacts is None:
      contacts = np.zeros(4, dtype=bool)
    frame = self.make_frame(gyro, gravity, command, qpos, qvel, contacts)
    if self.history is None:
      self.history = np.tile(frame, (self.history_len, 1))
    else:
      self.history = np.concatenate([self.history[1:], frame[None]], 0)
    obs = self.history.ravel()
    if self.perceptive:
      if height_scan is None:
        height_scan = np.zeros(self.n_scan, np.float32)
        scan_valid = np.zeros(self.n_scan, bool)
      elif scan_valid is None:
        scan_valid = np.ones(self.n_scan, bool)
      hs = np.asarray(height_scan, np.float32).reshape(-1)
      sv = np.asarray(scan_valid, bool).reshape(-1)
      self.last_scan = (hs, sv)
      obs = np.concatenate(
          [obs, perception.to_policy_input(np, hs, sv).astype(np.float32)])
    action = self.forward(obs).astype(np.float32)
    self.last_action = action
    self.phase = (self.phase + self.dt * self.freq) % 1.0
    targets = self.default_pose + action * self.action_scale
    return targets, action
