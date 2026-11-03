#!/usr/bin/env python3
"""Prueba la política en MuJoCo normal (sim2sim) con visor y teclado.

  python play.py --policy runs/stairs_XXXX/policy.npz
  python play.py --policy ... --tile stairs_up --level 6      # escaleras altas
  python play.py --policy ... --video demo.mp4                # video sin visor
  python play.py --policy ... --terrain climb_only --tile box_ring --level 9
                                                    # cajas de 40 cm (LiDAR)

Teclado (en la ventana del visor):
  ↑ / ↓        velocidad adelante / atrás (vx)
  ← / →        girar izquierda / derecha (wz)
  ,  /  .      caminar de lado izquierda / derecha (vy)
  Backspace    parar (comando cero)
  Enter        LEVANTAR / soltar al perro (verás las "pataditas")
  Insert       EMPUJÓN fuerte en una dirección al azar
  Ctrl + clic derecho y arrastrar sobre el perro = empujarlo / levantarlo a mano

Usa el mismo PolicyRunner que deploy_go2.py, así que si aquí camina bien,
las observaciones del robot real se arman igual. El contacto de cada pata se
estima con el torque de los motores (como en el Go2 Air real); la línea de
estado muestra la fuerza vertical estimada por pata en newtons.

Si la política usa LiDAR, en el visor se dibujan los 187 puntos del escaneo:
verde = lo ve, rojo = oculto (detrás de un borde) o fuera del campo de visión.
`--blind` corre la política sin mapa (como si el LiDAR fallara).
"""

import argparse
import os
import sys
import time

if "--video" in sys.argv:  # render sin ventana (antes de importar mujoco)
  os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from go2_stairs import constants as C  # noqa: E402
from go2_stairs import robot_xml  # noqa: E402
from go2_stairs import terrain as terrain_lib  # noqa: E402
from go2_stairs.runtime import PolicyRunner  # noqa: E402

KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
KEY_ENTER, KEY_BACKSPACE, KEY_COMMA, KEY_PERIOD = 257, 259, 44, 46
KEY_INSERT = 260  # empujón en dirección al azar


class Sim:
  """Go2 en MuJoCo nativo controlado por la política exportada."""

  def __init__(self, policy_path, terrain_name, tile, level, yaw=0.0,
               blind=False):
    self.runner = PolicyRunner(policy_path)
    self.blind = blind
    meta = self.runner.meta
    self.terrain = terrain_lib.generate(terrain_lib.preset(terrain_name))
    self.model = robot_xml.build_model(self.terrain, meta["Kp"], meta["Kd"],
                                       meta["sim_dt"])
    self.data = mujoco.MjData(self.model)
    self.n_sub = int(round(meta["ctrl_dt"] / meta["sim_dt"]))
    self.ctrl_dt = meta["ctrl_dt"]
    m = self.model
    self.imu = m.site("imu").id
    self.base = m.body("base").id
    self.mass = float(m.body_subtreemass[self.base])
    self.contact_adr = [m.sensor_adr[m.sensor(f"{l}_floor_found").id]
                        for l in C.LEGS]
    self.gyro_adr = m.sensor_adr[m.sensor("gyro").id]
    self.linvel_adr = m.sensor_adr[m.sensor("global_linvel").id]
    self.angvel_adr = m.sensor_adr[m.sensor("global_angvel").id]
    self.command = np.zeros(3, dtype=np.float32)
    self.lift_target = None
    self.lift_t = 0.0
    self.push_f = np.zeros(3)
    self.push_left = 0.0
    self.push_vis = 0.0   # s que se sigue viendo la flecha del empujón
    self.push_dir = np.zeros(3)
    self.spawn(tile, level, yaw)

  def spawn(self, tile, level, yaw):
    d = self.data
    mujoco.mj_resetDataKeyframe(self.model, d, self.model.key("home").id)
    p = self.terrain.find_spawn(tile, level)
    d.qpos[0:3] = [p[0], p[1], p[2] + 0.36]
    d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    d.ctrl[:] = C.DEFAULT_POSE
    mujoco.mj_forward(self.model, d)
    self.runner.reset()
    # medio segundo para asentarse en la pose por defecto
    for _ in range(25 * self.n_sub):
      mujoco.mj_step(self.model, d)

  # --- sensores (lo mismo que leerías del LowState real) ---
  def read(self):
    d = self.data
    gyro = d.sensordata[self.gyro_adr:self.gyro_adr + 3].copy()
    gravity = d.site_xmat[self.imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
    contacts = np.array([d.sensordata[a] > 0 for a in self.contact_adr])
    tau = d.actuator_force.copy()  # = tau_est del robot real
    return gyro, gravity, d.qpos[7:].copy(), d.qvel[6:].copy(), tau, contacts

  def toggle_lift(self):
    if self.lift_target is None:
      self.lift_target = self.data.xpos[self.base].copy() + [0, 0, 0.4]
      self.lift_t = 0.0
    else:
      self.lift_target = None

  def push(self, dv=1.0, angle=None, dur=0.15):
    """Empujón: cambia la velocidad del cuerpo en ~dv m/s en `dur` segundos."""
    if angle is None:
      angle = np.random.uniform(-np.pi, np.pi)
    self.push_f = self.mass * dv / dur * np.array(
        [np.cos(angle), np.sin(angle), 0.0])
    self.push_left = dur
    self.push_vis = 0.8
    self.push_dir = np.array([np.cos(angle), np.sin(angle), 0.0]) * dv
    return angle

  def _lift_force(self):
    d = self.data
    d.xfrc_applied[self.base] = 0
    if self.push_left > 0:
      d.xfrc_applied[self.base, :3] = self.push_f
    if self.lift_target is None:
      return
    pos = d.xpos[self.base]
    vel = d.sensordata[self.linvel_adr:self.linvel_adr + 3]
    m = self.mass
    ramp = min(1.0, self.lift_t / 0.3)
    err = self.lift_target - pos
    f = m * np.array([10 * err[0] - 5 * vel[0], 10 * err[1] - 5 * vel[1],
                      9.81 * ramp + 25 * err[2] * ramp - 8 * vel[2]])
    up = d.xmat[self.base].reshape(3, 3)[:, 2]
    angvel = d.sensordata[self.angvel_adr:self.angvel_adr + 3]
    tq = 30 * np.cross(up, [0, 0, 1.0]) - 2 * angvel
    d.xfrc_applied[self.base, :3] = f
    d.xfrc_applied[self.base, 3:] = tq

  def height_fn(self, xy):
    return self.terrain.height_at(xy[..., 0], xy[..., 1])

  def lidar(self):
    """Escaneo de alturas (lo que en el robot daría el mapa del L1)."""
    if not self.runner.perceptive or self.blind:
      return None, None
    d = self.data
    return self.runner.scan_from_height_fn(
        self.height_fn, d.xpos[self.base], d.xmat[self.base].reshape(3, 3))

  def control_step(self):
    gyro, gravity, qpos, qvel, tau, contacts = self.read()
    hs, hv = self.lidar()
    targets, _ = self.runner.act(gyro, gravity, self.command, qpos, qvel,
                                 tau=tau, contacts=contacts,
                                 height_scan=hs, scan_valid=hv)
    self.data.ctrl[:] = targets
    for _ in range(self.n_sub):
      self._lift_force()
      mujoco.mj_step(self.model, self.data)
    if self.lift_target is not None:
      self.lift_t += self.ctrl_dt
    self.push_left = max(0.0, self.push_left - self.ctrl_dt)
    self.push_vis = max(0.0, self.push_vis - self.ctrl_dt)

  def status(self):
    d = self.data
    vel = d.xmat[self.base].reshape(3, 3).T @ d.qvel[0:3]
    g = self.terrain.height_at(d.qpos[0], d.qpos[1])
    return (f"cmd vx={self.command[0]:+.2f} vy={self.command[1]:+.2f} "
            f"wz={self.command[2]:+.2f} | vel {vel[0]:+.2f} {vel[1]:+.2f} | "
            f"altura {d.qpos[2] - g:.2f} m | patas(N) "
            f"{np.round(self.runner.last_foot_fz).astype(int)} | "
            f"{self.scan_status()}"
            f"{'LEVANTADO' if self.lift_target is not None else ''}")


def _scan_status(sim):
  if not sim.runner.perceptive:
    return "ciega | "
  if sim.blind or sim.runner.last_scan is None:
    return "LiDAR OFF | "
  return f"LiDAR {int(sim.runner.last_scan[1].sum())}/187 | "


Sim.scan_status = _scan_status


def perception_yaw(sim: Sim) -> float:
  m = sim.data.xmat[sim.base].reshape(3, 3)
  return float(np.arctan2(m[1, 0], m[0, 0]))


def draw_push(sim: Sim, scn):
  """Flecha roja grande en la dirección del último empujón."""
  if sim.push_vis <= 0 or scn.ngeom >= scn.maxgeom:
    return
  p = sim.data.xpos[sim.base].copy()
  dv = float(np.linalg.norm(sim.push_dir))
  if dv < 1e-6:
    return
  length = 0.25 + 0.35 * dv
  start = p - sim.push_dir / dv * (length + 0.15)
  start[2] = p[2] + 0.1
  end = start + sim.push_dir / dv * length
  g = scn.geoms[scn.ngeom]
  mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                      np.zeros(3), np.eye(3).ravel(),
                      np.array([1.0, 0.1, 0.05, 1.0], np.float32))
  mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.025, start, end)
  scn.ngeom += 1


def draw_scan(sim: Sim, scn):
  """Dibuja los puntos del escaneo (y la flecha del empujón) en el visor."""
  scn.ngeom = 0
  if sim.runner.last_scan is None or sim.blind:
    draw_push(sim, scn)
    return
  from go2_stairs import perception  # pylint: disable=import-outside-toplevel
  d = sim.data
  rot = d.xmat[sim.base].reshape(3, 3)
  yaw = perception.yaw_from_mat(np, rot)
  w = perception.grid_world(np, sim.runner.scan_pts, d.xpos[sim.base][:2], yaw)
  h = sim.height_fn(w)
  valid = sim.runner.last_scan[1]
  for i in range(min(len(w), scn.maxgeom)):
    rgba = (np.array([0.1, 0.9, 0.2, 0.9]) if valid[i]
            else np.array([0.9, 0.15, 0.1, 0.6]))
    mujoco.mjv_initGeom(scn.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.012, 0, 0]),
                        np.array([w[i, 0], w[i, 1], h[i] + 0.012]),
                        np.eye(3).ravel(), rgba.astype(np.float32))
    scn.ngeom = i + 1
  draw_push(sim, scn)


def run_viewer(sim: Sim):
  import mujoco.viewer  # pylint: disable=import-outside-toplevel

  def on_key(key):
    c = sim.command
    if key == KEY_UP:
      c[0] = min(c[0] + 0.1, 1.0)
    elif key == KEY_DOWN:
      c[0] = max(c[0] - 0.1, -1.0)
    elif key == KEY_LEFT:
      c[2] = min(c[2] + 0.2, 1.0)
    elif key == KEY_RIGHT:
      c[2] = max(c[2] - 0.2, -1.0)
    elif key == KEY_COMMA:
      c[1] = min(c[1] + 0.1, 0.6)
    elif key == KEY_PERIOD:
      c[1] = max(c[1] - 0.1, -0.6)
    elif key == KEY_BACKSPACE:
      c[:] = 0
    elif key == KEY_ENTER:
      sim.toggle_lift()
    elif key == KEY_INSERT:
      sim.push(2.0)
    c[np.abs(c) < 0.05] = 0.0

  print(__doc__)
  with mujoco.viewer.launch_passive(sim.model, sim.data,
                                    key_callback=on_key) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    v.cam.trackbodyid = sim.base
    v.cam.distance = 1.8
    v.cam.elevation = -20
    last_print = 0.0
    while v.is_running():
      t0 = time.time()
      with v.lock():
        sim.control_step()
        draw_scan(sim, v.user_scn)
      v.sync()
      if t0 - last_print > 0.5:
        print("\r" + sim.status() + " " * 10, end="", flush=True)
        last_print = t0
      time.sleep(max(0.0, sim.ctrl_dt - (time.time() - t0)))


def run_video(sim: Sim, path: str, seconds: float, width=1280, height=720):
  import mediapy  # pylint: disable=import-outside-toplevel

  # Guion: subir del foso, girar, bajar, y al final levantarlo.
  script = [
      (1.5, (0.0, 0.0, 0.0), False),
      (6.5, (0.5, 0.0, 0.0), False),   # sube escalones
      (1.0, (0.0, 0.0, 0.0), False),
      (2.5, (0.0, 0.0, 1.25), False),  # media vuelta
      (6.5, (0.5, 0.0, 0.0), False),   # baja escalones
      (1.0, (0.0, 0.0, 0.0), False),
      (2.5, (0.0, 0.0, 0.0), True),    # levantado: pataditas
      (2.5, (0.0, 0.0, 0.0), False),   # lo suelta
  ]
  renderer = mujoco.Renderer(sim.model, height=height, width=width)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  cam.trackbodyid = sim.base
  cam.distance, cam.elevation, cam.azimuth = 2.0, -18, 120
  frames, t = [], 0.0
  for dur, cmd, lift in script:
    sim.command[:] = cmd
    if lift != (sim.lift_target is not None):
      sim.toggle_lift()
    for _ in range(int(dur / sim.ctrl_dt)):
      sim.control_step()
      renderer.update_scene(sim.data, camera=cam)
      frames.append(renderer.render())
      t += sim.ctrl_dt
      if t >= seconds:
        break
    print(sim.status())
    if t >= seconds:
      break
  mediapy.write_video(path, frames, fps=int(round(1 / sim.ctrl_dt)))
  print("Video guardado:", path)


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawTextHelpFormatter)
  ap.add_argument("--policy", required=True)
  ap.add_argument("--terrain", default="stairs_only",
                  choices=["stairs_only", "stairs", "flat", "climb",
                           "climb_only"])
  ap.add_argument("--tile", default="stairs_down",
                  help="stairs_down (empieza abajo) | stairs_up (arriba) | "
                       "box_ring (cajas, terreno climb) | ...")
  ap.add_argument("--level", type=int, default=5,
                  help="0 fácil ... 7 difícil (9 en climb: cajas de 40 cm)")
  ap.add_argument("--blind", action="store_true",
                  help="sin LiDAR (prueba el modo 'sin mapa')")
  ap.add_argument("--video", default=None, help="guardar mp4 sin visor")
  ap.add_argument("--seconds", type=float, default=25.0)
  args = ap.parse_args()

  sim = Sim(args.policy, args.terrain, args.tile, args.level,
            blind=args.blind)
  if args.video:
    run_video(sim, args.video, args.seconds)
  else:
    run_viewer(sim)


if __name__ == "__main__":
  main()
