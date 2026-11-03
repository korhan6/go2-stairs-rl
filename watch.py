#!/usr/bin/env python3
"""Visor en vivo del entrenamiento: carga cada checkpoint nuevo apenas se guarda.

Corre en la CPU (MuJoCo normal + red en numpy), al lado de train.py, sin
quitarle memoria de la GPU. No hay que escribir el nombre del run:

  python watch.py                      # el run más reciente de runs/
  python watch.py runs/climb_2026...   # un run en particular
  python watch.py --terrain climb_only --tile box_ring --level 9   # otro terreno

Por defecto usa el terreno de la etapa que se está entrenando y el perro
camina solo con un guion (adelante, girar, de lado, quieto). Si tocas una
flecha pasa a control manual (mismas teclas que play.py):
  ↑/↓ adelante/atrás · ←/→ girar · , . de lado · Backspace parar
  Enter levantar/soltar · Insert empujón fuerte · Ctrl + clic derecho
  arrastrando = empujarlo a mano
Cada 3-7 s recibe un EMPUJÓN al azar (0.6-1.4 m/s, flecha roja) para ver cómo se
estabiliza (--no_push para quitarlos). Si se cae, reaparece solo. Con política LiDAR se ven los puntos del escaneo
(verde = los ve, rojo = ocultos).
"""

import argparse
import glob
import json
import os
import sys
import time

# El visor NO toca la GPU (ni JAX ni Warp), para no quitarle VRAM al
# entrenamiento. El dibujo del visor usa OpenGL y no se ve afectado.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import play  # noqa: E402
from go2_stairs.runtime import PolicyRunner  # noqa: E402

# terreno y baldosa por defecto según la etapa del run
STAGE_VIEW = {
    "flat": ("flat", "rough", 2),
    "stairs": ("stairs_only", "stairs_down", 4),
    "climb": ("climb_only", "box_ring", 5),
}

# guion automático: (segundos, (vx, vy, wz))
SCRIPT = [
    (1.0, (0.0, 0.0, 0.0)),
    (5.0, (0.6, 0.0, 0.0)),
    (2.5, (0.0, 0.0, 1.0)),
    (5.0, (0.6, 0.0, 0.0)),
    (3.0, (0.0, 0.4, 0.0)),
    (2.5, (0.0, 0.0, -1.0)),
    (4.0, (-0.4, 0.0, 0.0)),
]

MANUAL_KEYS = {play.KEY_UP, play.KEY_DOWN, play.KEY_LEFT, play.KEY_RIGHT,
               play.KEY_COMMA, play.KEY_PERIOD, play.KEY_BACKSPACE}


def latest_run():
  runs = [d for d in glob.glob("runs/*") if os.path.isdir(d)]
  if not runs:
    return None
  return max(runs, key=os.path.getmtime)


def latest_ckpt(run):
  d = os.path.join(run, "checkpoints")
  if not os.path.isdir(d):
    return None
  steps = sorted(s for s in os.listdir(d) if s.isdigit())
  return os.path.join(d, steps[-1]) if steps else None


def export_ckpt(ck, out):
  from export_policy import export  # pylint: disable=import-outside-toplevel
  export(ck, out, verify=False)


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawTextHelpFormatter)
  ap.add_argument("run", nargs="?", default=None,
                  help="carpeta del run (por defecto el más reciente)")
  ap.add_argument("--terrain", default=None)
  ap.add_argument("--tile", default=None)
  ap.add_argument("--level", type=int, default=None)
  ap.add_argument("--manual", action="store_true",
                  help="sin guion: solo teclado")
  ap.add_argument("--no_push", action="store_true",
                  help="sin empujones automáticos")
  args = ap.parse_args()

  run = args.run or latest_run()
  while run is None:
    print("No hay runs todavía en runs/. Esperando a que arranque train.py...")
    time.sleep(5)
    run = latest_run()
  run = os.path.abspath(run)
  print("Run:", run)

  stage = "flat"
  try:
    with open(os.path.join(run, "env_config.json")) as f:
      stage = json.load(f).get("terrain", "flat")
  except (OSError, ValueError):
    pass
  terrain, tile, level = STAGE_VIEW.get(stage, STAGE_VIEW["flat"])
  terrain = args.terrain or terrain
  tile = args.tile or tile
  level = args.level if args.level is not None else level

  out = os.path.join(run, "watch_policy.npz")
  ck = latest_ckpt(run)
  while ck is None:
    print("Esperando el primer checkpoint (se guarda en cada evaluación)...")
    time.sleep(10)
    ck = latest_ckpt(run)
  time.sleep(3)  # que termine de escribirse
  print("Cargando", ck)
  export_ckpt(ck, out)
  loaded = ck

  sim = play.Sim(out, terrain, tile, level)
  print(f"Terreno {terrain} | baldosa {tile} nivel {level} | "
        f"{'LiDAR' if sim.runner.perceptive else 'ciega'}")
  state = {"manual": args.manual, "t": 0.0, "i": 0,
           "next_push": time.time() + np.random.uniform(3, 7)}
  dirs = ["adelante", "izquierda", "atrás", "derecha"]

  def do_push(dv):
    a = sim.push(dv)
    yaw = play.perception_yaw(sim)
    rel = (a - yaw + np.pi / 4) % (2 * np.pi)
    print(f"\n¡Empujón! {dv:.1f} m/s hacia {dirs[int(rel // (np.pi / 2))]}")

  def on_key(key):
    if key in MANUAL_KEYS and not state["manual"]:
      state["manual"] = True
      print("\nControl manual (teclado).")
    c = sim.command
    if key == play.KEY_UP:
      c[0] = min(c[0] + 0.1, 1.0)
    elif key == play.KEY_DOWN:
      c[0] = max(c[0] - 0.1, -1.0)
    elif key == play.KEY_LEFT:
      c[2] = min(c[2] + 0.2, 1.0)
    elif key == play.KEY_RIGHT:
      c[2] = max(c[2] - 0.2, -1.0)
    elif key == play.KEY_COMMA:
      c[1] = min(c[1] + 0.1, 0.6)
    elif key == play.KEY_PERIOD:
      c[1] = max(c[1] - 0.1, -0.6)
    elif key == play.KEY_BACKSPACE:
      c[:] = 0
    elif key == play.KEY_ENTER:
      sim.toggle_lift()
    elif key == play.KEY_INSERT:
      do_push(2.0)
    c[np.abs(c) < 0.05] = 0.0

  def respawn(msg):
    print("\n" + msg)
    sim.lift_target = None
    sim.spawn(tile, level, np.random.uniform(-np.pi, np.pi))
    state["t"], state["i"] = 0.0, 0

  last_check = last_print = time.time()
  with mujoco.viewer.launch_passive(sim.model, sim.data,
                                    key_callback=on_key) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    v.cam.trackbodyid = sim.base
    v.cam.distance = 2.0
    v.cam.elevation = -20
    while v.is_running():
      t0 = time.time()

      # ¿checkpoint nuevo?
      if t0 - last_check > 5.0:
        last_check = t0
        new = latest_ckpt(run)
        if new and new != loaded:
          time.sleep(3)
          try:
            export_ckpt(new, out)
            with v.lock():
              sim.runner = PolicyRunner(out)
              respawn(f"Checkpoint nuevo cargado: {os.path.basename(new)} "
                      f"pasos")
            loaded = new
          except Exception as e:  # pylint: disable=broad-except
            print("\nNo pude cargar", new, "(reintento):", e)

      # guion automático
      if not state["manual"] and sim.lift_target is None:
        dur, cmd = SCRIPT[state["i"]]
        sim.command[:] = cmd
        state["t"] += sim.ctrl_dt
        if state["t"] >= dur:
          state["t"], state["i"] = 0.0, (state["i"] + 1) % len(SCRIPT)

      # empujones al azar
      if (not args.no_push and t0 > state["next_push"]
          and sim.lift_target is None):
        do_push(np.random.uniform(0.6, 1.4))
        state["next_push"] = t0 + np.random.uniform(3, 7)

      with v.lock():
        sim.control_step()
        play.draw_scan(sim, v.user_scn)
        up = sim.data.xmat[sim.base].reshape(3, 3)[2, 2]
        if up < 0.3 and sim.lift_target is None:
          respawn("Se cayó: reaparece.")
      v.sync()

      if t0 - last_print > 0.5:
        print("\r" + sim.status() + " " * 10, end="", flush=True)
        last_print = t0
      time.sleep(max(0.0, sim.ctrl_dt - (time.time() - t0)))


if __name__ == "__main__":
  main()
