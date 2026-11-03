#!/usr/bin/env python3
"""EXPERIMENTAL: corre la política en el Go2 real (control de bajo nivel, DDS).

¡LEE LA SECCIÓN "ROBOT REAL" DEL README ANTES DE USARLO!
Primero pruébalo contra unitree_mujoco (simulador que habla DDS):

  python deploy_go2.py --policy runs/stairs_X/policy.npz --sim      # unitree_mujoco
  python deploy_go2.py --policy runs/stairs_X/policy.npz --iface eno1  # robot real

Requisitos: unitree_sdk2_python (pip install -e . desde su repo) y que el
robot acepte rt/lowcmd (EDU o funciones EDU desbloqueadas).

El contacto de cada pata se estima con el torque de los motores (tau_est, que
el Go2 calcula de la corriente): el Go2 Air no tiene sensores de presión.
Calibración: --show_contacts muestra la fuerza vertical estimada por pata.
De pie cada pata debe marcar ~30-40 N; levantado en el aire, ~0 N.

Teclas (en esta terminal):
  1  pararse (interpola a la pose por defecto)      2  arrancar la política
  w/x adelante/atrás   a/d girar   z/c de lado   s comando cero
  ESPACIO o 0  AMORTIGUAR (emergencia: el perro se echa suave)   q salir
Control remoto de Unitree (si está emparejado): START=pararse, A=política,
L2+B o SELECT=amortiguar; stick izq = vx/vy, stick der = giro.
"""

import argparse
import os
import select
import struct
import sys
import termios
import threading
import time
import tty

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from go2_stairs import constants as C  # noqa: E402
from go2_stairs.runtime import PolicyRunner, projected_gravity_from_quat  # noqa: E402

# Límites articulares (orden política) para el chequeo de seguridad.
Q_LOW = np.array([-0.95, -1.4, -2.62, -0.95, -1.4, -2.62,
                  -0.95, -0.42, -2.62, -0.95, -0.42, -2.62])
Q_HIGH = np.array([0.95, 2.5, -0.85, 0.95, 2.5, -0.85,
                   0.95, 2.5, -0.85, 0.95, 2.5, -0.85])
POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0

DAMP, STAND, POLICY = "AMORTIGUADO", "PARÁNDOSE", "POLÍTICA"


class Remote:
  """Decodifica LowState.wireless_remote (control remoto de Unitree)."""

  KEYS = ["R1", "L1", "start", "select", "R2", "L2", "F1", "F2",
          "A", "B", "X", "Y", "up", "right", "down", "left"]

  def __init__(self):
    self.btn = {k: 0 for k in self.KEYS}
    self.lx = self.ly = self.rx = self.ry = 0.0

  def parse(self, data):
    data = bytes(data)
    if len(data) < 24:
      return
    keys = struct.unpack("H", data[2:4])[0]
    for i, k in enumerate(self.KEYS):
      self.btn[k] = (keys >> i) & 1
    self.lx = struct.unpack("f", data[4:8])[0]
    self.rx = struct.unpack("f", data[8:12])[0]
    self.ry = struct.unpack("f", data[12:16])[0]
    self.ly = struct.unpack("f", data[20:24])[0]


class Go2Deploy:

  def __init__(self, args):
    from unitree_sdk2py.core.channel import (  # pylint: disable=import-outside-toplevel
        ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber)
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_  # pylint: disable=import-outside-toplevel
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_  # pylint: disable=import-outside-toplevel
    from unitree_sdk2py.utils.crc import CRC  # pylint: disable=import-outside-toplevel

    self.args = args
    self.runner = PolicyRunner(args.policy)
    if self.runner.perceptive:
      print("\n*** Esta política usa LiDAR (mapa de alturas). Todavía no hay "
            "un mapa conectado (ver get_height_scan), así que corre en modo "
            "'sin mapa': camina, pero NO anticipa escalones ni cajas. "
            "No la uses en escaleras reales hasta conectar el mapa. ***\n")
    self.Kp, self.Kd = self.runner.Kp, self.runner.Kd
    if args.contact_threshold is not None:
      self.runner.contact_threshold = args.contact_threshold
    print(f"Política: Kp={self.Kp} Kd={self.Kd} | contacto por "
          f"'{self.runner.contact_source}' (umbral "
          f"{self.runner.contact_threshold:.0f} N) | ctrl "
          f"{1 / self.runner.dt:.0f} Hz")

    domain = 1 if args.sim else 0
    iface = args.iface or ("lo" if args.sim else "eno1")
    ChannelFactoryInitialize(domain, iface)

    self.crc = CRC()
    self.cmd = unitree_go_msg_dds__LowCmd_()
    self.cmd.head[0], self.cmd.head[1] = 0xFE, 0xEF
    self.cmd.level_flag = 0xFF
    self.cmd.gpio = 0
    for i in range(20):
      m = self.cmd.motor_cmd[i]
      m.mode = 0x01
      m.q, m.dq, m.kp, m.kd, m.tau = POS_STOP_F, VEL_STOP_F, 0.0, 0.0, 0.0

    self.state = None
    self.state_time = 0.0
    self.remote = Remote()
    self.lock = threading.Lock()
    self.mode = DAMP
    self.command = np.zeros(3, dtype=np.float32)
    self.q_target = np.zeros(12)
    self.kp = 0.0
    self.kd = 2.0

    self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    self.pub.Init()
    self.sub = ChannelSubscriber("rt/lowstate", LowState_)
    self.sub.Init(self._on_state, 10)

    if not args.sim:
      self._release_sport_mode()

  # ---------------------------------------------------------------------------
  def _release_sport_mode(self):
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient  # pylint: disable=import-outside-toplevel
    from unitree_sdk2py.go2.sport.sport_client import SportClient  # pylint: disable=import-outside-toplevel

    sc = SportClient()
    sc.SetTimeout(5.0)
    sc.Init()
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()
    _, result = msc.CheckMode()
    while result and result.get("name"):
      print(f"Liberando el modo '{result['name']}' (el perro se va a echar)...")
      sc.StandDown()
      time.sleep(2.0)
      msc.ReleaseMode()
      time.sleep(1.0)
      _, result = msc.CheckMode()
    print("Modo deportivo liberado: control de bajo nivel activo.")

  def _on_state(self, msg):
    with self.lock:
      self.state = msg
      self.state_time = time.time()
      try:
        self.remote.parse(msg.wireless_remote)
      except Exception:  # pylint: disable=broad-except
        pass

  # ---------------------------------------------------------------------------
  def read(self):
    """Lee el LowState y lo pasa a orden de la política."""
    with self.lock:
      s = self.state
    if s is None:
      return None
    q = np.array([s.motor_state[C.POLICY_TO_SDK[i]].q for i in range(12)])
    dq = np.array([s.motor_state[C.POLICY_TO_SDK[i]].dq for i in range(12)])
    # torque estimado por el motor a partir de su corriente
    tau = np.array([s.motor_state[C.POLICY_TO_SDK[i]].tau_est
                    for i in range(12)])
    quat = np.array(s.imu_state.quaternion, dtype=np.float64)  # w x y z
    gyro = np.array(s.imu_state.gyroscope, dtype=np.float32)
    gravity = projected_gravity_from_quat(quat)
    ff = np.array(getattr(s, "foot_force", [0, 0, 0, 0]),
                  dtype=np.float32)[C.FOOT_FORCE_SDK_TO_POLICY]
    if self.runner.contact_source == "sensor":
      contacts = ff > self.args.foot_force_threshold
      fz = ff
    else:  # "torque" (Go2 Air) o "none"
      contacts = self.runner.contacts_from_torque(q, tau, gravity)
      fz = self.runner.last_foot_fz
    return dict(q=q, dq=dq, tau=tau, gyro=gyro, gravity=gravity,
                contacts=contacts, foot_fz=fz, foot_force=ff)

  def publish(self):
    for i in range(12):
      m = self.cmd.motor_cmd[C.POLICY_TO_SDK[i]]
      m.q = float(self.q_target[i])
      m.dq = 0.0
      m.kp = float(self.kp)
      m.kd = float(self.kd)
      m.tau = 0.0
    self.cmd.crc = self.crc.Crc(self.cmd)
    self.pub.Write(self.cmd)

  def set_damp(self, reason=""):
    if self.mode != DAMP:
      print(f"\n>>> AMORTIGUAR {reason}")
    self.mode = DAMP
    self.kp, self.kd = 0.0, 2.0
    st = self.read()
    if st is not None:
      self.q_target = st["q"]

  # ---------------------------------------------------------------------------
  def handle_input(self, ch):
    c = self.command
    r = self.remote.btn
    if ch in (" ", "0") or r["select"] or (r["L2"] and r["B"]):
      self.set_damp("(usuario)")
      return
    if ch == "1" or r["start"]:
      if self.mode == DAMP:
        st = self.read()
        if st is None:
          return
        self.stand_from = st["q"].copy()
        self.stand_t = 0.0
        self.mode = STAND
        print("\n>>> PARÁNDOSE")
    if ch == "2" or r["A"]:
      if self.mode == STAND and self.stand_t >= 2.0:
        self.runner.reset()
        self.command[:] = 0
        self.mode = POLICY
        print("\n>>> POLÍTICA ACTIVA")
      elif self.mode != POLICY and ch == "2":
        print("\n(primero '1' y espera 2 s a que termine de pararse)")
    step = {"w": (0, 0.1), "x": (0, -0.1), "a": (2, 0.2), "d": (2, -0.2),
            "z": (1, 0.1), "c": (1, -0.1)}
    if ch in step:
      i, v = step[ch]
      c[i] = np.clip(c[i] + v, [-0.8, -0.5, -1.0][i], [1.0, 0.5, 1.0][i])
    if ch == "s":
      c[:] = 0
    # sticks del control remoto (si se mueven, mandan)
    rm = self.remote
    if max(abs(rm.lx), abs(rm.ly), abs(rm.rx)) > 0.1:
      c[:] = [rm.ly * 0.8, -rm.lx * 0.4, -rm.rx * 0.8]
    c[np.abs(c) < 0.05] = 0.0

  def get_height_scan(self):
    """Escaneo de alturas para la política con LiDAR. -> (h_rel, valido).

    PENDIENTE de conectar al mapa de elevación del L1. Con un mapa en el
    marco de odometría (p. ej. elevation_mapping o un grid hecho con la
    nube de GLIM) y la pose del cuerpo en ese marco:

        def altura(xy):            # xy (N, 2) -> (N,), NaN si no hay dato
            return mapa.consultar(xy)
        return self.runner.scan_from_height_fn(altura, pos_cuerpo, R_cuerpo,
                                               occlusion=False)

    (occlusion=False: el mapa real ya trae sus huecos como NaN.)
    Mientras devuelva (None, None) la política corre "sin mapa".
    """
    return None, None

  def safety(self, st):
    if time.time() - self.state_time > 0.1:
      self.set_damp("(sin LowState > 100 ms)")
      return False
    tilt = np.arccos(np.clip(-st["gravity"][2], -1, 1))
    if tilt > self.args.max_tilt:
      self.set_damp(f"(inclinación {np.degrees(tilt):.0f}°)")
      return False
    if np.any(st["q"] < Q_LOW - 0.15) or np.any(st["q"] > Q_HIGH + 0.15):
      self.set_damp("(articulación fuera de rango)")
      return False
    return True

  # ---------------------------------------------------------------------------
  def run(self):
    dt = self.runner.dt
    print("Esperando LowState...")
    while self.read() is None:
      time.sleep(0.05)
    self.set_damp()
    print(__doc__)

    stop = threading.Event()

    def sender():  # publica a 200 Hz el último comando
      while not stop.is_set():
        self.publish()
        time.sleep(0.005)

    th = threading.Thread(target=sender, daemon=True)
    th.start()

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    last_print = 0.0
    warned_ff = False
    try:
      while True:
        t0 = time.time()
        ch = ""
        if select.select([sys.stdin], [], [], 0)[0]:
          ch = sys.stdin.read(1)
        if ch == "q":
          break
        self.handle_input(ch)
        st = self.read()

        if self.args.show_contacts and t0 - last_print > 0.3:
          print(f"\r[{self.mode}] fuerza vertical FL FR RL RR (N):",
                np.round(st["foot_fz"]).astype(int),
                "contacto:", st["contacts"].astype(int), " " * 6, end="")
          last_print = t0

        if self.mode == STAND:
          self.stand_t += dt
          a = min(self.stand_t / 2.0, 1.0)
          self.q_target = (1 - a) * self.stand_from + a * C.DEFAULT_POSE
          self.kp, self.kd = self.args.stand_kp, self.args.stand_kd
          if self.stand_t > 2.5 and not warned_ff:
            warned_ff = True
            n_on = int(np.sum(st["contacts"]))
            if self.runner.use_contacts and n_on < 4:
              print(f"\n¡OJO! De pie solo detecto {n_on}/4 patas apoyadas "
                    f"(fuerza estimada {np.round(st['foot_fz']).astype(int)} N)."
                    " Revisa con --show_contacts y ajusta --contact_threshold "
                    "antes de activar la política.")
        elif self.mode == POLICY and self.safety(st):
          hs, hv = self.get_height_scan()
          targets, _ = self.runner.act(st["gyro"], st["gravity"],
                                       self.command, st["q"], st["dq"],
                                       tau=st["tau"], contacts=st["contacts"],
                                       height_scan=hs, scan_valid=hv)
          self.q_target = np.clip(targets, Q_LOW, Q_HIGH)
          self.kp, self.kd = self.Kp, self.Kd
          if not self.args.show_contacts and t0 - last_print > 0.5:
            print(f"\r[{self.mode}] cmd vx={self.command[0]:+.1f} "
                  f"vy={self.command[1]:+.1f} wz={self.command[2]:+.1f} "
                  f"contactos {st['contacts'].astype(int)}   ", end="")
            last_print = t0
        time.sleep(max(0.0, dt - (time.time() - t0)))
    finally:
      self.set_damp("(saliendo)")
      time.sleep(1.0)
      stop.set()
      th.join(timeout=1.0)
      termios.tcsetattr(fd, termios.TCSADRAIN, old)
      print("\nListo. Para volver al modo de fábrica reinicia el perro "
            "(o usa MotionSwitcherClient.SelectMode('normal')).")


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawTextHelpFormatter)
  ap.add_argument("--policy", required=True)
  ap.add_argument("--iface", default=None, help="interfaz de red (ej. eno1)")
  ap.add_argument("--sim", action="store_true",
                  help="probar contra unitree_mujoco (dominio 1, interfaz lo)")
  ap.add_argument("--contact_threshold", type=float, default=None,
                  help="N de fuerza vertical estimada para 'pata apoyada' "
                       "(por defecto el del entrenamiento, 15 N)")
  ap.add_argument("--show_contacts", action="store_true",
                  help="imprime la fuerza estimada por pata para calibrar")
  ap.add_argument("--foot_force_threshold", type=float, default=20.0,
                  help="solo para políticas con contact_source=sensor")
  ap.add_argument("--max_tilt", type=float, default=1.2,
                  help="inclinación máxima (rad) antes de amortiguar")
  ap.add_argument("--stand_kp", type=float, default=40.0,
                  help="Kp solo para pararse (la política usa el suyo)")
  ap.add_argument("--stand_kd", type=float, default=1.0)
  args = ap.parse_args()
  Go2Deploy(args).run()


if __name__ == "__main__":
  main()
