"""Construye el MJCF de la escena: Go2 + terreno heightfield.

El robot se basa en `unitree_go2/go2_mjx.xml` de MuJoCo Menagerie (masas,
inercias y mallas reales), pero con colisiones rehechas para escaleras:
  - pata  : esfera r=2.2 cm (con fricción, condim 3)
  - canilla (calf) y muslo (thigh): cápsulas -> se pueden "enganchar" con el
    borde del escalón como en el robot real (se penaliza en la recompensa)
  - tronco: caja + 2 esferas en la cabeza
Solo hay colisión robot<->piso (sin auto-colisiones), como en Playground.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import mujoco
import numpy as np

from go2_stairs import constants as C
from go2_stairs import terrain as terrain_lib


def menagerie_go2_dir() -> str:
  env_dir = os.environ.get("GO2_MENAGERIE_DIR")
  if env_dir:
    return env_dir
  # Copias locales del modelo (evita bajar todo Menagerie)
  for cand in ("~/unitree-mujoco-web/public/assets/models/go2",
               "~/mujoco_menagerie/unitree_go2"):
    d = os.path.expanduser(cand)
    if os.path.isfile(os.path.join(d, "assets", "base_0.obj")):
      return d
  from mujoco_playground._src import mjx_env  # descarga Menagerie si falta

  mjx_env.ensure_menagerie_exists()
  return str(mjx_env.MENAGERIE_PATH / "unitree_go2")


def load_assets() -> Dict[str, bytes]:
  assets = {}
  adir = os.path.join(menagerie_go2_dir(), "assets")
  for f in os.listdir(adir):
    p = os.path.join(adir, f)
    if os.path.isfile(p):
      with open(p, "rb") as fh:
        assets[f] = fh.read()
  return assets


def _leg(name: str, side: float, front: bool) -> str:
  """XML de una pata. side=+1 izquierda, -1 derecha."""
  hip_x = 0.1934 if front else -0.1934
  hip_y = 0.0465 * side
  mirror = "" if side > 0 else "_mirror"
  thigh_cls = "front_hip" if front else "back_hip"
  # inercias de Menagerie (las de la derecha están espejadas)
  if side > 0:
    hip_in = ('pos="-0.0054 0.00194 -0.000105" quat="0.497014 0.499245 '
              '0.505462 0.498237"') if front else (
                  'pos="0.0054 0.00194 -0.000105" quat="0.505462 0.498237 '
                  '0.497014 0.499245"')
    thigh_in = ('pos="-0.00374 -0.0223 -0.0327" quat="0.829533 0.0847635 '
                '-0.0200632 0.551623"')
    calf_in = ('pos="0.00629595 -0.000622121 -0.141417" quat="0.710672 '
               '0.00154099 -0.00450087 0.703508"')
  else:
    hip_in = ('pos="-0.0054 -0.00194 -0.000105" quat="0.498237 0.505462 '
              '0.499245 0.497014"') if front else (
                  'pos="0.0054 -0.00194 -0.000105" quat="0.499245 0.497014 '
                  '0.498237 0.505462"')
    thigh_in = ('pos="-0.00374 0.0223 -0.0327" quat="0.551623 -0.0200632 '
                '0.0847635 0.829533"')
    calf_in = ('pos="0.00629595 0.000622121 -0.141417" quat="0.703508 '
               '-0.00450087 0.00154099 0.710672"')
  hip_quat = {
      (1, True): "1 0 0 0",
      (-1, True): "4.63268e-05 1 0 0",
      (1, False): "4.63268e-05 0 1 0",
      (-1, False): "2.14617e-09 4.63268e-05 4.63268e-05 -1",
  }[(int(side), front)]
  return f"""
      <body name="{name}_hip" pos="{hip_x} {hip_y} 0">
        <inertial {hip_in} mass="0.678" diaginertia="0.00088403 0.000596003 0.000479967"/>
        <joint name="{name}_hip_joint" class="abduction"/>
        <geom mesh="hip_0" material="metal" class="visual" quat="{hip_quat}"/>
        <geom mesh="hip_1" material="gray" class="visual" quat="{hip_quat}"/>
        <body name="{name}_thigh" pos="0 {0.0955 * side} 0">
          <inertial {thigh_in} mass="1.152" diaginertia="0.00594973 0.00584149 0.000878787"/>
          <joint name="{name}_thigh_joint" class="{thigh_cls}"/>
          <geom mesh="thigh{mirror}_0" material="metal" class="visual"/>
          <geom mesh="thigh{mirror}_1" material="gray" class="visual"/>
          <geom name="{name}_thigh_col" class="collision" type="capsule" size="0.02"
            fromto="0 0 -0.04 -0.01 0 -0.17"/>
          <body name="{name}_calf" pos="0 0 -0.213">
            <inertial {calf_in} mass="0.241352" diaginertia="0.0014901 0.00146356 5.31397e-05"/>
            <joint name="{name}_calf_joint" class="knee"/>
            <geom mesh="calf{mirror}_0" material="gray" class="visual"/>
            <geom mesh="calf{mirror}_1" material="black" class="visual"/>
            <geom pos="0 0 -0.213" mesh="foot" class="visual" material="black"/>
            <geom name="{name}_calf_col" class="collision" type="capsule" size="0.013"
              fromto="0.012 0 -0.01 0.006 0 -0.175"/>
            <geom name="{name}" class="foot"/>
            <site name="{name}_foot" pos="-0.002 0 -0.213" size="0.01" group="5"/>
          </body>
        </body>
      </body>"""


def build_scene_xml(terrain: terrain_lib.Terrain, sim_dt: float = 0.004,
                    solver_iterations: int = 2,
                    ls_iterations: int = 6) -> Tuple[str, dict]:
  hf = terrain_lib.hfield_arrays(terrain)
  sx, sy, sz, base = hf["size"]
  pose = " ".join(f"{v:g}" for v in C.DEFAULT_POSE)
  legs = "".join([
      _leg("FL", +1, True),
      _leg("FR", -1, True),
      _leg("RL", +1, False),
      _leg("RR", -1, False),
  ])
  sensors = []
  for leg in C.LEGS:
    sensors.append(
        f'<framepos objtype="site" objname="{leg}_foot" name="{leg}_pos"/>')
    sensors.append(
        f'<framelinvel objtype="site" objname="{leg}_foot" '
        f'name="{leg}_global_linvel"/>')
  for leg in C.LEGS:
    sensors.append(
        f'<contact name="{leg}_floor_found" geom1="{leg}" geom2="floor" '
        f'num="1" data="found" reduce="mindist"/>')
  for leg in C.LEGS:
    for part in ("thigh", "calf"):
      sensors.append(
          f'<contact name="{leg}_{part}_floor_found" geom1="{leg}_{part}_col" '
          f'geom2="floor" num="1" data="found" reduce="mindist"/>')
  for g in ("base_col", "head_col1", "head_col2"):
    sensors.append(
        f'<contact name="{g}_floor_found" geom1="{g}" geom2="floor" num="1" '
        'data="found" reduce="mindist"/>')
  sensors_xml = "\n    ".join(sensors)

  xml = f"""
<mujoco model="go2_stairs">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{sim_dt}" iterations="{solver_iterations}" ls_iterations="{ls_iterations}"
    integrator="Euler" cone="pyramidal">
    <flag eulerdamp="disable"/>
  </option>
  <size memory="64M"/>

  <visual>
    <headlight diffuse=".7 .7 .7" ambient=".3 .3 .3" specular="0 0 0"/>
    <global azimuth="140" elevation="-25" offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
    <map znear="0.01"/>
  </visual>

  <default>
    <default class="go2">
      <geom contype="0" conaffinity="0" condim="1" friction="0.6"/>
      <joint axis="0 1 0" damping="0.05" armature="0.01" frictionloss="0.1"/>
      <general biastype="affine" gainprm="25 0 0" biasprm="0 -25 -0.6"
        forcerange="-23.7 23.7"/>
      <default class="abduction">
        <joint axis="1 0 0" range="-1.0472 1.0472"/>
        <general ctrlrange="-0.9472 0.9472"/>
      </default>
      <default class="front_hip">
        <joint range="-1.5708 3.4907"/>
        <general ctrlrange="-1.4 2.5"/>
      </default>
      <default class="back_hip">
        <joint range="-0.5236 4.5379"/>
        <general ctrlrange="-0.4236 2.5"/>
      </default>
      <default class="knee">
        <joint range="-2.7227 -0.83776"/>
        <general ctrlrange="-2.6227 -0.84776" forcerange="-45.43 45.43"/>
      </default>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom group="3" contype="0" conaffinity="1" condim="1" rgba="0.8 0.3 0.3 0.5"/>
        <default class="foot">
          <geom type="sphere" size="{C.FOOT_RADIUS}" pos="-0.002 0 -0.213" condim="3"
            friction="0.8 0.02 0.01" priority="1" rgba="0.2 0.2 0.2 1"/>
        </default>
      </default>
    </default>
  </default>

  <asset>
    <material name="metal" rgba=".9 .95 .95 1"/>
    <material name="black" rgba="0.1 0.1 0.1 1"/>
    <material name="white" rgba="1 1 1 1"/>
    <material name="gray" rgba="0.671705 0.692426 0.774270 1"/>
    <mesh file="base_0.obj"/>
    <mesh file="base_1.obj"/>
    <mesh file="base_2.obj"/>
    <mesh file="base_3.obj"/>
    <mesh file="base_4.obj"/>
    <mesh file="hip_0.obj"/>
    <mesh file="hip_1.obj"/>
    <mesh file="thigh_0.obj"/>
    <mesh file="thigh_1.obj"/>
    <mesh file="thigh_mirror_0.obj"/>
    <mesh file="thigh_mirror_1.obj"/>
    <mesh file="calf_0.obj"/>
    <mesh file="calf_1.obj"/>
    <mesh file="calf_mirror_0.obj"/>
    <mesh file="calf_mirror_1.obj"/>
    <mesh file="foot.obj"/>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.7 0.9" rgb2="0.1 0.12 0.18" width="512" height="512"/>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.82 0.82 0.80"
      rgb2="0.72 0.72 0.70" markrgb="0.45 0.45 0.45" width="512" height="512"/>
    <material name="terrain" texture="grid" texuniform="true" texrepeat="2 2" reflectance="0"/>
    <hfield name="terrain" nrow="{hf['nrow']}" ncol="{hf['ncol']}" size="{sx} {sy} {sz} {base}"/>
  </asset>

  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="hfield" hfield="terrain" pos="0 0 {hf['z_offset']}"
      material="terrain" contype="1" conaffinity="0" condim="3" priority="2" friction="1.0"/>
    <body name="base" pos="0 0 0.45" childclass="go2">
      <camera name="track" pos="0.9 -1.6 0.7" xyaxes="0.866 0.500 0.000 -0.2 0.35 0.92" mode="trackcom"/>
      <camera name="side" pos="0 -1.6 0.3" xyaxes="1 0 0 0 0.2 1" mode="trackcom"/>
      <inertial pos="0.021112 0 -0.005366" quat="-0.000543471 0.713435 -0.00173769 0.700719"
        mass="6.921" diaginertia="0.107027 0.0980771 0.0244531"/>
      <freejoint name="root"/>
      <geom mesh="base_0" material="black" class="visual"/>
      <geom mesh="base_1" material="black" class="visual"/>
      <geom mesh="base_2" material="black" class="visual"/>
      <geom mesh="base_3" material="white" class="visual"/>
      <geom mesh="base_4" material="gray" class="visual"/>
      <geom name="base_col" class="collision" type="box" size="0.1881 0.04675 0.057"/>
      <geom name="head_col1" class="collision" type="sphere" size="0.05" pos="0.285 0 0.01"/>
      <geom name="head_col2" class="collision" type="sphere" size="0.047" pos="0.293 0 -0.06"/>
      <site name="imu" pos="-0.02557 0 0.04232" size="0.01" group="5"/>
      {legs}
    </body>
  </worldbody>

  <actuator>
    <general class="abduction" name="FL_hip" joint="FL_hip_joint"/>
    <general class="front_hip" name="FL_thigh" joint="FL_thigh_joint"/>
    <general class="knee" name="FL_calf" joint="FL_calf_joint"/>
    <general class="abduction" name="FR_hip" joint="FR_hip_joint"/>
    <general class="front_hip" name="FR_thigh" joint="FR_thigh_joint"/>
    <general class="knee" name="FR_calf" joint="FR_calf_joint"/>
    <general class="abduction" name="RL_hip" joint="RL_hip_joint"/>
    <general class="back_hip" name="RL_thigh" joint="RL_thigh_joint"/>
    <general class="knee" name="RL_calf" joint="RL_calf_joint"/>
    <general class="abduction" name="RR_hip" joint="RR_hip_joint"/>
    <general class="back_hip" name="RR_thigh" joint="RR_thigh_joint"/>
    <general class="knee" name="RR_calf" joint="RR_calf_joint"/>
  </actuator>

  <sensor>
    <gyro site="imu" name="gyro"/>
    <velocimeter site="imu" name="local_linvel"/>
    <accelerometer site="imu" name="accelerometer"/>
    <framezaxis objtype="site" objname="imu" name="upvector"/>
    <framequat objtype="site" objname="imu" name="orientation"/>
    <framelinvel objtype="site" objname="imu" name="global_linvel"/>
    <frameangvel objtype="site" objname="imu" name="global_angvel"/>
    {sensors_xml}
  </sensor>

  <keyframe>
    <key name="home" qpos="0 0 0.36 1 0 0 0 {pose}" ctrl="{pose}"/>
  </keyframe>
</mujoco>
"""
  return xml, hf


def build_model(terrain: terrain_lib.Terrain, Kp: float, Kd: float,
                sim_dt: float = 0.004, solver_iterations: int = 2,
                ls_iterations: int = 6) -> mujoco.MjModel:
  """Compila el MjModel, carga las alturas y fija las ganancias PD."""
  xml, hf = build_scene_xml(terrain, sim_dt, solver_iterations, ls_iterations)
  model = mujoco.MjModel.from_xml_string(xml, assets=load_assets())
  model.hfield_data[:] = hf["data"].ravel()
  model.actuator_gainprm[:, 0] = Kp
  model.actuator_biasprm[:, 1] = -Kp
  model.actuator_biasprm[:, 2] = -Kd
  # Verifica el orden de las articulaciones (freejoint primero).
  names = [model.joint(i).name for i in range(1, model.njnt)]
  assert tuple(names) == C.JOINT_NAMES, names
  return model
