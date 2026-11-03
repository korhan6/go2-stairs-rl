"""Entorno MJX: Go2 caminando en escaleras con marcha suave tipo Spot.

Ideas clave
-----------
* PERCEPCIÓN (LiDAR L1 simulado, ver perception.py): la política recibe una
  grilla de 17 x 11 alturas del terreno alrededor del robot, con oclusión
  desde la cabeza, ruido, huecos y deriva de odometría. Con eso levanta cada
  pata lo justo ANTES de llegar al escalón o la caja (recompensa de altura de
  paso que mira hacia adelante) y puede subirse a cosas altas. Algunos
  episodios son "ciegos" (sin mapa) para que siga caminando si el mapa falla.
  `perception.enable=False` vuelve a la política ciega original.
* Currículo: cada robot empieza en terreno fácil y sube de nivel si cruza su
  baldosa, baja si no avanza (como legged_gym).
* Propiocepción: IMU (giroscopio + gravedad proyectada),
  posiciones/velocidades articulares, última acción, comando de velocidad,
  reloj de marcha (seno/coseno) y CONTACTO DE LAS PATAS estimado con el
  TORQUE DE LOS MOTORES (corriente), porque el Go2 Air no tiene sensores de
  presión en las patas (ver contact.py). Con un historial de varios pasos la
  red "siente" el piso: detecta cuándo una pata golpea un escalón o pisa en
  falso y corrige, igual que el controlador de fábrica.
* Actor-crítico asimétrico: el crítico ve información privilegiada que solo
  existe en simulación (escaneo de alturas del terreno, velocidad real,
  fricción, fuerzas externas). Eso acelera muchísimo el aprendizaje, pero la
  política final NO la necesita -> se puede usar en el robot real.
* Marcha trote con reloj (2 Hz): FL+RR / FR+RL. Recompensas de suavidad
  (tasa de acción, aceleración articular, energía, deslizamiento) -> se ve
  fluido como el Spot. Con comando cero se queda quieto de pie.
* "Levantar al perro": durante el entrenamiento a veces se le aplica una
  fuerza que lo levanta (como si lo cargaras) y luego lo suelta. Mientras está
  en el aire se le premia por seguir dando pasitos (las "pataditas" del
  controlador de fábrica) y al soltarlo debe aterrizar y seguir.
* Empujones aleatorios, ruido de sensores, latencia y aleatorización de
  dinámica (fricción, masas, ganancias PD) para que transfiera al real.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env

from go2_stairs import constants as C
from go2_stairs import contact as contact_lib
from go2_stairs import perception
from go2_stairs import robot_xml
from go2_stairs import terrain as terrain_lib


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      terrain="stairs",            # "flat" (etapa 1) o "stairs" (etapa 2)
      terrain_seed=0,
      ctrl_dt=0.02,                # 50 Hz de política
      sim_dt=0.004,                # 250 Hz de física
      episode_length=1000,         # 20 s
      Kp=25.0,
      Kd=0.6,
      solver_iterations=2,
      ls_iterations=6,
      history_len=5,               # cuadros de propiocepción apilados
      # Contacto que ve la política:
      #   "torque" -> estimado con el torque de los motores (Go2 Air, real)
      #   "sensor" -> sensor de contacto ideal (robots con foot_force)
      #   "none"   -> sin contacto
      contact_source="torque",
      contact_threshold=15.0,      # N de fuerza vertical estimada
      tau_scale_range=[0.85, 1.15],  # error de calibración de tau_est
      tau_noise=0.5,               # N·m de ruido en el torque medido
      soft_joint_pos_limit_factor=0.95,
      spawn_noise=0.4,
      spawn_height=0.38,
      gait=config_dict.create(
          freq=2.0,                # Hz
          duty=0.5,                # fracción de apoyo
          swing_height=0.10,       # m sobre el terreno local
          air_swing_height=0.08,   # pasitos en el aire
          # Altura de paso que MIRA ADELANTE: para cada pata en vuelo se mira
          # el terreno a estas distancias en la dirección del comando; si hay
          # algo más alto (escalón, caja) se exige subir la pata por encima.
          lookahead=[0.05, 0.10, 0.15, 0.20, 0.25],
          max_step_up=0.45,        # m, lo más alto que se le pide subir
      ),
      perception=config_dict.create(
          enable=True,
          update_every=5,          # el mapa "se refresca" a 10 Hz
          ray_samples=10,          # muestras por rayo para la oclusión
          noise_std=0.02,          # m, ruido por punto (precisión del L1)
          outlier_prob=0.01,       # puntos falsos
          outlier_mag=0.30,        # m
          dropout=0.05,            # puntos sin dato
          rear_dropout=0.30,       # extra atrás (patas/cuerpo tapan)
          xy_bias=0.04,            # m, desalineación mapa-robot por episodio
          z_bias=0.04,             # m, deriva vertical de la odometría
          yaw_bias=0.05,           # rad
          blind_prob=0.10,         # episodios sin mapa (todo inválido)
      ),
      curriculum=config_dict.create(
          enable=True,
          init_max_level=2,        # niveles iniciales 0..2
          up_dist=2.0,             # m recorridos para subir de nivel
          down_ratio=0.35,         # baja si recorre < 35% de lo pedido
      ),
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              contact_flip_prob=0.02,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Seguimiento del comando.
              tracking_lin_vel=1.5,
              tracking_ang_vel=0.8,
              # Base estable (Spot).
              lin_vel_z=-0.5,
              ang_vel_xy=-0.05,
              orientation=-2.0,
              base_height=-30.0,
              # Marcha.
              gait_contact=0.4,
              swing_clearance=-40.0,
              feet_slip=-0.2,
              stand_still=-0.5,
              hip_pos=-0.3,
              # Suavidad / eficiencia.
              action_rate=-0.02,
              action_smoothness=-0.01,
              dof_acc=-2.5e-7,
              torques=-2e-4,
              energy=-2e-3,
              dof_pos_limits=-1.0,
              # Seguridad.
              collision=-1.0,
              stumble=-1.0,          # pata choca de frente con el escalón
              termination=-1.0,
              # Cargado en el aire.
              air_stepping=0.6,
          ),
          tracking_sigma=0.25,
          base_height_target=0.30,
          stumble_height=0.04,     # pata en contacto pero a >4 cm del piso
      ),
      pert_config=config_dict.create(
          enable=True,
          # cambio de velocidad máx ~ velocity_kick / pi (3.5 -> ~1.1 m/s)
          velocity_kick=[0.5, 3.5],
          kick_durations=[0.05, 0.2],
          kick_wait_times=[1.5, 5.0],
      ),
      lift_config=config_dict.create(
          enable=True,
          wait_time=[6.0, 22.0],    # s entre levantadas
          duration=[1.0, 3.0],      # s en el aire
          height=[0.25, 0.50],      # m sobre la altura actual
      ),
      command_config=config_dict.create(
          a=[1.0, 0.6, 1.0],        # |vx|, |vy|, |wz| máximos
          b=[0.9, 0.25, 0.5],       # prob. de NO poner a cero cada eje
          stand_prob=0.1,           # prob. de comando cero (quedarse quieto)
      ),
      action_delay_prob=0.25,       # prob. de un paso de latencia por episodio
      impl="warp",
      naconmax=32 * 4096,
      njmax=160,
  )


def _quiet_warp(m):
  """Apaga los avisos "solver iterations limit reached" de MuJoCo Warp.

  Con solver_iterations=2 (a propósito, como Playground) el solver siempre
  usa todas sus iteraciones y Warp lo imprime en CADA paso de cada robot.
  Es la opción `opt.warn_overflow = False` de MJWarp; aquí se busca donde
  la guarde MJX según la versión. Si no la encuentra, no cambia nada.
  """
  opt = getattr(m, "opt", None)
  for owner_name in ("_impl", None):
    try:
      owner = getattr(opt, owner_name) if owner_name else opt
      if owner is not None and hasattr(owner, "warn_overflow"):
        new_owner = owner.replace(warn_overflow=False)
        new_opt = opt.replace(**{owner_name: new_owner}) if owner_name \
            else new_owner
        print("MuJoCo Warp: avisos de iteraciones silenciados")
        return m.replace(opt=new_opt)
    except Exception:  # pylint: disable=broad-except
      pass
  impl = getattr(m, "_impl", None)
  try:
    wopt = getattr(impl, "opt", None)
    if wopt is not None and hasattr(wopt, "warn_overflow"):
      print("MuJoCo Warp: avisos de iteraciones silenciados")
      return m.replace(_impl=impl.replace(opt=wopt.replace(warn_overflow=False)))
  except Exception:  # pylint: disable=broad-except
    pass
  return m


class Go2Stairs(mjx_env.MjxEnv):
  """Go2: seguir comando de velocidad en escaleras con marcha suave."""

  def __init__(
      self,
      config: Optional[config_dict.ConfigDict] = None,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    config = config if config is not None else default_config()
    super().__init__(config, config_overrides)
    cfg = self._config

    self._terrain = terrain_lib.generate(
        terrain_lib.preset(cfg.terrain, cfg.terrain_seed))
    self._mj_model = robot_xml.build_model(
        self._terrain, cfg.Kp, cfg.Kd, cfg.sim_dt, cfg.solver_iterations,
        cfg.ls_iterations)
    self._mj_model.vis.global_.offwidth = 1920
    self._mj_model.vis.global_.offheight = 1080
    self._mjx_model = _quiet_warp(mjx.put_model(self._mj_model, impl=cfg.impl))
    self._xml_path = "go2_stairs(generated)"
    self._post_init()

  # ---------------------------------------------------------------------------
  # Inicialización
  # ---------------------------------------------------------------------------

  def _post_init(self) -> None:
    m = self._mj_model
    cfg = self._config
    self._init_q = jp.array(m.keyframe("home").qpos)
    self._default_pose = jp.array(C.DEFAULT_POSE)
    self._action_scale = jp.array(C.ACTION_SCALE)
    self._lowers, self._uppers = m.jnt_range[1:].T
    f = cfg.soft_joint_pos_limit_factor
    mid = (self._lowers + self._uppers) / 2
    half = (self._uppers - self._lowers) / 2
    self._soft_lowers = jp.array(mid - f * half)
    self._soft_uppers = jp.array(mid + f * half)

    self._base_id = m.body("base").id
    self._mass = float(m.body_subtreemass[self._base_id])
    self._imu_site_id = m.site("imu").id
    self._feet_site_id = np.array([m.site(s).id for s in C.FOOT_SITES])
    self._floor_geom_id = m.geom("floor").id

    def adr(name):
      i = m.sensor(name).id
      return int(m.sensor_adr[i]), int(m.sensor_dim[i])

    def vec_adr(name):
      a, d = adr(name)
      return np.arange(a, a + d)

    self._s_gyro = vec_adr("gyro")
    self._s_linvel = vec_adr("local_linvel")
    self._s_acc = vec_adr("accelerometer")
    self._s_up = vec_adr("upvector")
    self._s_glinvel = vec_adr("global_linvel")
    self._s_gangvel = vec_adr("global_angvel")
    self._s_feet_vel = np.stack([vec_adr(f"{l}_global_linvel") for l in C.LEGS])
    self._s_foot_contact = np.array(
        [adr(f"{l}_floor_found")[0] for l in C.LEGS])
    self._s_bad_contact = np.array(
        [adr(f"{l}_{p}_floor_found")[0] for l in C.LEGS
         for p in ("thigh", "calf")])
    self._s_base_contact = np.array(
        [adr(f"{g}_floor_found")[0] for g in
         ("base_col", "head_col1", "head_col2")])

    # Terreno como arreglo JAX para consultas de altura exactas.
    ter = self._terrain
    self._hmap = jp.asarray(ter.heights)
    self._hmap_x0 = float(ter.x0)
    self._hmap_y0 = float(ter.y0)
    self._hmap_res = float(ter.resolution)
    self._half_x = float(ter.half_x)
    self._half_y = float(ter.half_y)
    self._spawn = jp.asarray(ter.spawn)
    self._spawn_level = jp.asarray(ter.spawn_level)
    self._num_levels = ter.config.num_levels

    # Grilla del escaneo de alturas (actor con LiDAR y crítico sin ruido).
    self._scan_pts = jp.array(perception.grid_points())
    self._n_scan = perception.NUM_POINTS
    self._ncols = len(ter.config.columns)
    self._scan_rear = jp.array(perception.grid_points()[:, 0] < -0.25)
    self._lookahead = jp.array(cfg.gait.lookahead)

    self._offsets = jp.array(C.TROT_OFFSETS)
    self._cmd_a = jp.array(cfg.command_config.a)
    self._cmd_b = jp.array(cfg.command_config.b)
    self._frame_size = 3 + 3 + 3 + 12 + 12 + 12 + 2 + 4

  # ---------------------------------------------------------------------------
  # Utilidades de terreno
  # ---------------------------------------------------------------------------

  def height_at(self, xy: jax.Array) -> jax.Array:
    """Altura del terreno (bilineal). xy: (..., 2)."""
    h = self._hmap
    ny, nx = h.shape
    fx = (xy[..., 0] - self._hmap_x0) / self._hmap_res
    fy = (xy[..., 1] - self._hmap_y0) / self._hmap_res
    c0 = jp.clip(jp.floor(fx).astype(jp.int32), 0, nx - 2)
    r0 = jp.clip(jp.floor(fy).astype(jp.int32), 0, ny - 2)
    tx = jp.clip(fx - c0, 0.0, 1.0)
    ty = jp.clip(fy - r0, 0.0, 1.0)
    return ((1 - tx) * (1 - ty) * h[r0, c0] + tx * (1 - ty) * h[r0, c0 + 1]
            + (1 - tx) * ty * h[r0 + 1, c0] + tx * ty * h[r0 + 1, c0 + 1])

  def _height_scan(self, data: mjx.Data) -> jax.Array:
    """Escaneo ideal (sin ruido ni oclusión), para el crítico."""
    return perception.clean_scan(jp, self.height_at, data.xpos[self._base_id],
                                 data.xmat[self._base_id], self._scan_pts)

  # ---------------------------------------------------------------------------
  # Sensores
  # ---------------------------------------------------------------------------

  def get_gravity(self, data: mjx.Data) -> jax.Array:
    return data.site_xmat[self._imu_site_id].T @ jp.array([0.0, 0.0, -1.0])

  def _sens(self, data, idx):
    return data.sensordata[idx]

  def _foot_contact(self, data) -> jax.Array:
    return data.sensordata[self._s_foot_contact] > 0

  # ---------------------------------------------------------------------------
  # Reset
  # ---------------------------------------------------------------------------

  def _sample_timers(self, rng):
    pc, lc = self._config.pert_config, self._config.lift_config
    k = jax.random.split(rng, 6)
    dt = self.dt
    steps = lambda key, lo, hi: jp.round(
        jax.random.uniform(key, minval=lo, maxval=hi) / dt).astype(jp.int32)
    return dict(
        steps_until_next_pert=steps(k[0], *pc.kick_wait_times),
        pert_duration_seconds=jax.random.uniform(
            k[1], minval=pc.kick_durations[0], maxval=pc.kick_durations[1]),
        pert_mag=jax.random.uniform(
            k[2], minval=pc.velocity_kick[0], maxval=pc.velocity_kick[1]),
        lift_countdown=steps(k[3], *lc.wait_time),
        lift_duration=steps(k[4], *lc.duration),
        lift_height=jax.random.uniform(
            k[5], minval=lc.height[0], maxval=lc.height[1]),
    )

  def _episode_info(self, rng) -> Dict[str, Any]:
    """Info que se re-inicia en cada episodio."""
    k = jax.random.split(rng, 7)
    lo, hi = self._config.tau_scale_range
    pc = self._config.perception
    info = dict(
        tau_scale=jax.random.uniform(k[4], (12,), minval=lo, maxval=hi),
        command=self.sample_command(k[0], jp.zeros(3)),
        steps_until_next_cmd=jp.round(
            jax.random.exponential(k[1]) * 5.0 / self.dt).astype(jp.int32),
        last_act=jp.zeros(12),
        last_last_act=jp.zeros(12),
        phase=jax.random.uniform(k[2]),
        delay=jax.random.bernoulli(k[3], self._config.action_delay_prob),
        steps_since_last_pert=jp.array(0, jp.int32),
        pert_steps=jp.array(0, jp.int32),
        pert_dir=jp.zeros(3),
        lift_steps=jp.array(0, jp.int32),      # >0 mientras está levantado
        lift_elapsed=jp.array(0, jp.int32),
        lift_target=jp.zeros(3),
        last_contact=jp.ones(4, dtype=bool),
        # LiDAR: deriva (dx, dy, dz, dyaw) y "sin mapa" por episodio
        scan_bias=(2 * jax.random.uniform(k[5], (4,)) - 1) * jp.array(
            [pc.xy_bias, pc.xy_bias, pc.z_bias, pc.yaw_bias]),
        scan_blind=jax.random.bernoulli(k[6], pc.blind_prob),
        scan_timer=jp.array(0, jp.int32),
        scan_noise=jp.zeros(self._n_scan),
        scan_drop=jp.zeros(self._n_scan, dtype=bool),
        # currículo
        ep_dist=jp.zeros(()),
        cmd_dist=jp.zeros(()),
    )
    info.update(self._sample_timers(jax.random.fold_in(rng, 7)))
    return info

  def reset(self, rng: jax.Array) -> mjx_env.State:
    cfg = self._config
    rng, k_tile, k_col, k_xy, k_yaw, k_q, k_v, k_info = jax.random.split(rng, 8)

    if cfg.curriculum.enable:
      top = min(cfg.curriculum.init_max_level, self._num_levels - 1)
      lvl = jax.random.randint(k_tile, (), 0, top + 1)
      col = jax.random.randint(k_col, (), 0, self._ncols)
      tile = lvl * self._ncols + col
    else:
      tile = jax.random.randint(k_tile, (), 0, self._spawn.shape[0])
    center = self._spawn[tile]
    xy = center[:2] + jax.random.uniform(
        k_xy, (2,), minval=-cfg.spawn_noise, maxval=cfg.spawn_noise)
    ground = self.height_at(xy)

    qpos = self._init_q
    qpos = qpos.at[0:2].set(xy)
    qpos = qpos.at[2].set(ground + cfg.spawn_height)
    yaw = jax.random.uniform(k_yaw, (1,), minval=-jp.pi, maxval=jp.pi)
    quat = math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)
    qpos = qpos.at[3:7].set(math.quat_mul(qpos[3:7], quat))
    qpos = qpos.at[7:].add(jax.random.uniform(k_q, (12,), minval=-0.05,
                                              maxval=0.05))
    qvel = jp.zeros(self.mjx_model.nv)
    qvel = qvel.at[0:6].set(jax.random.uniform(k_v, (6,), minval=-0.3,
                                               maxval=0.3))

    data = mjx_env.make_data(
        self.mj_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:],
        impl=self.mjx_model.impl.value, naconmax=cfg.naconmax,
        njmax=cfg.njmax)
    data = mjx.forward(self.mjx_model, data)

    info = self._episode_info(k_info)
    info["rng"] = rng
    info["tile_level"] = self._spawn_level[tile]
    info["level"] = self._spawn_level[tile]
    info["ep_start_xy"] = xy
    info["history"] = jp.zeros((cfg.history_len, self._frame_size))

    contact = self._foot_contact(data)
    obs, info = self._get_obs(data, info, contact, reset_history=True)

    metrics = {f"reward/{k}": jp.zeros(()) for k in
               cfg.reward_config.scales.keys()}
    metrics.update({
        "lifted": jp.zeros(()),
        "terrain_level": jp.zeros(()),
        "base_height": jp.zeros(()),
        "swing_clearance_m": jp.zeros(()),
    })
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  # ---------------------------------------------------------------------------
  # Step
  # ---------------------------------------------------------------------------

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    cfg = self._config
    info = dict(state.info)

    # Nuevo episodio (el wrapper de auto-reset no reinicia `info`).
    data0 = state.data
    if "steps" in info:
      new_ep = info["steps"] == 0
      if cfg.curriculum.enable:
        data0, info = self._curriculum(data0, info, new_ep)
      info["rng"], k_ep = jax.random.split(info["rng"])
      fresh = self._episode_info(k_ep)
      for key, val in fresh.items():
        info[key] = jp.where(new_ep, val, info[key])
    else:
      new_ep = jp.array(False)

    # Latencia: a veces se aplica la acción anterior (como en el robot real).
    applied = jp.where(info["delay"], info["last_act"], action)
    motor_targets = self._default_pose + applied * self._action_scale

    # Fuerzas externas: empujones + "levantar al perro".
    xfrc, info = self._external_forces(data0, info)
    data = data0.replace(xfrc_applied=xfrc)
    data = mjx_env.step(self.mjx_model, data, motor_targets, self.n_substeps)

    # Reloj de marcha.
    info["phase"] = jp.mod(info["phase"] + self.dt * cfg.gait.freq, 1.0)

    contact = self._foot_contact(data)
    lifted = info["lift_steps"] > 0

    prev1, prev2 = info["last_act"], info["last_last_act"]  # a(t-1), a(t-2)
    info["last_last_act"] = prev1
    info["last_act"] = action
    obs, info = self._get_obs(data, info, contact, reset_history=new_ep)

    done = self._get_termination(data) & ~lifted
    rewards, extra = self._get_reward(data, action, prev1, prev2, info,
                                      contact, done, lifted)
    rewards = {k: v * cfg.reward_config.scales[k] for k, v in rewards.items()}
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    # Comandos nuevos cada cierto tiempo.
    info["steps_until_next_cmd"] -= 1
    info["rng"], k1, k2 = jax.random.split(info["rng"], 3)
    resample = info["steps_until_next_cmd"] <= 0
    info["command"] = jp.where(resample, self.sample_command(k1, info["command"]),
                               info["command"])
    info["steps_until_next_cmd"] = jp.where(
        done | resample,
        jp.round(jax.random.exponential(k2) * 5.0 / self.dt).astype(jp.int32),
        info["steps_until_next_cmd"])
    info["last_contact"] = contact
    info["ep_dist"] = jp.linalg.norm(
        data.xpos[self._base_id][:2] - info["ep_start_xy"])
    info["cmd_dist"] = info["cmd_dist"] + jp.linalg.norm(
        info["command"][:2]) * self.dt

    metrics = dict(state.metrics)
    for k, v in rewards.items():
      metrics[f"reward/{k}"] = v
    metrics["lifted"] = lifted.astype(jp.float32)
    metrics["terrain_level"] = info["tile_level"].astype(jp.float32)
    metrics["base_height"] = extra["base_height"]
    metrics["swing_clearance_m"] = extra["swing_clearance"]

    done = done.astype(reward.dtype)
    return state.replace(data=data, obs=obs, reward=reward, done=done,
                         metrics=metrics, info=info)

  # ---------------------------------------------------------------------------
  # Currículo de terreno
  # ---------------------------------------------------------------------------

  def _curriculum(self, data: mjx.Data, info: Dict[str, Any], new_ep):
    """Al empezar un episodio: sube/baja de nivel y teletransporta al robot.

    El wrapper de auto-reset de Brax siempre vuelve al MISMO estado inicial,
    así que aquí se mueve el robot a una baldosa nueva del nivel que le toca.
    """
    cfg = self._config
    cc = cfg.curriculum
    n_lv = self._num_levels
    info["rng"], k_col, k_xy, k_yaw, k_rand = jax.random.split(info["rng"], 5)
    moved, asked = info["ep_dist"], info["cmd_dist"]
    up = moved > cc.up_dist
    down = (moved < cc.down_ratio * asked) & ~up
    lvl = info["level"] + up.astype(jp.int32) - down.astype(jp.int32)
    # pasó el último nivel -> nivel al azar (para no olvidar los fáciles)
    lvl = jp.where(lvl >= n_lv, jax.random.randint(k_rand, (), 0, n_lv), lvl)
    lvl = jp.clip(lvl, 0, n_lv - 1)
    lvl = jp.where(new_ep, lvl, info["level"])
    col = jax.random.randint(k_col, (), 0, self._ncols)
    tile = lvl * self._ncols + col
    xy = self._spawn[tile][:2] + jax.random.uniform(
        k_xy, (2,), minval=-cfg.spawn_noise, maxval=cfg.spawn_noise)
    yaw = jax.random.uniform(k_yaw, minval=-jp.pi, maxval=jp.pi)
    q = data.qpos
    q_new = q.at[0:2].set(xy)
    q_new = q_new.at[2].set(self.height_at(xy) + cfg.spawn_height)
    q_new = q_new.at[3:7].set(
        jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)]))
    data = data.replace(qpos=jp.where(new_ep, q_new, q))
    info["level"] = lvl
    info["tile_level"] = jp.where(new_ep, lvl, info["tile_level"])
    info["ep_start_xy"] = jp.where(new_ep, xy, info["ep_start_xy"])
    return data, info

  # ---------------------------------------------------------------------------
  # Fuerzas externas
  # ---------------------------------------------------------------------------

  def _external_forces(self, data: mjx.Data, info: Dict[str, Any]):
    cfg = self._config
    force = jp.zeros(3)
    torque = jp.zeros(3)

    # --- Empujones (patada de velocidad, igual que Playground Go1) ---
    if cfg.pert_config.enable:
      info["rng"], k_dir = jax.random.split(info["rng"])
      angle = jax.random.uniform(k_dir, minval=0.0, maxval=2 * jp.pi)
      new_dir = jp.array([jp.cos(angle), jp.sin(angle), 0.0])
      active = info["steps_since_last_pert"] >= info["steps_until_next_pert"]
      t = info["pert_steps"] * self.dt
      u_t = 0.5 * jp.sin(jp.pi * t / info["pert_duration_seconds"])
      push = (u_t * self._mass * info["pert_mag"]
              / info["pert_duration_seconds"]) * info["pert_dir"]
      force = force + jp.where(active, push, 0.0)
      finished = active & (info["pert_steps"] >= jp.round(
          info["pert_duration_seconds"] / self.dt).astype(jp.int32))
      info["pert_steps"] = jp.where(active, info["pert_steps"] + 1, 0)
      info["steps_since_last_pert"] = jp.where(
          finished, 0, info["steps_since_last_pert"] + 1)
      info["pert_dir"] = jp.where(active, info["pert_dir"], new_dir)

    # --- Levantar al perro y soltarlo ---
    if cfg.lift_config.enable:
      pos = data.xpos[self._base_id]
      vel = data.sensordata[self._s_glinvel]
      angvel = data.sensordata[self._s_gangvel]
      up = data.xmat[self._base_id][:, 2]
      starting = (info["lift_steps"] == 0) & (info["lift_countdown"] <= 0)
      info["lift_steps"] = jp.where(starting, info["lift_duration"],
                                    info["lift_steps"])
      info["lift_elapsed"] = jp.where(starting, 0, info["lift_elapsed"])
      info["lift_target"] = jp.where(
          starting, pos + jp.array([0.0, 0.0, 1.0]) * info["lift_height"],
          info["lift_target"])
      lifting = info["lift_steps"] > 0
      ramp = jp.clip(info["lift_elapsed"] * self.dt / 0.3, 0.0, 1.0)
      m = self._mass
      err = info["lift_target"] - pos
      f_lift = jp.array([
          m * (10.0 * err[0] - 5.0 * vel[0]),
          m * (10.0 * err[1] - 5.0 * vel[1]),
          m * (9.81 * ramp + 25.0 * err[2] * ramp - 8.0 * vel[2]),
      ])
      tilt = jp.cross(up, jp.array([0.0, 0.0, 1.0]))  # eje para enderezar
      t_lift = 30.0 * tilt - 2.0 * angvel
      force = force + jp.where(lifting, f_lift, 0.0)
      torque = torque + jp.where(lifting, t_lift, 0.0)

      info["lift_elapsed"] = jp.where(lifting, info["lift_elapsed"] + 1, 0)
      ending = lifting & (info["lift_steps"] == 1)
      info["lift_steps"] = jp.maximum(info["lift_steps"] - 1, 0)
      info["lift_countdown"] = jp.where(lifting, info["lift_countdown"],
                                        info["lift_countdown"] - 1)
      info["rng"], k_t = jax.random.split(info["rng"])
      timers = self._sample_timers(k_t)
      for key in ("lift_countdown", "lift_duration", "lift_height"):
        info[key] = jp.where(ending, timers[key], info[key])

    xfrc = jp.zeros((self.mjx_model.nbody, 6))
    xfrc = xfrc.at[self._base_id, :3].set(force)
    xfrc = xfrc.at[self._base_id, 3:].set(torque)
    return xfrc, info

  # ---------------------------------------------------------------------------
  # Observaciones
  # ---------------------------------------------------------------------------

  def _gait_obs(self, phase):
    return jp.array([jp.sin(2 * jp.pi * phase), jp.cos(2 * jp.pi * phase)])

  def _get_obs(self, data, info, contact, reset_history):
    cfg = self._config
    sc = cfg.noise_config.scales
    lvl = cfg.noise_config.level
    info["rng"], k1, k2, k3, k4, k5, k6 = jax.random.split(info["rng"], 7)
    u = lambda k, shape: 2 * jax.random.uniform(k, shape) - 1

    gyro = data.sensordata[self._s_gyro]
    gravity = self.get_gravity(data)
    qpos = data.qpos[7:]
    qvel = data.qvel[6:]
    gait = self._gait_obs(info["phase"])
    contact_f = contact.astype(jp.float32)

    if cfg.contact_source == "torque":
      # Igual que en el robot real: tau_est (con error de escala y ruido)
      tau_meas = (data.actuator_force * info["tau_scale"]
                  + u(k6, (12,)) * lvl * cfg.tau_noise)
      est, _ = contact_lib.estimate_contacts(
          jp, qpos, tau_meas, gravity, cfg.contact_threshold)
      actor_contact = est.astype(jp.float32)
    elif cfg.contact_source == "sensor":
      actor_contact = contact_f
    else:
      actor_contact = jp.zeros(4)
    flip = jax.random.bernoulli(k5, sc.contact_flip_prob * lvl, (4,))
    noisy_contact = jp.where(flip, 1.0 - actor_contact, actor_contact)
    if cfg.contact_source == "none":
      noisy_contact = jp.zeros(4)

    frame = jp.hstack([
        gyro + u(k1, (3,)) * lvl * sc.gyro,
        gravity + u(k2, (3,)) * lvl * sc.gravity,
        info["command"],
        qpos - self._default_pose + u(k3, (12,)) * lvl * sc.joint_pos,
        qvel + u(k4, (12,)) * lvl * sc.joint_vel,
        info["last_act"],
        gait,
        noisy_contact,
    ])
    rolled = jp.concatenate([info["history"][1:], frame[None]], axis=0)
    filled = jp.tile(frame[None], (cfg.history_len, 1))
    info["history"] = jp.where(reset_history, filled, rolled)
    actor_obs = info["history"].ravel()

    base = data.xpos[self._base_id]
    rot = data.xmat[self._base_id]
    if cfg.perception.enable:
      actor_obs = jp.concatenate(
          [actor_obs, self._lidar_obs(base, rot, info, reset_history)])

    clean_frame = jp.hstack([
        gyro, gravity, info["command"], qpos - self._default_pose, qvel,
        info["last_act"], gait, contact_f,  # contacto real para el crítico
    ])
    feet_pos = data.site_xpos[self._feet_site_id]
    clearance = (feet_pos[:, 2] - self.height_at(feet_pos[:, :2])
                 - C.FOOT_RADIUS)
    privileged = jp.hstack([
        clean_frame,
        data.sensordata[self._s_linvel],
        data.sensordata[self._s_acc],
        data.sensordata[self._s_gangvel],
        perception.clean_scan(jp, self.height_at, base, rot, self._scan_pts),
        jp.clip(clearance, -0.2, 0.5),
        data.sensordata[self._s_feet_vel].ravel(),
        data.actuator_force / 20.0,
        (info["lift_steps"] > 0).astype(jp.float32)[None],
        data.xfrc_applied[self._base_id, :3] / (self._mass * 9.81),
        self.mjx_model.geom_friction[self._floor_geom_id, 0:1],
        jp.atleast_1d(self.mjx_model.body_mass[self._base_id]) - 6.921,
    ])
    return {"state": actor_obs, "privileged_state": privileged}, info

  def _lidar_obs(self, base, rot, info, reset_history):
    """Escaneo de alturas como lo daría el L1 + mapa de elevación real."""
    pc = self._config.perception
    lvl = self._config.noise_config.level
    n = self._n_scan
    info["rng"], kn, ko, km, kd = jax.random.split(info["rng"], 5)
    # ruido/huecos nuevos cada vez que "se refresca el mapa" (10 Hz)
    refresh = (info["scan_timer"] == 0) | reset_history
    noise = jax.random.normal(kn, (n,)) * pc.noise_std * lvl
    outlier = jax.random.bernoulli(ko, pc.outlier_prob * lvl, (n,))
    noise = noise + outlier * (2 * jax.random.uniform(km, (n,)) - 1) \
        * pc.outlier_mag
    p_drop = (pc.dropout + pc.rear_dropout * self._scan_rear) * lvl
    drop = jax.random.uniform(kd, (n,)) < p_drop
    info["scan_noise"] = jp.where(refresh, noise, info["scan_noise"])
    info["scan_drop"] = jp.where(refresh, drop, info["scan_drop"])
    info["scan_timer"] = jp.mod(info["scan_timer"] + 1, pc.update_every)

    # deriva de odometría: el mapa está algo corrido respecto al robot
    b = info["scan_bias"] * lvl
    yaw = perception.yaw_from_mat(jp, rot) + b[3]
    pts_w = perception.grid_world(jp, self._scan_pts, base[:2] + b[:2], yaw)
    h = self.height_at(pts_w)
    vis = perception.visibility(jp, self.height_at, base, rot, pts_w, h,
                                pc.ray_samples)
    h_rel = perception.heights_to_obs(jp, base[2] + b[2],
                                      h + info["scan_noise"])
    valid = vis & ~info["scan_drop"] & ~info["scan_blind"]
    return perception.to_policy_input(jp, h_rel, valid)

  # ---------------------------------------------------------------------------
  # Terminación
  # ---------------------------------------------------------------------------

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    up = data.sensordata[self._s_up]
    base = data.xpos[self._base_id]
    feet = data.site_xpos[self._feet_site_id]
    ground = jp.mean(self.height_at(feet[:, :2]))
    fallen = up[2] < 0.35
    base_hit = jp.any(data.sensordata[self._s_base_contact] > 0)
    too_low = (base[2] - ground) < 0.12
    out = ((jp.abs(base[0]) > self._half_x - 0.5)
           | (jp.abs(base[1]) > self._half_y - 0.5))
    nan = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    return fallen | base_hit | too_low | out | nan

  # ---------------------------------------------------------------------------
  # Recompensas
  # ---------------------------------------------------------------------------

  def _get_reward(self, data, action, prev1, prev2, info, contact, done,
                  lifted):
    cfg = self._config
    rc = cfg.reward_config
    cmd = info["command"]
    cmd_norm = jp.linalg.norm(cmd)
    moving = cmd_norm > 0.1
    ground_mask = (~lifted).astype(jp.float32)

    local_vel = data.sensordata[self._s_linvel]
    gyro = data.sensordata[self._s_gyro]
    gvel = data.sensordata[self._s_glinvel]
    gangvel = data.sensordata[self._s_gangvel]
    up = data.sensordata[self._s_up]
    gravity = self.get_gravity(data)
    qpos = data.qpos[7:]
    qvel = data.qvel[6:]
    qacc = data.qacc[6:]
    torque = data.actuator_force

    base = data.xpos[self._base_id]
    feet_pos = data.site_xpos[self._feet_site_id]
    feet_ground = self.height_at(feet_pos[:, :2])
    clearance = feet_pos[:, 2] - feet_ground - C.FOOT_RADIUS
    base_height = base[2] - jp.mean(feet_ground)

    # Reloj de marcha por pata.
    duty = cfg.gait.duty
    ph = jp.mod(info["phase"] + self._offsets, 1.0)
    stance_des = ph < duty
    swing_s = jp.clip((ph - duty) / (1.0 - duty), 0.0, 1.0)
    bump = jp.sin(jp.pi * swing_s) * (~stance_des)

    # --- Seguimiento ---
    sig = rc.tracking_sigma
    lin_err = jp.sum(jp.square(cmd[:2] - local_vel[:2]))
    ang_err = jp.square(cmd[2] - gyro[2])
    r_lin = jp.exp(-lin_err / sig)
    r_ang = jp.exp(-ang_err / sig)

    # --- Marcha ---
    desired_contact = jp.where(moving, stance_des, True)
    r_gait = jp.mean((contact == desired_contact).astype(jp.float32))
    # Altura de paso que mira adelante: si en la dirección del comando hay
    # algo más alto (escalón, caja) la pata en vuelo debe pasar por encima.
    rot = data.xmat[self._base_id]
    yaw = jp.arctan2(rot[1, 0], rot[0, 0])
    cy, sy = jp.cos(yaw), jp.sin(yaw)
    cmd_w = jp.array([cy * cmd[0] - sy * cmd[1], sy * cmd[0] + cy * cmd[1]])
    spd = jp.linalg.norm(cmd_w)
    dir_w = cmd_w / jp.maximum(spd, 1e-3)
    ahead = (feet_pos[:, None, :2]
             + self._lookahead[None, :, None] * dir_w[None, None, :])
    h_ahead = jp.max(self.height_at(ahead), axis=1)
    step_up = jp.clip(h_ahead - feet_ground, 0.0, cfg.gait.max_step_up)
    step_up = step_up * (spd > 0.1)
    h_ref = (cfg.gait.swing_height + step_up) * bump * moving
    deficit = jp.maximum(h_ref - clearance, 0.0)
    c_clear = jp.sum(jp.square(deficit))
    feet_vel = data.sensordata[self._s_feet_vel]
    c_slip = jp.sum(jp.sum(jp.square(feet_vel[:, :2]), -1) * contact)
    c_stand = jp.sum(jp.abs(qpos - self._default_pose)) * (~moving)
    hip_idx = np.array([0, 3, 6, 9])
    c_hip = jp.sum(jp.square(qpos[hip_idx] - self._default_pose[hip_idx]))

    # --- Pasitos en el aire (cuando lo levantan) ---
    base_z = base[2]
    rel_z = feet_pos[:, 2] - base_z
    z_ref = -0.30 + cfg.gait.air_swing_height * bump
    airborne = lifted & (info["lift_elapsed"] * self.dt > 0.3)
    r_air = jp.exp(-jp.sum(jp.square(rel_z - z_ref)) / 0.005) * airborne

    # --- Colisiones (muslo/canilla contra el escalón) ---
    bad = jp.sum(data.sensordata[self._s_bad_contact] > 0).astype(jp.float32)
    # pata tocando algo pero lejos del piso que tiene debajo = chocó de
    # frente con la contrahuella / el costado de la caja
    stumble = jp.sum(contact & (clearance > rc.stumble_height)).astype(
        jp.float32)

    # --- Orientación: se permite cabecear lo que pide el terreno ---
    # (subiendo una caja las patas de adelante están más arriba)
    dh = jp.mean(feet_ground[:2]) - jp.mean(feet_ground[2:])
    exp_gx = jp.clip(-dh / jp.sqrt(dh * dh + 0.387 ** 2), -0.5, 0.5)
    c_orient = jp.square(gravity[0] - exp_gx) + jp.square(gravity[1])

    # --- Límites articulares ---
    out_lim = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_lim += jp.clip(qpos - self._soft_uppers, 0.0, None)

    rewards = {
        "tracking_lin_vel": r_lin * ground_mask,
        "tracking_ang_vel": r_ang * ground_mask,
        "lin_vel_z": jp.square(gvel[2]) * ground_mask,
        "ang_vel_xy": jp.sum(jp.square(gangvel[:2])) * ground_mask,
        "orientation": c_orient * ground_mask,
        "base_height": jp.square(base_height - rc.base_height_target)
                       * ground_mask,
        "gait_contact": r_gait * ground_mask,
        "swing_clearance": c_clear * ground_mask,
        "feet_slip": c_slip * moving * ground_mask,
        "stand_still": c_stand * ground_mask,
        "hip_pos": c_hip,
        "action_rate": jp.sum(jp.square(action - prev1)),
        "action_smoothness": jp.sum(jp.square(action - 2 * prev1 + prev2)),
        "dof_acc": jp.sum(jp.square(qacc)),
        "torques": jp.sqrt(jp.sum(jp.square(torque))) + jp.sum(jp.abs(torque)),
        "energy": jp.sum(jp.abs(qvel) * jp.abs(torque)),
        "dof_pos_limits": jp.sum(out_lim),
        "collision": bad,
        "stumble": stumble * ground_mask,
        "termination": done.astype(jp.float32),
        "air_stepping": r_air,
    }
    del up
    extra = {
        "base_height": base_height,
        "swing_clearance": jp.max(clearance * (~stance_des)),
    }
    return rewards, extra

  # ---------------------------------------------------------------------------
  # Comandos
  # ---------------------------------------------------------------------------

  def sample_command(self, rng: jax.Array, x_k: jax.Array) -> jax.Array:
    rng, y_rng, w_rng, z_rng, s_rng = jax.random.split(rng, 5)
    y_k = jax.random.uniform(y_rng, shape=(3,), minval=-self._cmd_a,
                             maxval=self._cmd_a)
    z_k = jax.random.bernoulli(z_rng, self._cmd_b, shape=(3,))
    w_k = jax.random.bernoulli(w_rng, 0.5, shape=(3,))
    x_kp1 = x_k - w_k * (x_k - y_k * z_k)
    stand = jax.random.bernoulli(s_rng, self._config.command_config.stand_prob)
    x_kp1 = jp.where(stand, jp.zeros(3), x_kp1)
    # Comandos muy pequeños -> cero (evita "arrastrar" las patas).
    return jp.where(jp.abs(x_kp1) < 0.08, 0.0, x_kp1)

  # ---------------------------------------------------------------------------
  # Accesores
  # ---------------------------------------------------------------------------

  @property
  def terrain(self) -> terrain_lib.Terrain:
    return self._terrain

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
