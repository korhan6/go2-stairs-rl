#!/usr/bin/env python3
"""Exporta un checkpoint de Brax PPO a `policy.npz` (numpy puro).

  python export_policy.py runs/stairs_XXXX            # usa el último checkpoint
  python export_policy.py runs/stairs_XXXX/checkpoints/000400000000 -o mi.npz

Verifica que la red en numpy da EXACTAMENTE lo mismo que la de JAX.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from go2_stairs import constants as C  # noqa: E402
from go2_stairs import perception  # noqa: E402


def _find_env_config(ckpt_path):
  d = os.path.abspath(ckpt_path)
  for _ in range(3):
    p = os.path.join(d, "env_config.json")
    if os.path.exists(p):
      return p
    d = os.path.dirname(d)
  return None


def export(ckpt_path: str, out_path: str, env_cfg=None, verify: bool = True):
  from go2_stairs import compat  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
  from brax.training.agents.ppo import checkpoint  # pylint: disable=import-outside-toplevel

  ckpt_path = os.path.abspath(ckpt_path)  # Orbax exige ruta absoluta
  params = checkpoint.load(ckpt_path)
  # (no usamos checkpoint.load_config: en Brax 0.14.2 falla con init=None)
  with open(os.path.join(ckpt_path, "ppo_network_config.json")) as f:
    net_cfg = json.load(f)
  kw = net_cfg["network_factory_kwargs"]
  if kw.get("activation", "silu") not in ("silu", "swish"):
    raise ValueError(f"Activación no soportada: {kw['activation']}")
  key = kw.get("policy_obs_key", "state")

  if env_cfg is None:
    p = _find_env_config(ckpt_path)
    if p is None:
      raise FileNotFoundError("No encontré env_config.json del run")
    with open(p) as f:
      env_cfg = json.load(f)
  elif not isinstance(env_cfg, dict):
    env_cfg = json.loads(env_cfg.to_json_best_effort())

  norm = params[0]
  mean = np.asarray(norm.mean[key], np.float32)
  std = np.asarray(norm.std[key], np.float32)
  pol = params[1]["params"]
  names = sorted(pol.keys(), key=lambda s: int(s.split("_")[-1]))
  arrays = {}
  for i, n in enumerate(names):
    arrays[f"W{i}"] = np.asarray(pol[n]["kernel"], np.float32)
    arrays[f"b{i}"] = np.asarray(pol[n]["bias"], np.float32)

  frame_size = 51
  meta = dict(
      num_layers=len(names),
      activation=kw.get("activation", "swish"),
      history_len=int(env_cfg["history_len"]),
      frame_size=frame_size,
      ctrl_dt=float(env_cfg["ctrl_dt"]),
      sim_dt=float(env_cfg["sim_dt"]),
      Kp=float(env_cfg["Kp"]),
      Kd=float(env_cfg["Kd"]),
      gait_freq=float(env_cfg["gait"]["freq"]),
      contact_source=str(env_cfg.get("contact_source", "torque")),
      contact_threshold=float(env_cfg.get("contact_threshold", 15.0)),
      actor_contacts=env_cfg.get("contact_source", "torque") != "none",
      default_pose=C.DEFAULT_POSE.tolist(),
      action_scale=C.ACTION_SCALE.tolist(),
      joint_order=list(C.JOINT_NAMES),
      obs_layout=["gyro(3)", "projected_gravity(3)", "command vx,vy,wz(3)",
                  "qpos-default(12)", "qvel(12)", "last_action(12)",
                  "gait sin,cos(2)",
                  "foot_contact FL,FR,RL,RR(4) (estimado por torque si "
                  "contact_source=torque)"],
      checkpoint=os.path.abspath(ckpt_path),
  )
  pcfg = env_cfg.get("perception", {}) or {}
  n_obs = meta["history_len"] * frame_size
  if pcfg.get("enable", False):
    meta["perception"] = dict(
        enable=True,
        grid_x=perception.GRID_X.tolist(),
        grid_y=perception.GRID_Y.tolist(),
        lidar_pos=list(perception.LIDAR_POS),
        lidar_axis=list(perception.LIDAR_AXIS),
        nominal_height=perception.NOMINAL_HEIGHT,
        clip=perception.CLIP,
        layout="h_rel(187, 0 si no hay dato) + valido(187); fila = y",
    )
    n_obs += 2 * perception.NUM_POINTS
  else:
    meta["perception"] = dict(enable=False)
  assert mean.shape[0] == n_obs, (mean.shape, n_obs)
  os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
  np.savez(out_path, obs_mean=mean, obs_std=std, meta=json.dumps(meta),
           **arrays)

  if verify:
    import jax  # pylint: disable=import-outside-toplevel
    import jax.numpy as jp  # pylint: disable=import-outside-toplevel
    from go2_stairs.runtime import PolicyRunner  # pylint: disable=import-outside-toplevel

    from brax.training.acme import running_statistics  # pylint: disable=import-outside-toplevel
    from brax.training.agents.ppo import networks as ppo_networks  # pylint: disable=import-outside-toplevel

    priv = int(np.asarray(norm.mean["privileged_state"]).shape[0])
    net = ppo_networks.make_ppo_networks(
        {"state": (mean.shape[0],), "privileged_state": (priv,)}, 12,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=tuple(kw["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes=tuple(kw["value_hidden_layer_sizes"]),
        policy_obs_key=kw.get("policy_obs_key", "state"),
        value_obs_key=kw.get("value_obs_key", "privileged_state"))
    policy = ppo_networks.make_inference_fn(net)(params, deterministic=True)
    runner = PolicyRunner(out_path)
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(5):
      x = (mean + std * rng.normal(size=mean.shape)).astype(np.float32)
      obs = {"state": jp.asarray(x[None]),
             "privileged_state": jp.zeros((1, priv))}
      a_jax = np.asarray(policy(obs, jax.random.PRNGKey(0))[0])[0]
      a_np = runner.forward(x)
      max_err = max(max_err, float(np.abs(a_jax - a_np).max()))
    print(f"Verificación numpy vs JAX: error máx = {max_err:.2e}")
    if max_err > 1e-4:
      raise RuntimeError("La red exportada no coincide con la de JAX")
  return out_path


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("path", help="carpeta del run o de un checkpoint")
  ap.add_argument("-o", "--out", default=None)
  args = ap.parse_args()
  from train import latest_checkpoint  # pylint: disable=import-outside-toplevel

  ck = latest_checkpoint(args.path)
  out = args.out or os.path.join(
      os.path.dirname(os.path.dirname(ck)) if os.path.basename(
          os.path.dirname(ck)) == "checkpoints" else os.path.dirname(ck),
      "policy.npz")
  export(ck, out)
  print("Guardado:", out, "(desde", ck, ")")


if __name__ == "__main__":
  main()
