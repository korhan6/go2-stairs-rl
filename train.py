#!/usr/bin/env python3
"""Entrena la política de marcha del Go2 en GPU (MuJoCo MJX/Warp + PPO de Brax).

Receta recomendada (2 etapas):

  # Etapa 1: piso plano/rugoso -> aprende la marcha suave (~15-30 min)
  python train.py --stage flat --gpu 8gb

  # Etapa 2: escaleras, continuando desde la etapa 1 (~1-3 h)
  python train.py --stage stairs --gpu 8gb --resume runs/flat_<fecha>

Al final se exporta `policy.npz` (pesos en numpy) que usan play.py y
deploy_go2.py, sin necesitar JAX.
"""

import argparse
import datetime
import functools
import json
import os
import sys
import time

# --- Opciones de XLA/GPU (antes de importar jax) ---
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# MuJoCo Warp reserva su propia memoria FUERA de JAX (colisiones, etc.):
# hay que dejarle espacio, si no se cae con "Warp CUDA error 2: out of memory".
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.65")
_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_triton_gemm_any" not in _flags:
  os.environ["XLA_FLAGS"] = _flags + " --xla_gpu_triton_gemm_any=True"
os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax  # noqa: E402
from go2_stairs import compat  # noqa: E402,F401  (parche JAX/Brax)
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402
from mujoco_playground import wrapper  # noqa: E402

from go2_stairs import config as cfg_lib  # noqa: E402
from go2_stairs import env as env_lib  # noqa: E402
from go2_stairs import randomize  # noqa: E402

CONTACTS_PER_ENV = 32  # tamaño del buffer de contactos (Warp)


def newest_run_with_checkpoints(root: str = "runs"):
  """El run más reciente (por fecha del último checkpoint) que tenga alguno."""
  best, best_t = None, -1.0
  if not os.path.isdir(root):
    return None
  for d in os.listdir(root):
    ck = os.path.join(root, d, "checkpoints")
    if not os.path.isdir(ck):
      continue
    steps = [s for s in os.listdir(ck) if s.isdigit()]
    if steps:
      t = max(os.path.getmtime(os.path.join(ck, s)) for s in steps)
      if t > best_t:
        best, best_t = os.path.join(root, d), t
  return best


def latest_checkpoint(path: str) -> str:
  """Acepta una carpeta de run o de checkpoint y devuelve el último paso.

  path="auto": el run más reciente que tenga checkpoints.
  """
  if path == "auto":
    found = newest_run_with_checkpoints()
    if found is None:
      raise FileNotFoundError("No encontré ningún run con checkpoints en runs/")
    print("--resume auto ->", found)
    path = found
  path = os.path.abspath(path)
  if not os.path.exists(path):
    raise FileNotFoundError(
        f"No existe {path}. Revisa el nombre con `ls runs/` o usa --resume auto")
  if os.path.exists(os.path.join(path, "ppo_network_config.json")):
    return path
  ckdir = os.path.join(path, "checkpoints")
  if not os.path.isdir(ckdir):
    ckdir = path
  steps = sorted(d for d in os.listdir(ckdir) if d.isdigit())
  if not steps:
    raise FileNotFoundError(f"No hay checkpoints en {ckdir}")
  return os.path.join(ckdir, steps[-1])


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawTextHelpFormatter)
  p.add_argument("--stage", choices=["flat", "stairs", "climb"],
                 default="flat",
                 help="flat -> stairs -> climb (cajas altas, con LiDAR)")
  p.add_argument("--gpu", choices=list(cfg_lib.GPU_PRESETS), default="4gb",
                 help="preset según la VRAM de tu GPU")
  p.add_argument("--impl", choices=["warp", "jax"], default="warp",
                 help="motor de física en GPU (warp recomendado)")
  p.add_argument("--resume", default=None,
                 help="carpeta de un run (o checkpoint) para continuar, o "
                      "'auto' = el último run con checkpoints")
  p.add_argument("--timesteps", type=int, default=None)
  p.add_argument("--num_envs", type=int, default=None)
  p.add_argument("--num_evals", type=int, default=None)
  p.add_argument("--seed", type=int, default=1)
  p.add_argument("--run_name", default=None)
  p.add_argument("--full_reset", action="store_true",
                 help="re-sortea la baldosa en cada reset (más variedad, más lento)")
  p.add_argument("--contact_source", choices=["torque", "sensor", "none"],
                 default="torque",
                 help="cómo siente el piso la política: torque de los motores "
                      "(Go2 Air, por defecto), sensor de pata, o nada")
  p.add_argument("--no_lift", action="store_true",
                 help="desactiva el entrenamiento de 'levantar al perro'")
  p.add_argument("--blind", action="store_true",
                 help="política ciega (sin LiDAR), como la versión anterior")
  p.add_argument("--debug", action="store_true",
                 help="corrida mínima para probar la instalación (segundos)")
  args = p.parse_args()
  # antes de crear nada: si el --resume está mal, falla sin dejar carpetas
  restore = latest_checkpoint(args.resume) if args.resume else None

  print("JAX backend:", jax.default_backend(), jax.devices())
  if jax.default_backend() != "gpu":
    print("\n*** OJO: JAX no está usando la GPU. Revisa la instalación de "
          "jax[cuda12] (ver README). ***\n")

  ppo_params = cfg_lib.ppo_config(args.stage, args.gpu)
  if args.timesteps:
    ppo_params.num_timesteps = args.timesteps
  if args.num_envs:
    ppo_params.num_envs = args.num_envs
  if args.num_evals:
    ppo_params.num_evals = args.num_evals
  num_eval_envs = 128
  if args.debug:
    ppo_params.update(num_envs=16, batch_size=16, num_minibatches=2,
                      num_timesteps=args.timesteps or 3_000, num_evals=2,
                      episode_length=60, num_resets_per_eval=1)
    num_eval_envs = 16

  env_cfg = env_lib.default_config()
  env_cfg.terrain = args.stage
  env_cfg.impl = args.impl
  env_cfg.naconmax = CONTACTS_PER_ENV * max(ppo_params.num_envs, 128)
  env_cfg.contact_source = args.contact_source
  env_cfg.lift_config.enable = not args.no_lift
  env_cfg.perception.enable = not args.blind

  env = env_lib.Go2Stairs(config=env_cfg)
  eval_env = env_lib.Go2Stairs(config=env_cfg)

  stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  run_name = args.run_name or f"{args.stage}_{stamp}"
  run_dir = os.path.abspath(os.path.join("runs", run_name))
  ckpt_dir = os.path.join(run_dir, "checkpoints")
  os.makedirs(ckpt_dir, exist_ok=True)
  with open(os.path.join(run_dir, "env_config.json"), "w") as f:
    f.write(env_cfg.to_json_best_effort(indent=2))
  with open(os.path.join(run_dir, "ppo_config.json"), "w") as f:
    f.write(ppo_params.to_json_best_effort(indent=2))

  if restore:
    print("Reanudando desde:", restore)

  print(f"Run: {run_dir}")
  print(f"Etapa: {args.stage} | envs: {ppo_params.num_envs} | "
        f"pasos: {ppo_params.num_timesteps:,} | física: {args.impl}")
  print("Obs:", env.observation_size,
        "| percepción:", "LiDAR" if env_cfg.perception.enable else "ciega")

  log_path = os.path.join(run_dir, "progress.jsonl")
  t0 = time.time()
  times = [t0]

  def progress(num_steps, metrics):
    times.append(time.time())
    rew = metrics.get("eval/episode_reward", float("nan"))
    rew_std = metrics.get("eval/episode_reward_std", float("nan"))
    ep_len = metrics.get("eval/avg_episode_length", float("nan"))
    sps = metrics.get("training/sps", float("nan"))
    elapsed = (times[-1] - t0) / 60
    lin = metrics.get("eval/episode_reward/tracking_lin_vel", float("nan"))
    print(f"[{elapsed:6.1f} min] paso {num_steps:>12,} | recompensa "
          f"{rew:8.2f} ± {rew_std:6.2f} | largo ep {ep_len:6.1f} | "
          f"tracking {lin:7.2f} | {sps:,.0f} pasos/s", flush=True)
    with open(log_path, "a") as f:
      row = {k: float(v) for k, v in metrics.items()
             if hasattr(v, "__float__")}
      row["step"] = int(num_steps)
      f.write(json.dumps(row) + "\n")

  network_factory = functools.partial(
      ppo_networks.make_ppo_networks, **ppo_params.network_factory)
  rand_fn = functools.partial(
      randomize.domain_randomize,
      floor_geom_id=env.mj_model.geom("floor").id)
  wrap_fn = functools.partial(wrapper.wrap_for_brax_training,
                              full_reset=args.full_reset)

  params = ppo_params.to_dict()
  params.pop("network_factory")
  train_fn = functools.partial(
      ppo.train,
      **params,
      network_factory=network_factory,
      randomization_fn=rand_fn,
      wrap_env_fn=wrap_fn,
      seed=args.seed,
      num_eval_envs=num_eval_envs,
      progress_fn=progress,
      save_checkpoint_path=ckpt_dir,
      restore_checkpoint_path=restore,
  )

  make_inference_fn, final_params, _ = train_fn(
      environment=env, eval_env=eval_env)
  print(f"Entrenamiento listo en {(time.time() - t0) / 60:.1f} min")

  # Exporta automáticamente el último checkpoint a numpy.
  from export_policy import export  # pylint: disable=import-outside-toplevel
  out = os.path.join(run_dir, "policy.npz")
  export(latest_checkpoint(run_dir), out, env_cfg)
  print("Política exportada:", out)
  print(f"Pruébala:  python play.py --policy {out}")
  del make_inference_fn, final_params


if __name__ == "__main__":
  main()
