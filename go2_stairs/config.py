"""Configuración de PPO y presets según la VRAM de tu GPU."""

from ml_collections import config_dict

# num_envs, batch_size, num_minibatches  (batch*minibatches % num_envs == 0)
GPU_PRESETS = {
    "4gb": (2048, 256, 8),
    "8gb": (4096, 256, 16),     # laptops RTX 3060/4060/4070
    "12gb": (8192, 256, 32),    # RTX 3080/4070 Ti/4080
    "24gb": (16384, 512, 32),   # RTX 3090/4090
}

STAGE_TIMESTEPS = {
    "flat": 100_000_000,     # ~aprender la marcha suave
    "stairs": 400_000_000,   # escaleras (reanudando desde flat)
    "climb": 600_000_000,    # cajas altas + escaleras de 23 cm (desde stairs)
}


def ppo_config(stage: str, preset: str) -> config_dict.ConfigDict:
  num_envs, batch_size, num_minibatches = GPU_PRESETS[preset]
  return config_dict.create(
      num_timesteps=STAGE_TIMESTEPS[stage],
      num_evals=10 if stage == "flat" else 20,
      reward_scaling=1.0,
      episode_length=1000,
      normalize_observations=True,
      action_repeat=1,
      unroll_length=20,
      num_minibatches=num_minibatches,
      num_updates_per_batch=4,
      discounting=0.97,
      learning_rate=3e-4,
      entropy_cost=0.01 if stage == "flat" else 0.005,
      num_envs=num_envs,
      batch_size=batch_size,
      max_grad_norm=1.0,
      clipping_epsilon=0.2,
      # 0: el currículo de terreno (env.py) mueve a cada robot de baldosa; un
      # reset global lo devolvería a los niveles fáciles.
      num_resets_per_eval=0,
      network_factory=config_dict.create(
          policy_hidden_layer_sizes=(512, 256, 128),
          value_hidden_layer_sizes=(512, 256, 128),
          policy_obs_key="state",
          value_obs_key="privileged_state",
      ),
  )
