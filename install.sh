#!/usr/bin/env bash
# Instalación en Ubuntu 22.04/24.04 con GPU NVIDIA.
# Uso:  bash install.sh            (crea ./.venv)
set -e
cd "$(dirname "$0")"

if ! command -v nvidia-smi >/dev/null; then
  echo "No encuentro nvidia-smi: instala el driver NVIDIA (sudo ubuntu-drivers install) y reinicia."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

sudo apt-get install -y python3-venv python3-dev git libegl1 libgl1 ffmpeg >/dev/null

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
# JAX con CUDA 12 incluido (no necesitas instalar CUDA aparte, solo el driver).
pip install -U "jax[cuda12]"
pip install -r requirements.txt

echo
echo "== Verificando GPU =="
python -c "import jax; print('JAX backend:', jax.default_backend(), jax.devices())"
echo
echo "== Descargando MuJoCo Menagerie (modelo del Go2) =="
python -c "from mujoco_playground._src import mjx_env; mjx_env.ensure_menagerie_exists(); print('OK')"
echo
echo "Listo. Activa el entorno con:  source .venv/bin/activate"
echo "Prueba rápida:                 python train.py --debug"
