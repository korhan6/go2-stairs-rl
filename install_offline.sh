#!/usr/bin/env bash
# Instala JAX con GPU (CUDA 12) + playground reutilizando las ruedas que ya
# están en tu caché de pip (~/.cache/pip). Solo baja lo que falte.
# Uso:  bash install_offline.sh
set -e
cd "$(dirname "$0")"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  || { echo "nvidia-smi no funciona: revisa el driver NVIDIA"; exit 1; }

[ -x .venv/bin/python ] || python3 -m venv .venv
source .venv/bin/activate

echo "== 1/3 Recuperando ruedas de la caché de pip =="
python build_wheelhouse.py --out wheelhouse

PKGS=("jax[cuda12]" -r requirements.txt)
C=(-c wheelhouse/constraints.txt)

echo
echo "== 2/3 Instalando SIN internet (solo caché) =="
if pip install --no-index --find-links wheelhouse "${C[@]}" "${PKGS[@]}"; then
  echo "-> Todo salió de la caché, 0 MB bajados."
else
  echo
  echo "-> Faltan algunos paquetes en la caché. Bajando SOLO esos"
  echo "   (la pila CUDA queda fijada a la versión que ya tienes)..."
  if ! pip install --find-links wheelhouse "${C[@]}" "${PKGS[@]}"; then
    echo
    echo "-> Las versiones de la caché chocan con algún requisito."
    echo "   Reintentando sin fijarlas (puede bajar más)..."
    pip install --find-links wheelhouse "${PKGS[@]}"
  fi
fi

echo
echo "== 3/3 Verificando GPU =="
python - <<'EOF'
import jax, jax.numpy as jp
print("JAX", jax.__version__, "| backend:", jax.default_backend(), jax.devices())
if jax.default_backend() != "gpu":
    raise SystemExit("*** JAX NO ve la GPU. Pégale esta salida a Claude. ***")
x = jp.ones((2048, 2048)); print("matmul en GPU OK:", float((x @ x)[0, 0]))
EOF

python -c "from mujoco_playground._src import mjx_env; mjx_env.ensure_menagerie_exists(); print('Menagerie OK')" \
  || echo "(No pude bajar MuJoCo Menagerie ahora; se baja sola en el primer train.py)"

echo
echo "Listo. Ahora:"
echo "  source .venv/bin/activate"
echo "  python train.py --debug            # prueba de 1-2 min"
echo "  python train.py --stage flat       # (preset 4gb por defecto para tu RTX 3050)"
