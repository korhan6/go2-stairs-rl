# Go2 Stairs RL — marcha suave tipo Spot + escaleras (MuJoCo, GPU)

Política de locomoción para el **Unitree Go2** entrenada con **RL (PPO)** en
**MuJoCo MJX / MuJoCo Warp** sobre tu GPU NVIDIA. Lo que aprende:

| Qué | Cómo se logra |
|---|---|
| Marcha fluida tipo Spot (trote regular, cuerpo estable) | Reloj de marcha a 2 Hz (FL+RR / FR+RL), recompensas de suavidad: tasa y "jerk" de la acción, aceleración articular, energía, altura y orientación del cuerpo |
| Subir y bajar escaleras (adelante, atrás, de lado) | Terreno procedural con pirámides de escaleras de 5 → 19 cm, foso/plataforma, rampas, bloques. Recompensa de **altura del pie sobre el escalón** y penalización si la canilla/muslo golpea el borde |
| **Ver el terreno (LiDAR L1 simulado)** | La política recibe una grilla de 17×11 alturas alrededor del robot (cada 10 cm, ±0.8 m adelante/atrás, ±0.5 m a los lados), como la daría un mapa de elevación hecho con el L1. En simulación se arma con **oclusión real desde la cabeza** (no ve detrás del borde de una caja alta), ruido de 2 cm, puntos falsos, huecos, deriva de odometría y episodios **sin mapa** (sigue caminando si el LiDAR falla). Ver `go2_stairs/perception.py` |
| Levantar la pata según lo que ve | Recompensa de altura de paso que **mira adelante**: si en la dirección del comando hay un escalón o una caja, la pata en vuelo debe pasar por encima. Penalización si la pata choca de frente con la contrahuella (`stumble`) y se permite cabecear lo que pide el terreno |
| Subir cosas altas | Etapa `climb`: escaleras de hasta 23 cm, **cajas de hasta 40 cm** (anillo alrededor del robot: sube y baja en cualquier dirección), bloques de 25 cm. **Currículo**: cada robot sube de nivel si cruza su baldosa y baja si no avanza |
| "Sentir" el piso (sin sensores en las patas) | El Go2 Air no mide presión en las patas, así que el contacto se **estima con el torque de los motores** (`tau_est`, que el robot calcula de la corriente): con la cinemática de cada pata, `F = -(Jᵀ)⁻¹·τ`, y la pata está apoyada si la fuerza vertical supera 15 N (ver `go2_stairs/contact.py`). El mismo cálculo se hace en el entrenamiento, en `play.py` y en el robot. Con eso + historial de 5 pasos de IMU y articulaciones, la política detecta cuándo un pie pega contra un escalón o pisa en falso y corrige |
| Pataditas cuando lo levantas | En el entrenamiento a veces se "levanta" al perro con una fuerza (como si lo cargaras) 1–3 s y luego se suelta. En el aire se le premia por seguir dando pasitos; al soltarlo aprende a aterrizar |
| Equilibrio / robustez | Empujones aleatorios, ruido de sensores, latencia, aleatorización de fricción (0.3–1.2), masas (+2 kg de carga), ganancias PD |
| Que funcione en el robot real | Actor-crítico asimétrico: el crítico ve info privilegiada (mapa de alturas, velocidad real, fricción), pero la **política solo usa sensores que el Go2 tiene** |

Basado en [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
(Google DeepMind) y el modelo del Go2 de MuJoCo Menagerie.

---

## 1. Instalación (Ubuntu 24.04 + NVIDIA)

Necesitas el driver NVIDIA instalado (`nvidia-smi` debe funcionar). CUDA lo trae JAX.

```bash
cd go2_stairs_rl
bash install.sh                 # crea .venv, instala jax[cuda12] + playground
source .venv/bin/activate
python train.py --debug         # prueba de 1-2 min: si termina, todo está bien
```

Instalación manual equivalente:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip "jax[cuda12]"
pip install -r requirements.txt
python -c "import jax; print(jax.default_backend())"   # debe decir: gpu
```

### Instalación con poco internet (reusar la caché de pip)

Si ya bajaste alguna vez `jax[cuda12]` o torch con CUDA, las ruedas siguen en
`~/.cache/pip`. Este instalador las recupera e instala sin internet; solo baja
lo que no encuentre (la pila CUDA queda fijada a las versiones que ya tienes):

```bash
bash install_offline.sh
```

## 2. Entrenar (3 etapas)

Elige el preset según la VRAM de tu GPU: `--gpu 4gb | 8gb | 12gb | 24gb` (por defecto `4gb`)
(mira la memoria con `nvidia-smi`).

```bash
# Etapa 1 — piso plano/rugoso: aprende la marcha suave (100 M pasos)
python train.py --stage flat --gpu 8gb

# Etapa 2 — escaleras, continuando desde la etapa 1 (400 M pasos)
python train.py --stage stairs --gpu 8gb --resume runs/flat_AAAAMMDD_HHMMSS

# Etapa 3 — cajas altas + escaleras de 23 cm, desde la etapa 2 (600 M pasos)
python train.py --stage climb --gpu 8gb --resume runs/stairs_AAAAMMDD_HHMMSS
```

* Por defecto la política usa el **LiDAR simulado**. `--blind` entrena la
  versión ciega anterior. Ojo: una política ciega y una con LiDAR tienen
  entradas distintas, así que **no se puede reanudar una con la otra**
  (hay que entrenar las 3 etapas con la misma opción).
* Con 4 GB de VRAM (`--gpu 4gb`) cuenta con más horas: el LiDAR agrega
  ~190 rayos por robot y paso.

* Cada evaluación guarda un checkpoint en `runs/<run>/checkpoints/`. Si cortas
  con `Ctrl+C`, sigue con `--resume runs/<run>`.
* Al terminar se exporta solo **`runs/<run>/policy.npz`** (red en numpy, sin JAX).
* Exportar un checkpoint intermedio: `python export_policy.py runs/<run>`

Qué mirar en el log:

```
[  12.3 min] paso   52,428,800 | recompensa    14.20 ± 2.1 | largo ep  987.0 | tracking  21.30 | 180,000 pasos/s
```

* **largo ep** cerca de 1000 = casi no se cae (episodio de 20 s).
* **recompensa** debe subir y estabilizarse. En la etapa de escaleras baja al
  principio (terreno nuevo) y vuelve a subir.
* Tiempos orientativos: etapa 1 ≈ 15–40 min, etapa 2 ≈ 1–3 h, según la GPU.

Opciones útiles: `--timesteps 600000000` (entrenar más), `--impl jax`
(si Warp da problemas), `--contact_source torque|sensor|none` (por defecto
`torque`, el del Go2 Air), `--no_lift` (sin entrenamiento de levantarlo),
`--full_reset` (más variedad).

## 3. Probar en simulación (sim2sim)

**Mirar mientras entrena** (en otra terminal; corre en la CPU, no le quita GPU):

```bash
source .venv/bin/activate
python watch.py            # toma solo el run más reciente y recarga cada checkpoint nuevo
```

El perro camina solo con un guion; si tocas una flecha pasas a control manual.

```bash
python play.py --policy runs/stairs_XXXX/policy.npz                     # empieza en un foso de escaleras
python play.py --policy runs/stairs_XXXX/policy.npz --tile stairs_up --level 7   # arriba de la pirámide más alta
python play.py --policy runs/stairs_XXXX/policy.npz --video demo.mp4    # video automático (sube, gira, baja, lo levantan)
```

Teclas en el visor: **↑/↓** adelante/atrás · **←/→** girar · **, .** de lado ·
**Backspace** parar · **Enter** levantar/soltar (pataditas) ·
**Ctrl + clic derecho arrastrando** sobre el perro = empujarlo a mano.

`play.py` usa MuJoCo normal (no MJX) y el mismo código de observaciones que el
robot real (`go2_stairs/runtime.py`), así que es la prueba previa al despliegue.

Con una política con LiDAR:

```bash
python play.py --policy runs/climb_XXXX/policy.npz --terrain climb_only --tile box_ring --level 9   # cajas de 40 cm
python play.py --policy runs/climb_XXXX/policy.npz --terrain climb_only --tile stairs_down --level 9  # escalones de 23 cm
python play.py --policy runs/climb_XXXX/policy.npz --terrain climb_only --tile box_ring --level 9 --blind  # sin mapa
```

En el visor se dibujan los puntos del escaneo: **verde** = el LiDAR los ve,
**rojo** = ocultos (detrás de un borde) o fuera de su campo de visión.

## 4. Ajustar el estilo de marcha

En `go2_stairs/env.py → default_config()`:

| Parámetro | Efecto |
|---|---|
| `gait.freq` (2.0 Hz) | Cadencia. Más alto = pasos más cortos y rápidos |
| `gait.swing_height` (0.10 m) | Cuánto levanta la pata. Súbelo a 0.12 si tropieza en escalones altos |
| `reward_config.scales.action_rate / action_smoothness / dof_acc` | Más negativos = más suave (pero más "perezoso") |
| `reward_config.base_height_target` (0.30 m) | Altura del cuerpo |
| `command_config.a` | Velocidades máximas [vx, vy, wz] |
| `Kp`, `Kd` (25 / 0.6) | Rigidez del PD. Deben ser iguales en el robot real |

Cambiar la marcha requiere re-entrenar (puedes reanudar con `--resume` para ir más rápido).

## 5. Robot real (EXPERIMENTAL — `deploy_go2.py`)

Envía comandos de bajo nivel (`rt/lowcmd`) con `unitree_sdk2_python`. Requiere
que el robot acepte control de bajo nivel (EDU o Air/Pro con jailbreak y
funciones EDU desbloqueadas). Lee `tau_est` de cada motor para estimar el
contacto de las patas; no necesita `foot_force`.

**Antes del robot real, pruébalo contra el simulador DDS de Unitree**
([unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco), `simulate_python`):

```bash
python deploy_go2.py --policy runs/stairs_XXXX/policy.npz --sim
```

Lista de seguridad para el robot real:

1. Perro **echado** en el piso, área despejada, alguien al lado.
2. Primera vez: **colgado de un arnés** o sostenido, con las patas sin tocar el piso.
3. Revisa la detección de contacto por torque: corre con `--show_contacts`,
   párate con `1` y mira la fuerza estimada por pata. De pie cada pata debe
   marcar **~30–40 N**; si lo levantas (con los motores rígidos), **~0 N**.
   Si alguna pata de pie marca menos de ~20 N, baja el umbral con
   `--contact_threshold 12` (por defecto 15 N, el mismo del entrenamiento).
4. Ejecuta (cable Ethernet en `eno1`, red 192.168.123.x):
   ```bash
   python deploy_go2.py --policy runs/stairs_XXXX/policy.npz --iface eno1
   ```
5. `1` = pararse (2 s) → `2` = activar la política → `w/x a/d z/c` comandos.
   **ESPACIO = amortiguar** (se echa suave). También se amortigua solo si se
   inclina más de ~70°, si una articulación sale de rango o si se pierde el LowState.
6. Empieza en plano y velocidades bajas; escaleras solo cuando camine perfecto.

El script libera el modo deportivo de fábrica; para volver a él reinicia el perro.

### LiDAR en el robot real (pendiente)

`deploy_go2.py` todavía **no tiene el mapa de elevación conectado**: una
política con LiDAR corre ahí en modo "sin mapa" (camina, pero no anticipa
escalones) y lo avisa al arrancar. Para conectarlo, completa
`get_height_scan()` en `deploy_go2.py`: consulta tu mapa de elevación (hecho
con el L1 + odometría, p. ej. elevation_mapping o un grid con la nube de
GLIM) en los 187 puntos de la grilla y usa
`runner.scan_from_height_fn(altura, pos, R, occlusion=False)`. Las celdas sin
dato van como `NaN`. **No uses la política en escaleras o cajas reales
hasta que el mapa esté conectado y lo verifiques.**

## 6. Estructura

```
go2_stairs/
  env.py         entorno MJX: observaciones, recompensas, levantar/empujar
  terrain.py     generador de escaleras / rampas / rugoso (heightfield)
  robot_xml.py   MJCF del Go2 con colisiones para escaleras
  randomize.py   aleatorización de dominio
  config.py      PPO y presets por GPU
  runtime.py     política en numpy (sim2sim y robot real)
  contact.py     contacto de las patas estimado por torque de los motores
  perception.py  LiDAR L1 simulado -> escaneo de alturas (oclusión, FOV)
  constants.py   orden de articulaciones, pose, mapeo SDK de Unitree
train.py         entrenamiento en GPU
play.py          visor / video
export_policy.py checkpoint -> policy.npz
deploy_go2.py    robot real (experimental)
```

Orden de articulaciones de la política: `FL, FR, RL, RR` × `hip, thigh, calf`
(el SDK de Unitree usa `FR, FL, RR, RL`; `constants.py` hace el mapeo).

## 7. Problemas comunes

* **`JAX backend: cpu`** → `pip install -U "jax[cuda12]"` dentro del venv; revisa `nvidia-smi`.
* **Sin memoria (OOM / RESOURCE_EXHAUSTED)** → usa un preset menor (`--gpu 4gb`)
  o `--num_envs 1024`.
* **Error de Warp / contactos (`naconmax`)** → prueba `--impl jax` (más lento pero muy estable).
* **El video no se genera (EGL)** → `sudo apt install libegl1` o usa `MUJOCO_GL=osmesa`.
* **Tropieza en escalones altos** → entrena más (`--timesteps 600000000 --resume ...`) o sube `gait.swing_height`.
# go2-stairs-rl
