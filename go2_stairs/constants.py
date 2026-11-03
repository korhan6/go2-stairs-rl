"""Constantes del Unitree Go2 (orden de articulaciones, pose por defecto, mapeos).

Orden de la POLÍTICA / MuJoCo (igual que MuJoCo Menagerie):
    FL_hip, FL_thigh, FL_calf,
    FR_hip, FR_thigh, FR_calf,
    RL_hip, RL_thigh, RL_calf,
    RR_hip, RR_thigh, RR_calf

Orden del SDK de Unitree (LowCmd/LowState motor_cmd[0..11]):
    FR(0,1,2)  FL(3,4,5)  RR(6,7,8)  RL(9,10,11)
"""

import numpy as np

LEGS = ("FL", "FR", "RL", "RR")
JOINT_NAMES = tuple(
    f"{leg}_{j}_joint" for leg in LEGS for j in ("hip", "thigh", "calf")
)
FEET = LEGS  # geoms de las patas se llaman "FL", "FR", "RL", "RR"
FOOT_SITES = tuple(f"{leg}_foot" for leg in LEGS)

# Pose de pie por defecto (radianes), orden de la política.
# Igual a la usada por unitree_rl_gym para el Go2, ligeramente agachada:
# es la que mejor funciona para escaleras (más recorrido de pata).
DEFAULT_POSE = np.array(
    [
        0.1, 0.8, -1.5,   # FL
        -0.1, 0.8, -1.5,  # FR
        0.1, 1.0, -1.5,   # RL
        -0.1, 1.0, -1.5,  # RR
    ],
    dtype=np.float32,
)

# Escala de la acción por articulación (hip, thigh, calf). La política sale
# en [-1, 1] (tanh), así que esto es la desviación máxima del objetivo PD
# respecto a DEFAULT_POSE. El calf necesita más recorrido para subir escalones.
ACTION_SCALE = np.array([0.4, 0.9, 1.1] * 4, dtype=np.float32)

# Límites de torque de los motores del Go2 (N·m).
TORQUE_LIMIT = np.array([23.7, 23.7, 45.43] * 4, dtype=np.float32)

# Radio de la pata (esfera de colisión), metros.
FOOT_RADIUS = 0.022

# Marcha trote: FL+RR juntos, FR+RL juntos (desfase de media fase).
TROT_OFFSETS = np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32)

# ---------------------------------------------------------------------------
# Mapeos política <-> SDK de Unitree
# ---------------------------------------------------------------------------
# índice SDK de cada articulación de la política
POLICY_TO_SDK = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
# es una permutación que es su propia inversa
SDK_TO_POLICY = POLICY_TO_SDK.copy()
# LowState.foot_force viene en orden FR, FL, RR, RL -> orden política FL,FR,RL,RR
FOOT_FORCE_SDK_TO_POLICY = np.array([1, 0, 3, 2])
