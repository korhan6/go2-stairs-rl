"""Detección de contacto de las patas SIN sensores de presión.

El Go2 Air no reporta presión en las patas, pero cada motor reporta su
torque estimado (`tau_est`, calculado a partir de la corriente). Con la
cinemática de la pata se convierte ese torque en la fuerza que el piso hace
sobre el pie:

    tau = -J(q)^T F      =>      F = -(J^T)^-1 tau

(J = Jacobiano 3x3 de la posición del pie respecto a las 3 articulaciones de
esa pata, en el marco del cuerpo). La pata está apoyada si la componente
VERTICAL de F (usando la gravedad medida por la IMU) supera un umbral.

Se ignora la inercia/peso de la propia pata (~1.6 kg): eso equivale a unos
pocos newtons, muy por debajo del umbral (~15 N; una pata de apoyo carga
30-80 N).

El MISMO código se usa en el entrenamiento (con jax.numpy, torque del
simulador), en play.py y en el robot real (con numpy, tau_est), así la
política ve exactamente la misma señal en simulación y en el robot.
"""

# Geometría del Go2 (igual que el MJCF / URDF de Unitree), orden FL, FR, RL, RR.
HIP_POS = ((0.1934, 0.0465, 0.0), (0.1934, -0.0465, 0.0),
           (-0.1934, 0.0465, 0.0), (-0.1934, -0.0465, 0.0))
SIDE = (1.0, -1.0, 1.0, -1.0)
THIGH_OFFSET = 0.0955
THIGH_LEN = 0.213
FOOT_OFFSET = (-0.002, 0.0, -0.213)

DEFAULT_THRESHOLD = 15.0  # N de fuerza vertical para considerar apoyo


def foot_position(xp, q3, leg):
  """Posición del pie en el marco del cuerpo. q3 = (hip, thigh, calf)."""
  q0, q1, q2 = q3[0], q3[1], q3[2]
  hx, hy, hz = HIP_POS[leg]
  # pie respecto al muslo: [0,0,-L] + Ry(q2) @ FOOT_OFFSET
  fx, _, fz = FOOT_OFFSET
  c2, s2 = xp.cos(q2), xp.sin(q2)
  kx = c2 * fx + s2 * fz
  kz = -s2 * fx + c2 * fz - THIGH_LEN
  # rotación del muslo Ry(q1)
  c1, s1 = xp.cos(q1), xp.sin(q1)
  tx = c1 * kx + s1 * kz
  tz = -s1 * kx + c1 * kz
  ty = SIDE[leg] * THIGH_OFFSET
  # abducción Rx(q0)
  c0, s0 = xp.cos(q0), xp.sin(q0)
  return xp.stack([hx + tx, hy + c0 * ty - s0 * tz, hz + s0 * ty + c0 * tz])


def leg_jacobian(xp, q3, leg, eps=1e-4):
  """Jacobiano 3x3 por diferencias centrales (idéntico en numpy y jax)."""
  eye = xp.eye(3) * eps
  cols = []
  for j in range(3):
    dq = eye[j]
    cols.append((foot_position(xp, q3 + dq, leg)
                 - foot_position(xp, q3 - dq, leg)) / (2 * eps))
  return xp.stack(cols, axis=1)


def foot_forces(xp, qpos12, tau12):
  """Fuerza del piso sobre cada pie (4x3) en el marco del cuerpo."""
  forces = []
  for leg in range(4):
    q3 = qpos12[3 * leg:3 * leg + 3]
    t3 = tau12[3 * leg:3 * leg + 3]
    J = leg_jacobian(xp, q3, leg)
    forces.append(xp.linalg.solve(J.T, -t3))
  return xp.stack(forces)


def vertical_forces(xp, qpos12, tau12, gravity):
  """Componente vertical (hacia arriba, N) de la fuerza de cada pie.

  gravity: gravedad proyectada en el cuerpo (vector unitario hacia abajo).
  """
  f = foot_forces(xp, qpos12, tau12)
  g = gravity / (xp.linalg.norm(gravity) + 1e-6)
  return -(f @ g)


def estimate_contacts(xp, qpos12, tau12, gravity,
                      threshold=DEFAULT_THRESHOLD):
  fz = vertical_forces(xp, qpos12, tau12, gravity)
  return fz > threshold, fz
