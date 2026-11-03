"""Aleatorización de dominio (para que la política funcione en el Go2 real)."""

import jax
from mujoco import mjx

BASE_BODY_ID = 1


def domain_randomize(model: mjx.Model, rng: jax.Array, floor_geom_id: int):
  @jax.vmap
  def rand(rng):
    k = jax.random.split(rng, 9)
    # Fricción del piso: U(0.3, 1.2) (baldosa pulida ... concreto).
    geom_friction = model.geom_friction.at[floor_geom_id, 0].set(
        jax.random.uniform(k[0], minval=0.3, maxval=1.2))
    # Fricción estática de las articulaciones: *U(0.5, 1.5).
    frictionloss = model.dof_frictionloss.at[6:].multiply(
        jax.random.uniform(k[1], (12,), minval=0.5, maxval=1.5))
    # Armadura (inercia del rotor): *U(0.8, 1.2).
    armature = model.dof_armature.at[6:].multiply(
        jax.random.uniform(k[2], (12,), minval=0.8, maxval=1.2))
    # Centro de masa del tronco: +-3 cm.
    body_ipos = model.body_ipos.at[BASE_BODY_ID].add(
        jax.random.uniform(k[3], (3,), minval=-0.03, maxval=0.03))
    # Masas de todos los cuerpos: *U(0.9, 1.1).
    body_mass = model.body_mass * jax.random.uniform(
        k[4], (model.nbody,), minval=0.9, maxval=1.1)
    # Carga extra en el tronco: U(-0.5, 2.0) kg (batería, Jetson, LiDAR...).
    body_mass = body_mass.at[BASE_BODY_ID].add(
        jax.random.uniform(k[5], minval=-0.5, maxval=2.0))
    # Ganancias PD: Kp *U(0.85, 1.15), Kd *U(0.8, 1.2).
    kp_s = jax.random.uniform(k[6], (12,), minval=0.85, maxval=1.15)
    kd_s = jax.random.uniform(k[7], (12,), minval=0.8, maxval=1.2)
    gainprm = model.actuator_gainprm.at[:, 0].multiply(kp_s)
    biasprm = model.actuator_biasprm.at[:, 1].multiply(kp_s)
    biasprm = biasprm.at[:, 2].multiply(kd_s)
    # Amortiguamiento pasivo de las articulaciones: *U(0.5, 2.0).
    damping = model.dof_damping.at[6:].multiply(
        jax.random.uniform(k[8], (12,), minval=0.5, maxval=2.0))
    return (geom_friction, frictionloss, armature, body_ipos, body_mass,
            gainprm, biasprm, damping)

  (geom_friction, frictionloss, armature, body_ipos, body_mass, gainprm,
   biasprm, damping) = rand(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "body_ipos": 0,
      "body_mass": 0,
      "actuator_gainprm": 0,
      "actuator_biasprm": 0,
      "dof_damping": 0,
  })
  model = model.tree_replace({
      "geom_friction": geom_friction,
      "dof_frictionloss": frictionloss,
      "dof_armature": armature,
      "body_ipos": body_ipos,
      "body_mass": body_mass,
      "actuator_gainprm": gainprm,
      "actuator_biasprm": biasprm,
      "dof_damping": damping,
  })
  return model, in_axes
