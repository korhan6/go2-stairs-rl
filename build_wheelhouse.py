#!/usr/bin/env python3
"""Recupera de la caché de pip las ruedas que YA bajaste, para no volver a bajarlas.

pip guarda cada rueda descargada en ~/.cache/pip/http-v2/**/<hash>.body con un
nombre ilegible. Este script las abre, lee su nombre/versión/tags y las copia a
./wheelhouse/ con su nombre correcto, para instalarlas con
`pip install --no-index --find-links wheelhouse`.

Además escribe wheelhouse/constraints.txt fijando la pila JAX + CUDA 12 a las
versiones que ya tienes, para que pip no intente bajar otras más nuevas.

Uso:  python build_wheelhouse.py [--cache ~/.cache/pip] [--out wheelhouse]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import zipfile

try:
  from packaging.tags import parse_tag, sys_tags
  from packaging.version import Version
except ImportError:  # venv recién creado: usar la copia que trae pip
  from pip._vendor.packaging.tags import parse_tag, sys_tags  # type: ignore
  from pip._vendor.packaging.version import Version  # type: ignore

# Paquetes que se fijan a la versión de la caché (lo pesado de bajar).
PIN_RE = re.compile(r"^(jax|jaxlib|jax-cuda12-.*|nvidia-.*-cu12)$")


def norm(name: str) -> str:
  return re.sub(r"[-_.]+", "-", name).lower()


def wheel_info(path: str):
  """(nombre, versión, [tags]) si el archivo es una rueda, si no None."""
  try:
    z = zipfile.ZipFile(path)
  except (zipfile.BadZipFile, OSError):
    return None
  with z:
    names = set(z.namelist())
    for n in names:
      top = n.split("/", 1)[0]
      if (top.endswith(".dist-info") and f"{top}/WHEEL" in names
          and f"{top}/METADATA" in names):
        wheel = z.read(f"{top}/WHEEL").decode("utf-8", "replace")
        meta = z.read(f"{top}/METADATA").decode("utf-8", "replace")
        break
    else:
      return None
  m_name = re.search(r"^Name:\s*(.+?)\s*$", meta, re.M)
  m_ver = re.search(r"^Version:\s*(.+?)\s*$", meta, re.M)
  tags = re.findall(r"^Tag:\s*(.+?)\s*$", wheel, re.M)
  if not (m_name and m_ver and tags):
    return None
  return m_name.group(1), m_ver.group(1), tags


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--cache", default=os.path.expanduser("~/.cache/pip"))
  ap.add_argument("--out", default="wheelhouse")
  args = ap.parse_args()

  supported = {t: i for i, t in enumerate(sys_tags())}
  os.makedirs(args.out, exist_ok=True)

  found = {}  # norm(nombre) -> (Version, nombre)
  n_ok = 0
  for root, _, files in os.walk(args.cache):
    for f in files:
      if not f.endswith(".body"):
        continue
      path = os.path.join(root, f)
      info = wheel_info(path)
      if info is None:
        continue
      name, ver, tags = info
      # el tag compatible con ESTE python/plataforma que pip prefiera
      best = None
      for t in tags:
        for tag in parse_tag(t):
          r = supported.get(tag)
          if r is not None and (best is None or r < best[0]):
            best = (r, t)
      if best is None:
        continue  # otra versión de python u otra plataforma
      fname = f"{re.sub(r'[-_.]+', '_', name)}-{ver}-{best[1]}.whl"
      dst = os.path.join(args.out, fname)
      if not os.path.exists(dst):
        try:
          os.link(path, dst)  # sin copiar bytes si está en el mismo disco
        except OSError:
          shutil.copy2(path, dst)
      n_ok += 1
      key = norm(name)
      try:
        v = Version(ver)
      except Exception:  # pylint: disable=broad-except
        continue
      if key not in found or v > found[key][0]:
        found[key] = (v, ver)

  pins = sorted(f"{k}=={v[1]}" for k, v in found.items() if PIN_RE.match(k))
  with open(os.path.join(args.out, "constraints.txt"), "w") as fh:
    fh.write("\n".join(pins) + "\n")

  print(f"{n_ok} ruedas compatibles recuperadas en ./{args.out}/")
  print("Pila JAX/CUDA encontrada en la caché:")
  for p in pins:
    print("  ", p)
  if not any(p.startswith("jaxlib==") for p in pins):
    print("  (no hay jaxlib en la caché: habrá que bajarlo)")
  if not any(p.startswith("nvidia-cudnn-cu12==") for p in pins):
    print("  (no hay cuDNN cu12 en la caché: habrá que bajarlo, ~700 MB)")
  return 0


if __name__ == "__main__":
  sys.exit(main())
