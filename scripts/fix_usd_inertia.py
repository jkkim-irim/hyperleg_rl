# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""One-shot patch: copy inertial data from HyperLeg.xml into HyperLeg.usd.

The pre-converted ``HyperLeg.usd`` ships with ``physics:mass`` and
``physics:centerOfMass`` but ``physics:diagonalInertia`` is zero (or
``MassAPI`` is missing entirely on some bodies). This script reads
``<inertial>`` blocks from the source MJCF and applies
``UsdPhysics.MassAPI`` (mass / diagonal inertia / center of mass /
principal axes) on every matching RigidBody prim in the USD so that the
loaded articulation has proper inertial properties.

Run once:
    ./isaaclab.sh -p projects/hyperleg_physx/scripts/fix_usd_inertia.py
"""

from __future__ import annotations

import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from pxr import Gf, Usd, UsdPhysics


PROJECTS_ROOT = Path(__file__).resolve().parents[2]
XML_PATH = PROJECTS_ROOT / "assets" / "hyperleg" / "HyperLeg.xml"
USD_PATH = PROJECTS_ROOT / "assets" / "hyperleg" / "HyperLeg.usd"
XML_BODY_PREFIX = "_HyperLegRL_"


def parse_xml_inertials(xml_path: Path) -> dict[str, dict]:
    """Return ``{prim_name: {"pos", "mass", "diag", "quat"}}``.

    Prim name = XML body name with the ``_HyperLegRL_`` prefix stripped.
    """
    tree = ET.parse(xml_path)
    out: dict[str, dict] = {}
    for body in tree.iter("body"):
        name = body.get("name") or ""
        inert = body.find("inertial")
        if inert is None:
            continue
        prim_name = name[len(XML_BODY_PREFIX):] if name.startswith(XML_BODY_PREFIX) else name
        pos = tuple(float(x) for x in (inert.get("pos") or "0 0 0").split())
        mass = float(inert.get("mass") or 0.0)
        diag = tuple(float(x) for x in (inert.get("diaginertia") or "0 0 0").split())
        quat_str = inert.get("quat")
        quat = tuple(float(x) for x in quat_str.split()) if quat_str else None
        out[prim_name] = {"pos": pos, "mass": mass, "diag": diag, "quat": quat}
    return out


def apply_mass_api(prim: Usd.Prim, info: dict) -> None:
    """Apply ``UsdPhysics.MassAPI`` with mass / inertia / CoM / principal axes."""
    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr(float(info["mass"]))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*info["diag"]))
    api.CreateCenterOfMassAttr(Gf.Vec3f(*info["pos"]))
    if info["quat"] is not None:
        # MuJoCo quat is (w, x, y, z); Gf.Quatf takes (real, imaginary-vec3).
        w, x, y, z = info["quat"]
        api.CreatePrincipalAxesAttr(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def main() -> int:
    if not USD_PATH.is_file():
        print(f"[ERR] USD not found: {USD_PATH}", file=sys.stderr)
        return 1
    if not XML_PATH.is_file():
        print(f"[ERR] XML not found: {XML_PATH}", file=sys.stderr)
        return 1

    inertials = parse_xml_inertials(XML_PATH)
    print(f"[INFO] Parsed {len(inertials)} <inertial> blocks from {XML_PATH.name}")

    backup = USD_PATH.with_suffix(USD_PATH.suffix + f".bak.{datetime.now():%Y%m%d-%H%M%S}")
    shutil.copy2(USD_PATH, backup)
    print(f"[INFO] Backup -> {backup.name}")

    stage = Usd.Stage.Open(str(USD_PATH))
    patched, skipped = 0, []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        name = prim.GetName()
        info = inertials.get(name)
        if info is None:
            skipped.append((name, str(prim.GetPath())))
            continue
        apply_mass_api(prim, info)
        patched += 1
        print(
            f"  patched {name:10s} mass={info['mass']} "
            f"diag={info['diag']} com={info['pos']}"
            + (" +quat" if info["quat"] is not None else "")
        )

    if skipped:
        print(f"[WARN] No XML inertial for {len(skipped)} body prims:")
        for name, path in skipped:
            print(f"   - {name} ({path})")

    stage.GetRootLayer().Save()
    print(f"[OK] Saved {USD_PATH.name} with {patched} bodies updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
