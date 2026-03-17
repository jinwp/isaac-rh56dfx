"""Prebuild cached USD files for RH56DFX left/right assets."""

from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Build RH56DFX USD assets from cached URDF files.")
parser.add_argument("--force", action="store_true", help="Force USD regeneration.")
parser.add_argument(
    "--side",
    action="append",
    choices=("left", "right"),
    help="Which side to build. Repeat to build multiple sides. Defaults to both.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverter
from pxr import Sdf, Usd, UsdPhysics

from isaaclab_rh56dfx.robots import RH56DFX_LEFT_CFG, RH56DFX_RIGHT_CFG


def _selected_sides() -> list[str]:
    return args_cli.side if args_cli.side else ["left", "right"]


def _spawn_cfg_for_side(side: str):
    if side == "left":
        return RH56DFX_LEFT_CFG.spawn
    if side == "right":
        return RH56DFX_RIGHT_CFG.spawn
    raise ValueError(f"Unsupported side: {side}")


def _collect_collision_filter_pairs(urdf_path: str, side: str) -> list[tuple[str, str]]:
    root = ET.parse(urdf_path).getroot()
    links_with_collision = {
        link.attrib["name"] for link in root.findall("link") if link.find("collision") is not None
    }

    pairs: set[tuple[str, str]] = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.attrib["link"]
        child_name = child.attrib["link"]
        if parent_name in links_with_collision and child_name in links_with_collision:
            pairs.add(tuple(sorted((parent_name, child_name))))

    # The wrist stack still has a verified zero-pose overlap after importer convex-hull simplification.
    pairs.add(tuple(sorted((f"{side}_wrist_base_link", f"{side}_hand_base"))))

    # Proximal knuckle links are tightly packed around the palm and each other.
    # Filtering these preserves distal fingertip contacts while removing unstable internal base contacts.
    palm = f"{side}_palm"
    proximal_links = [
        f"{side}_thumb_1",
        f"{side}_index_1",
        f"{side}_middle_1",
        f"{side}_ring_1",
        f"{side}_little_1",
    ]
    for link_name in proximal_links:
        pairs.add(tuple(sorted((palm, link_name))))
    for left_name, right_name in zip(proximal_links, proximal_links[1:], strict=False):
        pairs.add(tuple(sorted((left_name, right_name))))

    return sorted(pairs)


def _apply_collision_filters(usd_path: str | Path, urdf_path: str, side: str) -> None:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    root_prim = stage.GetDefaultPrim()
    if not root_prim.IsValid():
        raise RuntimeError(f"USD stage has no default prim: {usd_path}")

    root_path = root_prim.GetPath()
    pairs = _collect_collision_filter_pairs(urdf_path, side)
    for link_a, link_b in pairs:
        prim_a = stage.GetPrimAtPath(root_path.AppendChild(link_a))
        prim_b = stage.GetPrimAtPath(root_path.AppendChild(link_b))
        if not prim_a.IsValid() or not prim_b.IsValid():
            raise RuntimeError(f"Missing rigid-body prim for collision filter pair: {link_a}, {link_b}")
        if not prim_a.HasAPI(UsdPhysics.RigidBodyAPI) or not prim_b.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Collision filter pair is not rigid-body-backed: {link_a}, {link_b}")

        rel_a = UsdPhysics.FilteredPairsAPI.Apply(prim_a).CreateFilteredPairsRel()
        rel_b = UsdPhysics.FilteredPairsAPI.Apply(prim_b).CreateFilteredPairsRel()
        rel_a.AddTarget(Sdf.Path(prim_b.GetPath()))
        rel_b.AddTarget(Sdf.Path(prim_a.GetPath()))

    stage.Save()


def main():
    sim_utils.create_new_stage()

    for side in _selected_sides():
        cfg = copy.deepcopy(_spawn_cfg_for_side(side))
        cfg.force_usd_conversion = args_cli.force
        converter = UrdfConverter(cfg)
        _apply_collision_filters(converter.usd_path, cfg.asset_path, side)
        print(f"[OK] Built {side} USD: {converter.usd_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
