"""Prebuild cached USD files for RH56DFX left/right assets."""

from __future__ import annotations

import argparse
import copy

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

from isaaclab_rh56dfx.robots import RH56DFX_LEFT_CFG, RH56DFX_RIGHT_CFG


def _selected_sides() -> list[str]:
    return args_cli.side if args_cli.side else ["left", "right"]


def _spawn_cfg_for_side(side: str):
    if side == "left":
        return RH56DFX_LEFT_CFG.spawn
    if side == "right":
        return RH56DFX_RIGHT_CFG.spawn
    raise ValueError(f"Unsupported side: {side}")


def main():
    sim_utils.create_new_stage()

    for side in _selected_sides():
        cfg = copy.deepcopy(_spawn_cfg_for_side(side))
        cfg.force_usd_conversion = args_cli.force
        converter = UrdfConverter(cfg)
        print(f"[OK] Built {side} USD: {converter.usd_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
