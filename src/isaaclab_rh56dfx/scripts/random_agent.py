# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run random joint commands on an RH56DFX Isaac Lab play environment."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Random agent for RH56DFX Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the registered RH56DFX task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.envs import ManagerBasedEnv
import isaaclab_rh56dfx.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = ManagerBasedEnv(cfg=env_cfg)

    print(f"[INFO]: Observation groups: {list(env.observation_manager.active_terms.keys())}")
    print(f"[INFO]: Action shape: {tuple(env.action_manager.action.shape)}")

    env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = 2 * torch.rand_like(env.action_manager.action) - 1
            env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
