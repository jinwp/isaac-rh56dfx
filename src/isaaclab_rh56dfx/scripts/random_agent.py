# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a scripted thumb-to-pinky finger sequence on an RH56DFX play environment."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Scripted finger-motion agent for RH56DFX Isaac Lab environments.")
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


_FINGER_SEQUENCE = (
    ("thumb", (2, 3)),
    ("index", (4,)),
    ("middle", (5,)),
    ("ring", (6,)),
    ("little", (7,)),
)
_SETTLE_STEPS = 20
_OPEN_STEPS = 30
_CLOSE_STEPS = 30
_STEPS_PER_FINGER = _OPEN_STEPS + _CLOSE_STEPS
_TOTAL_SEQUENCE_STEPS = _SETTLE_STEPS + len(_FINGER_SEQUENCE) * _STEPS_PER_FINGER


def _interp(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return end
    alpha = step / float(total_steps - 1)
    return (1.0 - alpha) * start + alpha * end


def _scripted_actions(action_template: torch.Tensor, step_count: int) -> torch.Tensor:
    """Return normalized actions for a thumb-to-pinky open/close sequence.

    Action order is:
    0 wrist_yaw, 1 hand_base, 2 thumb_1, 3 thumb_2, 4 index_1, 5 middle_1, 6 ring_1, 7 little_1.
    Wrist/base stay neutral while fingers move. Finger actions use +1 for closed and -1 for open.
    """

    actions = torch.zeros_like(action_template)
    # Keep all fingers closed by default, then open/close one finger group at a time.
    actions[:, 2:] = 1.0

    cycle_step = step_count % _TOTAL_SEQUENCE_STEPS
    if cycle_step < _SETTLE_STEPS:
        return actions

    cycle_step -= _SETTLE_STEPS
    finger_idx = cycle_step // _STEPS_PER_FINGER
    phase_step = cycle_step % _STEPS_PER_FINGER
    _, joint_ids = _FINGER_SEQUENCE[finger_idx]

    if phase_step < _OPEN_STEPS:
        target = _interp(1.0, -1.0, phase_step, _OPEN_STEPS)
    else:
        target = _interp(-1.0, 1.0, phase_step - _OPEN_STEPS, _CLOSE_STEPS)

    actions[:, list(joint_ids)] = target
    return actions


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
    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = _scripted_actions(env.action_manager.action, step_count)
            env.step(actions)
            step_count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
