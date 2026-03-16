from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


logger = logging.getLogger(__name__)


class Rh56dfxMimicJointPositionAction(ActionTerm):
    """Map an 8-DoF RH56DFX action into full joint targets, including mimic joints."""

    _asset: Articulation

    def __init__(self, cfg: "Rh56dfxMimicJointPositionActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        self._actuated_joint_ids, self._actuated_joint_names = self._asset.find_joints(
            cfg.actuated_joint_names, preserve_order=cfg.preserve_order
        )
        self._num_actions = len(self._actuated_joint_ids)
        self._raw_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._full_targets = torch.zeros((self.num_envs, self._asset.num_joints), device=self.device)

        joint_names = self._asset.data.joint_names
        name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
        self._mimic_pairs: list[tuple[int, int, float, float]] = []
        for child_name, parent_name, multiplier, offset in cfg.mimic_rules:
            if child_name not in name_to_idx or parent_name not in name_to_idx:
                raise ValueError(
                    f"Could not resolve mimic pair ({child_name}, {parent_name}) for articulation joints: {joint_names}"
                )
            self._mimic_pairs.append((name_to_idx[child_name], name_to_idx[parent_name], multiplier, offset))

        logger.info(
            "Resolved RH56DFX actuated joints for %s: %s [%s]",
            self.__class__.__name__,
            self._actuated_joint_names,
            self._actuated_joint_ids,
        )

        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones((self.num_envs, self.action_dim), device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                cfg.scale, self._actuated_joint_names, preserve_order=cfg.preserve_order
            )
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}")

        self._clip = None
        if cfg.clip is not None:
            if not isinstance(cfg.clip, dict):
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}")
            self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                self.num_envs, self.action_dim, 1
            )
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                cfg.clip, self._actuated_joint_names, preserve_order=cfg.preserve_order
            )
            self._clip[:, index_list] = torch.tensor(value_list, device=self.device)

    @property
    def action_dim(self) -> int:
        return self._num_actions

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale
        if self._clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        if self.cfg.rescale_to_limits:
            clamped_actions = self._processed_actions.clamp(-1.0, 1.0)
            self._processed_actions[:] = math_utils.unscale_transform(
                clamped_actions,
                self._asset.data.soft_joint_pos_limits[:, self._actuated_joint_ids, 0],
                self._asset.data.soft_joint_pos_limits[:, self._actuated_joint_ids, 1],
            )

    def apply_actions(self):
        self._full_targets[:] = self._asset.data.default_joint_pos
        self._full_targets[:, self._actuated_joint_ids] = self._processed_actions
        for child_idx, parent_idx, multiplier, offset in self._mimic_pairs:
            self._full_targets[:, child_idx] = multiplier * self._full_targets[:, parent_idx] + offset
        self._asset.set_joint_position_target(self._full_targets)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0


@configclass
class Rh56dfxMimicJointPositionActionCfg(ActionTermCfg):
    """Configuration for the RH56DFX mimic-aware position action."""

    class_type: type[ActionTerm] = Rh56dfxMimicJointPositionAction

    actuated_joint_names: list[str] = MISSING
    mimic_rules: list[tuple[str, str, float, float]] = MISSING
    scale: float | dict[str, float] = 1.0
    rescale_to_limits: bool = True
    preserve_order: bool = True
