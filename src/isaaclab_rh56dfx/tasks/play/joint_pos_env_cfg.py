from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_rh56dfx.mdp.actions import Rh56dfxMimicJointPositionActionCfg
from isaaclab_rh56dfx.robots import (
    RH56DFX_LEFT_CFG,
    RH56DFX_RIGHT_CFG,
    logical_joint_limits,
    logical_joint_names,
    mimic_rules,
)
from isaaclab_rh56dfx.tasks.play.base_env_cfg import RH56DFXPlayEnvCfg

_HAND_SPAWN_HEIGHT = 0.3


@configclass
class RH56DFXLeftPlayEnvCfg(RH56DFXPlayEnvCfg):
    """Vectorized left-hand play environment with random or manual joint commands."""

    def __post_init__(self):
        super().__post_init__()

        left_logical = logical_joint_names("left")
        self.scene.robot = RH56DFX_LEFT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, _HAND_SPAWN_HEIGHT)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.actions.joint_pos = Rh56dfxMimicJointPositionActionCfg(
            asset_name="robot",
            actuated_joint_names=left_logical,
            mimic_rules=mimic_rules("left"),
            joint_limits=logical_joint_limits("left"),
            scale=1.0,
            rescale_to_limits=True,
        )
        self.observations.policy.enable_corruption = False
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=left_logical, preserve_order=True
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=left_logical, preserve_order=True
        )


@configclass
class RH56DFXRightPlayEnvCfg(RH56DFXPlayEnvCfg):
    """Vectorized right-hand play environment with random or manual joint commands."""

    def __post_init__(self):
        super().__post_init__()

        right_logical = logical_joint_names("right")
        self.scene.robot = RH56DFX_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, _HAND_SPAWN_HEIGHT)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.actions.joint_pos = Rh56dfxMimicJointPositionActionCfg(
            asset_name="robot",
            actuated_joint_names=right_logical,
            mimic_rules=mimic_rules("right"),
            joint_limits=logical_joint_limits("right"),
            scale=1.0,
            rescale_to_limits=True,
        )
        self.observations.policy.enable_corruption = False
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=right_logical, preserve_order=True
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=right_logical, preserve_order=True
        )
