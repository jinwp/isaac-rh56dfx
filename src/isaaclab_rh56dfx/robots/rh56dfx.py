from __future__ import annotations

from pathlib import Path
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


RH56DFX_EXT_DIR = Path(__file__).resolve().parents[3]
RH56DFX_DATA_DIR = RH56DFX_EXT_DIR / "data"
RH56DFX_URDF_DIR = RH56DFX_DATA_DIR / "urdf"
RH56DFX_USD_DIR = RH56DFX_DATA_DIR / "usd"

Side = Literal["left", "right"]
MimicRule = tuple[str, str, float, float]


def actuated_joint_names(side: Side) -> list[str]:
    prefix = f"{side}_"
    return [
        f"{prefix}wrist_yaw_joint",
        f"{prefix}hand_base_joint",
        f"{prefix}thumb_1_joint",
        f"{prefix}thumb_2_joint",
        f"{prefix}index_1_joint",
        f"{prefix}middle_1_joint",
        f"{prefix}ring_1_joint",
        f"{prefix}little_1_joint",
    ]


def mimic_rules(side: Side) -> list[MimicRule]:
    prefix = f"{side}_"
    return [
        (f"{prefix}thumb_3_joint", f"{prefix}thumb_2_joint", 1.1425, 0.0),
        (f"{prefix}thumb_4_joint", f"{prefix}thumb_3_joint", 0.7508, 0.0),
        (f"{prefix}index_2_joint", f"{prefix}index_1_joint", 1.1169, -0.15),
        (f"{prefix}middle_2_joint", f"{prefix}middle_1_joint", 1.1169, -0.15),
        (f"{prefix}ring_2_joint", f"{prefix}ring_1_joint", 1.1169, -0.15),
        (f"{prefix}little_2_joint", f"{prefix}little_1_joint", 1.1169, -0.15),
    ]


RH56DFX_LEFT_URDF_PATH = str(RH56DFX_URDF_DIR / "rh56dfx_left.urdf")
RH56DFX_RIGHT_URDF_PATH = str(RH56DFX_URDF_DIR / "rh56dfx_right.urdf")


def _make_hand_cfg(side: Side, urdf_path: str, usd_subdir: str) -> ArticulationCfg:
    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            asset_path=urdf_path,
            usd_dir=str(RH56DFX_USD_DIR / usd_subdir),
            usd_file_name=f"rh56dfx_{side}.usd",
            force_usd_conversion=False,
            fix_base=True,
            merge_fixed_joints=False,
            convert_mimic_joints_to_normal_joints=False,
            make_instanceable=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=None,
                    damping=None,
                )
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.85),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "hand": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=20.0,
                damping=2.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


RH56DFX_LEFT_CFG = _make_hand_cfg("left", RH56DFX_LEFT_URDF_PATH, "left")
RH56DFX_RIGHT_CFG = _make_hand_cfg("right", RH56DFX_RIGHT_URDF_PATH, "right")
