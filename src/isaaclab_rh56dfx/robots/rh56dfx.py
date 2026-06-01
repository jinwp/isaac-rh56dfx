from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import PhysxCfg

from isaaclab_rh56dfx.actuators import DelayedDCMotorCfg


RH56DFX_EXT_DIR = Path(__file__).resolve().parents[3]
RH56DFX_DATA_DIR = RH56DFX_EXT_DIR / "data"
RH56DFX_CONFIG_DIR = RH56DFX_DATA_DIR / "config"
RH56DFX_URDF_DIR = RH56DFX_DATA_DIR / "urdf"
RH56DFX_USD_DIR = RH56DFX_DATA_DIR / "usd"
RH56DFX_PHYSICS_CONFIG_PATH = RH56DFX_CONFIG_DIR / "rh56dfx_physics.toml"

Side = Literal["left", "right"]
MimicRule = tuple[str, str, float, float]

LOGICAL_DOF_ORDER = (
    "little",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rotation",
)

# Manual command-space angle ranges. These are interface-level references and
# are intentionally kept separate from the URDF's internal joint coordinates.
MANUAL_ANGLE_LIMITS_DEG = {
    "finger": (19.0, 176.7),
    "thumb_bend": (-13.0, 53.6),
    "thumb_rotation": (90.0, 165.0),
}


def _load_physics_cfg() -> dict[str, Any]:
    with RH56DFX_PHYSICS_CONFIG_PATH.open("rb") as config_file:
        return tomllib.load(config_file)


_PHYSICS_CFG = _load_physics_cfg()


def physics_dt() -> float:
    return float(_PHYSICS_CFG["sim"]["physics_dt"])


def control_decimation() -> int:
    return int(_PHYSICS_CFG["sim"]["action_decimation"])


def sim_physx_cfg() -> PhysxCfg:
    physx_cfg = _PHYSICS_CFG["sim"]["physx"]
    return PhysxCfg(
        enable_ccd=bool(physx_cfg["enable_ccd"]),
        bounce_threshold_velocity=float(physx_cfg["bounce_threshold_velocity"]),
        friction_offset_threshold=float(physx_cfg["friction_offset_threshold"]),
        friction_correlation_distance=float(physx_cfg["friction_correlation_distance"]),
        max_position_iteration_count=int(physx_cfg["max_position_iteration_count"]),
        max_velocity_iteration_count=int(physx_cfg["max_velocity_iteration_count"]),
    )


def sim_physics_material_cfg() -> sim_utils.RigidBodyMaterialCfg:
    material_cfg = _PHYSICS_CFG["materials"]
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=float(material_cfg["static_friction"]),
        dynamic_friction=float(material_cfg["dynamic_friction"]),
        restitution=float(material_cfg["restitution"]),
        friction_combine_mode=str(material_cfg["friction_combine_mode"]),
        restitution_combine_mode=str(material_cfg["restitution_combine_mode"]),
    )


def mount_joint_names(side: Side) -> list[str]:
    prefix = f"{side}_"
    return [
        f"{prefix}wrist_yaw_joint",
        f"{prefix}hand_base_joint",
    ]


def actuated_joint_names(side: Side) -> list[str]:
    return mount_joint_names(side) + logical_joint_names(side)


def logical_joint_names(side: Side) -> list[str]:
    prefix = f"{side}_"
    return [
        f"{prefix}little_1_joint",
        f"{prefix}ring_1_joint",
        f"{prefix}middle_1_joint",
        f"{prefix}index_1_joint",
        f"{prefix}thumb_2_joint",
        f"{prefix}thumb_1_joint",
    ]


def finger_joint_names(side: Side) -> list[str]:
    prefix = f"{side}_"
    return [
        f"{prefix}little_1_joint",
        f"{prefix}little_2_joint",
        f"{prefix}ring_1_joint",
        f"{prefix}ring_2_joint",
        f"{prefix}middle_1_joint",
        f"{prefix}middle_2_joint",
        f"{prefix}index_1_joint",
        f"{prefix}index_2_joint",
    ]


def thumb_bend_joint_names(side: Side) -> list[str]:
    prefix = f"{side}_"
    return [
        f"{prefix}thumb_2_joint",
        f"{prefix}thumb_3_joint",
        f"{prefix}thumb_4_joint",
    ]


def thumb_rotation_joint_names(side: Side) -> list[str]:
    prefix = f"{side}_"
    return [f"{prefix}thumb_1_joint"]


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


def logical_joint_limits(side: Side) -> dict[str, tuple[float, float]]:
    prefix = f"{side}_"
    # Keep the proximal commands inside the mimic-safe range so the generated
    # distal targets remain within the URDF joint bounds.
    finger_min = 0.15 / 1.1169
    thumb_bend_max = min(0.606844 / 1.1425, 0.430866 / (1.1425 * 0.7508))
    return {
        f"{prefix}little_1_joint": (finger_min, 1.344277),
        f"{prefix}ring_1_joint": (finger_min, 1.344277),
        f"{prefix}middle_1_joint": (finger_min, 1.344277),
        f"{prefix}index_1_joint": (finger_min, 1.344277),
        f"{prefix}thumb_2_joint": (0.0, thumb_bend_max),
        f"{prefix}thumb_1_joint": (0.0, 1.310384),
    }


def default_joint_positions(side: Side) -> dict[str, float]:
    prefix = f"{side}_"
    finger_open = logical_joint_limits(side)[f"{prefix}index_1_joint"][0]
    return {
        f"{prefix}wrist_yaw_joint": 0.0,
        f"{prefix}hand_base_joint": 0.0,
        f"{prefix}thumb_1_joint": 0.0,
        f"{prefix}thumb_2_joint": 0.0,
        f"{prefix}thumb_3_joint": 0.0,
        f"{prefix}thumb_4_joint": 0.0,
        f"{prefix}index_1_joint": finger_open,
        f"{prefix}index_2_joint": 0.0,
        f"{prefix}middle_1_joint": finger_open,
        f"{prefix}middle_2_joint": 0.0,
        f"{prefix}ring_1_joint": finger_open,
        f"{prefix}ring_2_joint": 0.0,
        f"{prefix}little_1_joint": finger_open,
        f"{prefix}little_2_joint": 0.0,
    }


def _delay_range_steps() -> tuple[int, int]:
    actuation_cfg = _PHYSICS_CFG["actuation"]
    dt = physics_dt()
    nominal_delay_steps = int(round(float(actuation_cfg["latency_s"]) / dt))
    jitter_steps = int(round(float(actuation_cfg["latency_jitter_s"]) / dt))
    return max(nominal_delay_steps - jitter_steps, 0), nominal_delay_steps + jitter_steps


def _mount_actuator_cfg(side: Side) -> ImplicitActuatorCfg:
    cfg = _PHYSICS_CFG["actuator"]["mount"]
    return ImplicitActuatorCfg(
        joint_names_expr=mount_joint_names(side),
        stiffness=float(cfg["stiffness"]),
        damping=float(cfg["damping"]),
        effort_limit_sim=float(cfg["effort_limit_sim"]),
        velocity_limit_sim=float(cfg["velocity_limit_sim"]),
        armature=float(cfg["armature"]),
        friction=float(cfg["friction"]),
        dynamic_friction=float(cfg["dynamic_friction"]),
        viscous_friction=float(cfg["viscous_friction"]),
    )


def _hand_motor_cfg(
    cfg: dict[str, Any],
    joint_names: list[str],
    min_delay: int,
    max_delay: int,
) -> DelayedDCMotorCfg:
    return DelayedDCMotorCfg(
        joint_names_expr=joint_names,
        min_delay=min_delay,
        max_delay=max_delay,
        stiffness=float(cfg["stiffness"]),
        damping=float(cfg["damping"]),
        effort_limit=float(cfg["effort_limit"]),
        effort_limit_sim=float(cfg["effort_limit_sim"]),
        saturation_effort=float(cfg["saturation_effort"]),
        velocity_limit=float(cfg["velocity_limit"]),
        velocity_limit_sim=float(cfg["velocity_limit_sim"]),
        armature=float(cfg["armature"]),
        friction=float(cfg["friction"]),
        dynamic_friction=float(cfg["dynamic_friction"]),
        viscous_friction=float(cfg["viscous_friction"]),
    )


RH56DFX_LEFT_URDF_PATH = str(RH56DFX_URDF_DIR / "rh56dfx_left.urdf")
RH56DFX_RIGHT_URDF_PATH = str(RH56DFX_URDF_DIR / "rh56dfx_right.urdf")


def _make_hand_cfg(side: Side, urdf_path: str, usd_subdir: str) -> ArticulationCfg:
    articulation_cfg = _PHYSICS_CFG["articulation"]
    collision_cfg = _PHYSICS_CFG["collision"]
    min_delay, max_delay = _delay_range_steps()
    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            asset_path=urdf_path,
            usd_dir=str(RH56DFX_USD_DIR / usd_subdir),
            usd_file_name=f"rh56dfx_{side}.usd",
            force_usd_conversion=False,
            copy_from_source=False,
            fix_base=True,
            merge_fixed_joints=False,
            convert_mimic_joints_to_normal_joints=False,
            make_instanceable=False,
            collider_type=str(articulation_cfg["collider_type"]),
            self_collision=bool(articulation_cfg["import_self_collision"]),
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=bool(articulation_cfg["enable_self_collisions"]),
                solver_position_iteration_count=int(articulation_cfg["solver_position_iteration_count"]),
                solver_velocity_iteration_count=int(articulation_cfg["solver_velocity_iteration_count"]),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                enable_gyroscopic_forces=bool(articulation_cfg["enable_gyroscopic_forces"]),
                solver_position_iteration_count=int(articulation_cfg["solver_position_iteration_count"]),
                solver_velocity_iteration_count=int(articulation_cfg["solver_velocity_iteration_count"]),
                max_depenetration_velocity=float(articulation_cfg["max_depenetration_velocity"]),
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=float(collision_cfg["contact_offset"]),
                rest_offset=float(collision_cfg["rest_offset"]),
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0,
                    damping=0.0,
                )
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.85),
            joint_pos=default_joint_positions(side),
            joint_vel={".*": 0.0},
        ),
        actuators={
            "mount": _mount_actuator_cfg(side),
            "thumb_rotation": _hand_motor_cfg(
                _PHYSICS_CFG["actuator"]["thumb_rotation"],
                thumb_rotation_joint_names(side),
                min_delay=min_delay,
                max_delay=max_delay,
            ),
            "thumb_bend": _hand_motor_cfg(
                _PHYSICS_CFG["actuator"]["thumb_bend"],
                thumb_bend_joint_names(side),
                min_delay=min_delay,
                max_delay=max_delay,
            ),
            "fingers": _hand_motor_cfg(
                _PHYSICS_CFG["actuator"]["fingers"],
                finger_joint_names(side),
                min_delay=min_delay,
                max_delay=max_delay,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


RH56DFX_LEFT_CFG = _make_hand_cfg("left", RH56DFX_LEFT_URDF_PATH, "left")
RH56DFX_RIGHT_CFG = _make_hand_cfg("right", RH56DFX_RIGHT_URDF_PATH, "right")
