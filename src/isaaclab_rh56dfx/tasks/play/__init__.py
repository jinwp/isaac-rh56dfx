from __future__ import annotations

import gymnasium as gym


def _unsupported_gym_make(**_kwargs):
    raise RuntimeError(
        "RH56DFX play tasks expose ManagerBasedEnv configs for direct instantiation. "
        "Use parse_env_cfg(...) and then ManagerBasedEnv(cfg=env_cfg) instead of gym.make(...)."
    )


gym.register(
    id="Isaac-RH56DFX-Left-Play-v0",
    entry_point=f"{__name__}:_unsupported_gym_make",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:RH56DFXLeftPlayEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-RH56DFX-Right-Play-v0",
    entry_point=f"{__name__}:_unsupported_gym_make",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:RH56DFXRightPlayEnvCfg",
    },
    disable_env_checker=True,
)
