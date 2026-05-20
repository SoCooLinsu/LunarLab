import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

GO2W_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/LunarLab/source/LunarLab/LunarLab/robots/go2w.usd",
        activate_contact_sensors=True,
    ),
    prim_path="{ENV_REGEX_NS}/Robot",
    

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),  # 바닥에 파묻히지 않게 공중에서 스폰
        joint_pos={
            ".*_hip_joint": 0.0,       # -1 ~ 1 // scale 1.0
            ".*_thigh_joint": 0.96,    # -1.57 ~ 3.49 // scale 2.53
            ".*_calf_joint": -1.775,   # -2.72 ~ -0.83 // scale 0.945
            ".*_foot_joint": 0.0,      # 바퀴는 0
        },
    ),
    
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            stiffness=50.0,
            damping=0.1,
            effort_limit_sim=30.0,
            velocity_limit_sim=30.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"], # 아까 URDF의 revolute 바퀴 조인트
            stiffness=0.0,
            damping=5.0,
            effort_limit_sim=30.0,
            velocity_limit_sim=30.0,
        ),
    },
)
