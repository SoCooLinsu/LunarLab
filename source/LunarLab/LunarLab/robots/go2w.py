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
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            ".*_hip_joint": 0.0,       # -1 ~ 1 // scale 1.0
            ".*_thigh_joint": 0.96,    # -1.57 ~ 3.49 // scale 2.53
            ".*_calf_joint": -1.775,   # -2.72 ~ -0.83 // scale 0.945
            ".*_foot_joint": 0.0,      # 바퀴는 0
        },
    ),
    
    actuators={
        "legs1": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
            stiffness=20.0,
            damping=0.5,
            friction=0.01,
            effort_limit_sim=23.7,
            velocity_limit_sim=30.0,
        ),
        "legs2": ImplicitActuatorCfg(
            joint_names_expr=[".*_calf_joint"],
            stiffness=20.0,
            damping=0.5,
            friction=0.01,
            effort_limit_sim=35.5,
            velocity_limit_sim=30.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"], # 아까 URDF의 revolute 바퀴 조인트
            stiffness=0.0,
            damping=2.0,
            friction=0.01,
            effort_limit_sim=23.7,
            velocity_limit_sim=30.0,
        ),
    },
)
