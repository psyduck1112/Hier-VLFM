# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
# Modifications Copyright (c) 2026 Yikang.

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

# VLFM项目内部模块
from vlfm.policy.base_objectnav_policy import VLFMConfig
from vlfm.policy.itm_policy import ITMPolicyV2


class JetRacerROSMixin:
    """
    JetRacer算法混入类
    只包含VLFM导航算法逻辑，不包含任何ROS通信代码
    所有ROS通信（传感器订阅、指令发布）由外部控制器VLFMJetRacerController负责
    对应Habitat架构中的Policy角色
    """

    # 策略配置
    _stop_action: Tensor = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    _load_yolo: bool = True
    _non_coco_caption: str = (
        "chair . table . tv . laptop . microwave . toaster . sink . refrigerator . book"
        " . clock . vase . scissors . teddy bear . hair drier . toothbrush ."
    )

    # 观测数据缓存
    _observations_cache: Dict[str, Any] = {}
    _policy_info: Dict[str, Any] = {}
    _done_initializing: bool = False
    _initial_yaws: List[float] = []
    _navigate_forward_done: bool = False

    def __init__(self: Union["JetRacerROSMixin", ITMPolicyV2], *args: Any, **kwargs: Any) -> None:
        """初始化JetRacer算法混入，不进行任何ROS初始化"""
        # 兼容性处理：忽略旧版本的_skip_ros_init参数
        kwargs.pop("_skip_ros_init", None)
        kwargs["sync_explored_areas"] = True
        super().__init__(*args, **kwargs)
        # 配置对象地图使用真实深度数据
        self._object_map.use_dbscan = True

    @classmethod
    def from_config(cls, config: DictConfig, *args_unused: Any, **kwargs_unused: Any) -> Any:
        """从配置文件创建策略实例"""
        policy_config: VLFMConfig = config.policy
        kwargs = {k: policy_config[k] for k in VLFMConfig.kwaarg_names}
        instance = cls(**kwargs)
        print("✅ JetRacer策略初始化完成，支持连续控制")
        return instance

    def act(
        self: Union["JetRacerROSMixin", ITMPolicyV2],
        observations: Dict[str, Any],
        rnn_hidden_states: Union[Tensor, Any],
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """
        执行导航算法，返回速度控制字典
        不发布ROS指令，由控制器负责将返回值发布到/cmd_vel
        """
        # 更新目标对象描述
        if observations["objectgoal"] not in self._non_coco_caption:
            self._non_coco_caption = observations["objectgoal"] + " . " + self._non_coco_caption

        # 调用父类策略获取动作
        parent_cls: ITMPolicyV2 = super()
        action: Tensor = parent_cls.act(observations, rnn_hidden_states, prev_actions, masks, deterministic)[0]
        mode = self._policy_info.get("mode", "")

        # 判断动作格式：初始化阶段的连续动作 vs 导航阶段的离散动作
        if (
            action.shape[-1] == 2
            and action.numel() >= 2
            and not (action[0][0].item() in [0, 1, 2, 3] and action[0][1].item() in [0, 1, 2, 3])
        ):
            # 初始化阶段的连续动作格式：[angular, linear]
            angular_vel = float(action[0][0].item())
            linear_vel = float(action[0][1].item())
        else:
            # 导航阶段的离散动作：映射为连续控制
            from vlfm.policy.utils.pointnav_policy import discrete_action_to_continuous

            discrete_action = action.flatten()[0] if action.numel() >= 2 else action
            linear_vel, angular_vel = discrete_action_to_continuous(discrete_action)

        # JetRacer定制逻辑：进入navigate后仅执行一步前进，随后停止并退出导航
        if mode == "navigate":
            from vlfm.policy.utils.pointnav_policy import discrete_action_to_continuous

            if not self._navigate_forward_done:
                linear_vel, angular_vel = discrete_action_to_continuous(1)  # FORWARD
                self._navigate_forward_done = True
                self._called_stop = True
            else:
                linear_vel, angular_vel = 0.0, 0.0
                self._called_stop = True
                self._policy_info["stop_reason"] = "target_reached"
        else:
            self._navigate_forward_done = False

        # 返回动作字典，由控制器负责发布ROS指令
        action_dict = {
            "linear": linear_vel,
            "angular": angular_vel,
            "info": self._policy_info,
        }

        if "rho_theta" in self._policy_info:
            action_dict["rho_theta"] = self._policy_info["rho_theta"]

        return action_dict

    def get_action(self, observations: Dict[str, Any], masks: Tensor, deterministic: bool = True) -> Dict[str, Any]:
        """获取动作的简化接口"""
        return self.act(observations, None, None, masks, deterministic=deterministic)

    def _reset(self: Union["JetRacerROSMixin", ITMPolicyV2]) -> None:
        """重置策略状态"""
        parent_cls: ITMPolicyV2 = super()
        parent_cls._reset()
        self._navigate_forward_done = False

    def _initialize(self) -> torch.Tensor:
        """JetRacer初始化动作 - 原地转圈观察环境"""
        if not hasattr(self, "_init_steps"):
            self._init_steps = 0

        self._init_steps += 1

        # 原地转圈：分6步完成约135度
        if self._init_steps <= 6:
            print(f"🚗 JetRacer初始化步骤 {self._init_steps}/6: 原地左转 ({self._init_steps * 22.5:.1f}°)")
            return torch.tensor([[0.8, 0.4]], dtype=torch.float32)  # [angular, linear]
        else:
            if not self._done_initializing:
                print("✅ JetRacer初始化完成，切换到VLFM策略控制")
                self._done_initializing = True
            return self._stop_action

    def _cache_observations(self: Union["JetRacerROSMixin", ITMPolicyV2], observations: Dict[str, Any]) -> None:
        """
        缓存观测数据
        直接使用控制器传入的observations，不再读取内部传感器
        对应Habitat中HabitatMixin._cache_observations()的角色
        """
        self._observations_cache.clear()

        robot_xy = np.array(observations["robot_xy"])
        robot_heading = observations["robot_heading"]

        # 从控制器传入的depth数据生成nav_depth（供PointNav使用）
        if "depth" in observations:
            depth_tensor = observations["depth"]
            if isinstance(depth_tensor, torch.Tensor):
                depth_np = depth_tensor.squeeze().cpu().numpy()
            else:
                depth_np = np.array(depth_tensor)
            # 确保shape为[H, W]
            if len(depth_np.shape) > 2:
                depth_np = depth_np.squeeze()
            h, w = depth_np.shape
            nav_depth = torch.from_numpy(depth_np).reshape(1, h, w, 1)
            nav_depth = nav_depth.to("cuda" if torch.cuda.is_available() else "cpu")
        else:
            nav_depth = torch.zeros(1, 480, 640, 1)

        # 直接使用控制器已经处理好的数据
        self._observations_cache = {
            "frontier_sensor": observations.get("frontier_sensor", np.array([])),
            "nav_depth": nav_depth,
            "robot_xy": robot_xy,
            "robot_heading": robot_heading,
            "object_map_rgbd": observations.get("object_map_rgbd", []),
            "value_map_rgbd": observations.get("value_map_rgbd", []),
        }


@dataclass
class JetRacerROSConfig(DictConfig):
    """JetRacer配置类"""

    policy: VLFMConfig = VLFMConfig()

    # 运动限制
    max_linear_velocity: float = 0.3
    max_angular_velocity: float = 0.4

    # 相机参数
    camera_fx: float = 616.0
    camera_fy: float = 616.0
    camera_cx: float = 320.0
    camera_cy: float = 240.0
    camera_width: int = 640
    camera_height: int = 480
    depth_scale: float = 0.001


class JetRacerROSITMPolicyV2(JetRacerROSMixin, ITMPolicyV2):
    """
    JetRacer ITM策略V2
    纯算法策略，结合JetRacer初始化逻辑和VLFM ITM导航算法
    ROS通信完全由外部VLFMJetRacerController负责（对应Habitat的Environment角色）
    """

    pass
