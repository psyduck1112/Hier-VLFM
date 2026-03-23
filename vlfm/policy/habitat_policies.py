# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any, Dict, Union

import numpy as np
import torch
from depth_camera_filtering import filter_depth
from frontier_exploration.base_explorer import BaseExplorer
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.config.default_structured_configs import (
    PolicyConfig,
)
from habitat_baselines.rl.ppo.policy import PolicyActionData
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from torch import Tensor

from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix
from vlfm.vlm.grounding_dino import ObjectDetections

from ..mapping.obstacle_map import ObstacleMap
from .base_objectnav_policy import BaseObjectNavPolicy, VLFMConfig
from .itm_policy import ITMPolicy, ITMPolicyV2, ITMPolicyV3

'''
def debug_tensor_info(name: str, tensor: torch.Tensor) -> None:
    """调试辅助函数：打印张量信息"""
    if tensor is not None and hasattr(tensor, 'shape'):
        print(f"[DEBUG] {name}: shape={tensor.shape}, dtype={tensor.dtype}, "
              f"device={tensor.device}, range=[{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
    else:
        print(f"[DEBUG] {name}: {type(tensor)}")


def debug_observations_detail(observations: TensorDict, step_count: int = 0) -> None:
    """详细调试观测数据的辅助函数"""
    print(f"\n[DEBUG] ===== Step {step_count} Observations Debug =====")
    for key, value in observations.items():
        if hasattr(value, 'shape') and len(value.shape) > 0:
            if key in ['rgb', 'depth']:
                # 对于图像数据，显示更多细节
                batch_size = value.shape[0] if len(value.shape) > 3 else 1
                print(f"[DEBUG] {key}: shape={value.shape}, dtype={value.dtype}, "
                      f"batch_size={batch_size}")
                if len(value.shape) >= 3:
                    sample = value[0] if batch_size > 0 else value
                    print(f"         sample range: [{sample.min().item():.3f}, {sample.max().item():.3f}]")
            elif key in ['gps', 'compass']:
                # 对于传感器数据
                if len(value.shape) > 1:
                    print(f"[DEBUG] {key}: shape={value.shape}, value={value[0].cpu().numpy()}")
                else:
                    print(f"[DEBUG] {key}: shape={value.shape}, value={value.cpu().numpy()}")
            else:
                debug_tensor_info(key, value)
        else:
            print(f"[DEBUG] {key}: {type(value)} - {value}")
    print("[DEBUG] =======================================\n")
'''
    

# HM3D数据集中对象ID到名称的映射
HM3D_ID_TO_NAME = ["chair", "bed", "potted plant", "toilet", "tv", "couch"]
# MP3D数据集中对象ID到名称的映射
MP3D_ID_TO_NAME = [
    "chair",
    "table|dining table|coffee table|side table|desk",  # "table",
    "framed photograph",  # "picture",
    "cabinet",
    "pillow",  # "cushion",
    "couch",  # "sofa",
    "bed",
    "nightstand",  # "chest of drawers",
    "potted plant",  # "plant",
    "sink",
    "toilet",
    "stool",
    "towel",
    "tv",  # "tv monitor",
    "shower",
    "bathtub",
    "counter",
    "fireplace",
    "gym equipment",
    "seating",
    "clothes",
]


class TorchActionIDs:
    """定义动作ID的张量表示"""
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)


class HabitatMixin:
    """
    这个Python mixin只包含在Habitat环境中显式运行BaseObjectNavPolicy的相关代码，
    它将为任何父类（BaseObjectNavPolicy的子类）提供在Habitat中运行所需的方法。
    This Python mixin only contains code relevant for running a BaseObjectNavPolicy
    explicitly within Habitat (vs. the real world, etc.) and will endow any parent class
    (that is a subclass of BaseObjectNavPolicy) with the necessary methods to run in
    Habitat.
    """
    _stop_action: Tensor = TorchActionIDs.STOP
    _start_yaw: Union[float, None] = None  # must be set by _reset() method
    _observations_cache: Dict[str, Any] = {}
    _policy_info: Dict[str, Any] = {}
    _compute_frontiers: bool = False

    def __init__(
        self,
        camera_height: float,
        min_depth: float,
        max_depth: float,
        camera_fov: float,
        image_width: int,
        dataset_type: str = "hm3d",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        初始化HabitatMixin
        
        Args:
            camera_height (float): 相机高度
            min_depth (float): 深度图像的最小深度值
            max_depth (float): 深度图像的最大深度值
            camera_fov (float): 相机视场角（角度）
            image_width (int): 图像宽度
            dataset_type (str): 数据集类型，默认为"hm3d"
            *args (Any): 其他位置参数
            **kwargs (Any): 其他关键字参数
        """
        super().__init__(*args, **kwargs)
        self._camera_height = camera_height
        self._min_depth = min_depth
        self._max_depth = max_depth
        camera_fov_rad = np.deg2rad(camera_fov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = image_width / (2 * np.tan(camera_fov_rad / 2))
        self._dataset_type = dataset_type

    @classmethod
    def from_config(cls, config: DictConfig, *args_unused: Any, **kwargs_unused: Any) -> "HabitatMixin":
        """
        从配置创建HabitatMixin实例
        
        Args:
            config (DictConfig): 配置对象
            *args_unused (Any): 未使用的参数
            **kwargs_unused (Any): 未使用的关键字参数
            
        Returns:
            HabitatMixin: 创建的HabitatMixin实例
        """
        policy_config: VLFMPolicyConfig = config.habitat_baselines.rl.policy
        kwargs = {k: policy_config[k] for k in VLFMPolicyConfig.kwaarg_names}  # type: ignore

        # In habitat, we need the height of the camera to generate the camera transform
        sim_sensors_cfg = config.habitat.simulator.agents.main_agent.sim_sensors
        kwargs["camera_height"] = sim_sensors_cfg.rgb_sensor.position[1]

        # Synchronize the mapping min/max depth values with the habitat config
        kwargs["min_depth"] = sim_sensors_cfg.depth_sensor.min_depth
        kwargs["max_depth"] = sim_sensors_cfg.depth_sensor.max_depth
        kwargs["camera_fov"] = sim_sensors_cfg.depth_sensor.hfov
        kwargs["image_width"] = sim_sensors_cfg.depth_sensor.width

        # Only bother visualizing if we're actually going to save the video
        kwargs["visualize"] = len(config.habitat_baselines.eval.video_option) > 0

        if "hm3d" in config.habitat.dataset.data_path:
            kwargs["dataset_type"] = "hm3d"
        elif "mp3d" in config.habitat.dataset.data_path:
            kwargs["dataset_type"] = "mp3d"
        else:
            raise ValueError("Dataset type could not be inferred from habitat config")

        return cls(**kwargs)

    def act(
        self: Union["HabitatMixin", BaseObjectNavPolicy], # 向前引用，在定义HabitatMixin的内部方法时，类本身还没有完全定义完成
        observations: TensorDict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> PolicyActionData:
        """
        执行动作
        
        Args:
            self (Union["HabitatMixin", BaseObjectNavPolicy]): 当前实例
            observations (TensorDict): 观测数据
            rnn_hidden_states (Any): RNN隐藏状态
            prev_actions (Any): 上一个动作
            masks (Tensor): 掩码
            deterministic (bool): 是否确定性执行，默认为False
            
        Returns:
            PolicyActionData: 动作数据
        """
        """Converts object ID to string name, returns action as PolicyActionData"""
        object_id: int = observations[ObjectGoalSensor.cls_uuid][0].item()
        obs_dict = observations.to_tree() # 转换成树状字典
        if self._dataset_type == "hm3d":
            obs_dict[ObjectGoalSensor.cls_uuid] = HM3D_ID_TO_NAME[object_id]
        elif self._dataset_type == "mp3d":
            obs_dict[ObjectGoalSensor.cls_uuid] = MP3D_ID_TO_NAME[object_id]
            self._non_coco_caption = " . ".join(MP3D_ID_TO_NAME).replace("|", " . ") + " ."
        else:
            raise ValueError(f"Dataset type {self._dataset_type} not recognized")
        parent_cls: BaseObjectNavPolicy = super() # 用于调用父类的方法。在多重继承的情况下，它会根据方法解析顺序（MRO）找到合适的父类 type: ignore
        try:
            action, rnn_hidden_states = parent_cls.act(obs_dict, rnn_hidden_states, prev_actions, masks, deterministic)
        except StopIteration:
            action = self._stop_action
        return PolicyActionData(
            actions=action,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=[self._policy_info],
        )

    def _initialize(self) -> Tensor:
        """
        初始化阶段动作：左转12次，获得360度视野
        
        Returns:
            Tensor: 动作张量
        """
        """Turn left 30 degrees 12 times to get a 360 view at the beginning"""
        self._done_initializing = not self._num_steps < 11  # type: ignore
        return TorchActionIDs.TURN_LEFT

    def _reset(self) -> None:
        """
        重置环境状态
        """
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        parent_cls._reset()
        self._start_yaw = None

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        """
        获取策略信息用于日志记录
        
        Args:
            detections (ObjectDetections): 对象检测结果
            
        Returns:
            Dict[str, Any]: 策略信息字典
        """
        """Get policy info for logging"""
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        info = parent_cls._get_policy_info(detections)

        if not self._visualize:  # type: ignore
            return info

        if self._start_yaw is None:
            self._start_yaw = self._observations_cache["habitat_start_yaw"]
        info["start_yaw"] = self._start_yaw
        return info

    def _cache_observations(self: Union["HabitatMixin", BaseObjectNavPolicy], observations: TensorDict) -> None:
        """
        缓存观测数据，包括RGB图像、深度图像和相机变换
        用于pre_step
        Args:
           observations (TensorDict): 当前时间步的观测数据
        """
        """Caches the rgb, depth, and camera transform from the observations.

        Args:
           observations (TensorDict): The observations from the current timestep.
        """
        # 断点1: 详细观察输入的observations结构
        #debug_observations_detail(observations, getattr(self, '_debug_step_count', 0))
        # breakpoint()  # 取消注释以进入交互式调试器

        # 更新步数计数器
        if not hasattr(self, '_debug_step_count'):
            self._debug_step_count = 0
        self._debug_step_count += 1

        if len(self._observations_cache) > 0:
            return

        # 断点2: 观察原始传感器数据
        # 传感器数据 将GPU张量转到CPU并转为numpy数组
        # observations["rgb"].shape = [batch_size, height, width, channels]
        rgb = observations["rgb"][0].cpu().numpy() # 选择batch索引为0的传感器数据
        depth = observations["depth"][0].cpu().numpy()
        x, y = observations["gps"][0].cpu().numpy()
        camera_yaw = observations["compass"][0].cpu().item()

        #print(f"[DEBUG] RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
        #print(f"[DEBUG] Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")
        #print(f"[DEBUG] GPS position: ({x:.3f}, {y:.3f})")
        #print(f"[DEBUG] Camera yaw: {camera_yaw:.3f} radians")
        # breakpoint()  # 取消注释以进入交互式调试器

        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)

        # 断点3: 观察处理后的深度数据
        #print(f"[DEBUG] Filtered depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")

        # Habitat GPS makes west negative, so flip y  Habitat坐标系中西方向为负，需要翻转
        camera_position = np.array([x, -y, self._camera_height])
        robot_xy = camera_position[:2]
        tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

        # 断点4: 观察坐标变换
        #print(f"[DEBUG] Camera position: {camera_position}")
        #print(f"[DEBUG] Robot XY: {robot_xy}")
        #print(f"[DEBUG] Transform matrix shape: {tf_camera_to_episodic.shape}")
        # breakpoint()  # 取消注释以进入交互式调试器

        self._obstacle_map: ObstacleMap
        if self._compute_frontiers:
            print(f"[DEBUG] Computing frontiers using obstacle map")
            self._obstacle_map.update_map( # 更新障碍物地图
                depth,
                tf_camera_to_episodic,
                self._min_depth,
                self._max_depth,
                self._fx,
                self._fy,
                self._camera_fov,
            )
            frontiers = self._obstacle_map.frontiers
            #print(f"[DEBUG] Computed frontiers shape: {frontiers.shape}")
            #######################
            # 直接打印掩码
            step_count = getattr(self, '_debug_step_count', 0)
            #print(f"[Step {step_count}] Navigable Map:")
            #print(self._obstacle_map._navigable_map)
            #print(f"[Step {step_count}] Explored Area:")
            #print(self._obstacle_map.explored_area)
            # 记录智能体的移动轨迹
            self._obstacle_map.update_agent_traj(robot_xy, camera_yaw)
        else: # 如果不计算前沿，尝试从传感器直接获取或设为空数组
            #print(f"[DEBUG] Using frontier sensor from observations")
            if "frontier_sensor" in observations:
                frontiers = observations["frontier_sensor"][0].cpu().numpy()
                # print(f"[DEBUG] Frontier sensor shape: {frontiers.shape}")
            else:
                frontiers = np.array([])
                #print(f"[DEBUG] No frontier sensor, using empty array")

        # 断点5: 观察最终缓存的数据
       # print(f"[DEBUG] Final frontiers shape: {frontiers.shape}")
        # breakpoint()  # 取消注释以进入交互式调试器

        self._observations_cache = {
            "frontier_sensor": frontiers,
            "nav_depth": observations["depth"],  # for pointnav
            "robot_xy": robot_xy,
            "robot_heading": camera_yaw,
            "object_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._fx,
                    self._fy,
                )
            ],
            "value_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._camera_fov,
                )
            ],
            "habitat_start_yaw": observations["heading"][0].item(),
        }


@baseline_registry.register_policy
class OracleFBEPolicy(HabitatMixin, BaseObjectNavPolicy):
    """
    Oracle FBE策略类，继承自HabitatMixin和BaseObjectNavPolicy
    """
    
    def _explore(self, observations: TensorDict) -> Tensor:
        """
        探索函数，获取探索动作
        
        Args:
            observations (TensorDict): 观测数据
            
        Returns:
            Tensor: 探索动作
        """
        explorer_key = [k for k in observations.keys() if k.endswith("_explorer")][0]
        pointnav_action = observations[explorer_key]
        return pointnav_action


@baseline_registry.register_policy
class SuperOracleFBEPolicy(HabitatMixin, BaseObjectNavPolicy):
    """
    Super Oracle FBE策略类，继承自HabitatMixin和BaseObjectNavPolicy
    """
    
    def act(
        self,
        observations: TensorDict,
        rnn_hidden_states: Any,  # can be anything because it is not used
        *args: Any,
        **kwargs: Any,
    ) -> PolicyActionData:
        """
        执行动作
        
        Args:
            observations (TensorDict): 观测数据
            rnn_hidden_states (Any): RNN隐藏状态（未使用）
            *args (Any): 其他位置参数
            **kwargs (Any): 其他关键字参数
            
        Returns:
            PolicyActionData: 动作数据
        """
        return PolicyActionData(
            actions=observations[BaseExplorer.cls_uuid],
            rnn_hidden_states=rnn_hidden_states,
            policy_info=[self._policy_info],
        )


@baseline_registry.register_policy
class HabitatITMPolicy(HabitatMixin, ITMPolicy):
    """
    Habitat ITM策略类，继承自HabitatMixin和ITMPolicy
    """
    pass


@baseline_registry.register_policy
class HabitatITMPolicyV2(HabitatMixin, ITMPolicyV2):
    """
    Habitat ITM策略V2类，继承自HabitatMixin和ITMPolicyV2
    """
    pass


@baseline_registry.register_policy
class HabitatITMPolicyV3(HabitatMixin, ITMPolicyV3):
    """
    Habitat ITM策略V3类，继承自HabitatMixin和ITMPolicyV3
    """
    pass




@dataclass 
class VLFMPolicyConfig(VLFMConfig, PolicyConfig): 
    """
    VLFM策略配置类，继承自VLFMConfig和PolicyConfig
    """
    pass


cs = ConfigStore.instance()
cs.store(group="habitat_baselines/rl/policy", name="vlfm_policy", node=VLFMPolicyConfig)