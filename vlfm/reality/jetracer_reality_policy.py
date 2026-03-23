# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

# JetRacer Reality Policy - 基于 reality_policies 为 JetRacer 专门设计

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

# VLFM项目内部模块
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.policy.base_objectnav_policy import VLFMConfig
from vlfm.policy.itm_policy import ITMPolicyV2


class JetRacerRealityMixin:
    """
    JetRacer实机运行混入类
    专为JetRacer小车设计的Reality策略组件，提供在真实世界中运行ITMPolicyV2所需的代码。
    与Spot的RealityMixin不同，这个版本针对地面机器人优化，去除了机械臂相关功能。
    """

    # 停止动作:线性速度和角速度都为0 (适用于地面机器人)
    _stop_action: Tensor = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    # 是否加载YOLO模型的标志(实机运行时设为False，由VLM服务器处理)
    _load_yolo: bool = False
    # 非COCO数据集对象的文本描述(用于GroundingDINO检测)
    _non_coco_caption: str = (
        "chair . table . tv . laptop . microwave . toaster . sink . refrigerator . book"
        " . clock . vase . scissors . teddy bear . hair drier . toothbrush ."
    )
    # 观测数据缓存字典
    _observations_cache: Dict[str, Any] = {}
    # 策略信息字典
    _policy_info: Dict[str, Any] = {}
    # 初始化完成标志 (JetRacer无需机械臂初始化，直接设为True)
    _done_initializing: bool = True

    def __init__(self: Union["JetRacerRealityMixin", ITMPolicyV2], *args: Any, **kwargs: Any) -> None:
        # 确保 sync_explored_areas 设为 True（JetRacer实机运行需要）
        kwargs.setdefault('sync_explored_areas', True)
        # 调用父类初始化方法，启用探索区域同步
        super().__init__(*args, **kwargs)  # type: ignore
        
        # 禁用对象地图的DBSCAN聚类(实机环境下不使用)
        self._object_map.use_dbscan = False  # type: ignore

    @classmethod
    def from_config(cls, config: DictConfig, *args_unused: Any, **kwargs_unused: Any) -> Any:
        """从配置文件创建策略实例的类方法"""
        # 提取策略配置部分
        policy_config: VLFMConfig = config.policy
        # 从配置中提取关键字参数
        kwargs = {k: policy_config[k] for k in VLFMConfig.kwaarg_names}  # type: ignore
        
        # 使用提取的参数创建类实例
        return cls(**kwargs)

    def act(
        self: Union["JetRacerRealityMixin", ITMPolicyV2],
        observations: Dict[str, Any],  # 观测数据字典
        rnn_hidden_states: Union[Tensor, Any],  # RNN隐藏状态
        prev_actions: Any,  # 前一步动作
        masks: Tensor,  # 掩码张量
        deterministic: bool = False,  # 是否确定性行为
    ) -> Dict[str, Any]:
        # 如果目标对象不在非COCO描述中，则将其添加到描述前面
        if observations["objectgoal"] not in self._non_coco_caption:
            self._non_coco_caption = observations["objectgoal"] + " . " + self._non_coco_caption
        
        # 获取父类引用并调用其act方法
        parent_cls: ITMPolicyV2 = super()  # type: ignore
        action: Tensor = parent_cls.act(observations, rnn_hidden_states, prev_actions, masks, deterministic)[0]

        # JetRacer策略输出是一个(1,2)的张量，第一个元素是线性速度，第二个元素是角速度。
        # 将其转换为包含"linear"和"angular"键的字典，以便传递给JetRacer机器人。
        # 与Spot不同，JetRacer无需机械臂控制，简化动作空间
        action_dict = {
            "linear": action[0][0].item(),   # 线性速度(米/秒)
            "angular": action[0][1].item(),  # 角速度(弧度/秒)
            "info": self._policy_info,       # 策略信息
        }

        # 如果策略信息中包含rho_theta(极坐标)，则添加到动作字典中
        if "rho_theta" in self._policy_info:
            action_dict["rho_theta"] = self._policy_info["rho_theta"]

        return action_dict

    def get_action(self, observations: Dict[str, Any], masks: Tensor, deterministic: bool = True) -> Dict[str, Any]:
        """获取动作的简化接口，使用默认的RNN状态和前一步动作"""
        return self.act(observations, None, None, masks, deterministic=deterministic)

    def _reset(self: Union["JetRacerRealityMixin", ITMPolicyV2]) -> None:
        """重置策略状态"""
        # 调用父类的重置方法
        parent_cls: ITMPolicyV2 = super()  # type: ignore
        parent_cls._reset()
        # JetRacer无机械臂初始化，直接设为完成
        self._done_initializing = True

    def _cache_observations(self: Union["JetRacerRealityMixin", ITMPolicyV2], observations: Dict[str, Any]) -> None:
        """缓存当前时间步的RGB、深度和相机变换等观测数据
        
        JetRacer版本：简化了缓存逻辑，专注于单相机数据处理

        Args:
           observations (Dict[str, Any]): 当前时间步的观测数据
        """
        # 如果观测缓存已有数据，则直接返回(避免重复缓存)
        if len(self._observations_cache) > 0:
            return

        # 获取障碍物地图对象
        self._obstacle_map: ObstacleMap
        
        # 处理JetRacer的障碍物地图深度数据
        # JetRacer使用单个RGB-D相机，简化数据处理流程
        for obs_map_data in observations["obstacle_map_depths"][:-1]:
            depth, tf, min_depth, max_depth, fx, fy, topdown_fov = obs_map_data
            # 更新障碍物地图，但不进行探索
            self._obstacle_map.update_map(
                depth,           # 深度图像
                tf,              # 变换矩阵
                min_depth,       # 最小深度
                max_depth,       # 最大深度
                fx,              # X方向焦距
                fy,              # Y方向焦距
                topdown_fov,     # 俯视视野
                explore=False,   # 不进行探索
            )

        # 处理最后一个障碍物地图深度数据
        _, tf, min_depth, max_depth, fx, fy, topdown_fov = observations["obstacle_map_depths"][-1]
        self._obstacle_map.update_map(
            None,                    # 不使用深度数据
            tf,                      # 变换矩阵
            min_depth,               # 最小深度
            max_depth,               # 最大深度
            fx,                      # X方向焦距
            fy,                      # Y方向焦距
            topdown_fov,             # 俯视视野
            explore=True,            # 进行探索
            update_obstacles=False,  # 不更新障碍物
        )

        # 更新智能体轨迹信息
        self._obstacle_map.update_agent_traj(observations["robot_xy"], observations["robot_heading"])
        # 获取前沿点
        frontiers = self._obstacle_map.frontiers

        # 处理导航深度数据
        height, width = observations["nav_depth"].shape
        nav_depth = torch.from_numpy(observations["nav_depth"])  # 转换为PyTorch张量
        nav_depth = nav_depth.reshape(1, height, width, 1).to("cuda")  # 重塑形状并移动到GPU

        # 缓存观测数据 - JetRacer简化版
        self._observations_cache = {
            "frontier_sensor": frontiers,                            # 前沿传感器数据
            "nav_depth": nav_depth,                                 # 导航深度(用于PointNav)
            "robot_xy": observations["robot_xy"],                   # 机器人XY坐标(2维numpy数组)
            "robot_heading": observations["robot_heading"],         # 机器人朝向(弧度浮点数)
            "object_map_rgbd": observations["object_map_rgbd"],     # 对象地图RGBD数据
            "value_map_rgbd": observations["value_map_rgbd"],       # 价值地图RGBD数据
        }



@dataclass
class JetRacerRealityConfig(DictConfig):
    """JetRacer实机运行配置类"""
    policy: VLFMConfig = VLFMConfig()  # VLFM策略配置


class JetRacerRealityITMPolicyV2(JetRacerRealityMixin, ITMPolicyV2):
    """JetRacer实机ITM策略版本2
    
    这个类通过多重继承组合了JetRacerRealityMixin(提供JetRacer实机运行能力)和ITMPolicyV2(提供ITM策略逻辑)。
    它是VLFM系统在JetRacer机器人上运行的具体策略实现。
    
    与Spot的RealityITMPolicyV2不同：
    - 去除了机械臂初始化逻辑
    - 简化了动作空间(只有linear和angular)
    - 优化了单相机数据处理流程
    - 针对地面机器人的运动特性调整
    """
    pass