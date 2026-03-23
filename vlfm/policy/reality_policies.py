# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
# Modifications Copyright (c) 2026 Yikang.

# 导入必要的模块
from dataclasses import dataclass  # 用于创建数据类
from typing import Any, Dict, List, Union  # 类型注解

import numpy as np  # 数值计算库
import torch  # PyTorch深度学习框架
from omegaconf import DictConfig  # 配置管理库
from PIL import Image  # 图像处理库
from torch import Tensor  # PyTorch张量类型

# VLFM项目内部模块
from vlfm.mapping.obstacle_map import ObstacleMap  # 障碍物地图模块
from vlfm.policy.base_objectnav_policy import VLFMConfig  # VLFM配置基类
from vlfm.policy.itm_policy import ITMPolicyV2  # ITM策略版本2

# 定义Spot机器人机械臂初始化时的偏航角度序列(从-90度到90度,最后回到0度)
INITIAL_ARM_YAWS = np.deg2rad([-90, -60, -30, 0, 30, 60, 90, 0]).tolist()


class RealityMixin:
    """
    实机运行混入类
    这个Python混入类仅包含在真实世界(而非Habitat仿真)中运行ITMPolicyV2所需的代码,
    为任何父类(ITMPolicyV2的子类)提供在Spot机器人上运行的必要方法。
    """

    # 停止动作:线性速度和角速度都为0
    _stop_action: Tensor = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    # 是否加载YOLO模型的标志(实机运行时设为False)
    _load_yolo: bool = False
    # 非COCO数据集对象的文本描述(用于GroundingDINO检测)
    _non_coco_caption: str = (
        "chair . table . tv . laptop . microwave . toaster . sink . refrigerator . book"
        " . clock . vase . scissors . teddy bear . hair drier . toothbrush ."
    )
    # 机械臂初始化偏航角度序列的副本
    _initial_yaws: List = INITIAL_ARM_YAWS.copy()
    # 观测数据缓存字典
    _observations_cache: Dict[str, Any] = {}
    # 策略信息字典
    _policy_info: Dict[str, Any] = {}
    # 初始化完成标志
    _done_initializing: bool = False

    def __init__(self: Union["RealityMixin", ITMPolicyV2], *args: Any, **kwargs: Any) -> None:
        # 调用父类初始化方法,启用探索区域同步
        super().__init__(sync_explored_areas=True, *args, **kwargs)  # type: ignore
        # 从torch hub加载ZoeDepth深度估计模型,用于从RGB图像推断深度
        self._depth_model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", config_mode="eval", pretrained=True).to(
            "cuda" if torch.cuda.is_available() else "cpu"  # 如果有GPU则使用CUDA,否则使用CPU
        )
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
        self: Union["RealityMixin", ITMPolicyV2],
        observations: Dict[str, Any],  # 观测数据字典
        rnn_hidden_states: Union[Tensor, Any],  # RNN隐藏状态
        prev_actions: Any,  # 前一步动作
        masks: Tensor,  # 掩码张量
        deterministic: bool = False,  # 是否确定性行为
    ) -> Dict[str, Any]:
        # 如果目标对象不在非COCO描述中,则将其添加到描述前面
        if observations["objectgoal"] not in self._non_coco_caption:
            self._non_coco_caption = observations["objectgoal"] + " . " + self._non_coco_caption
        # 获取父类引用并调用其act方法
        parent_cls: ITMPolicyV2 = super()  # type: ignore
        action: Tensor = parent_cls.act(observations, rnn_hidden_states, prev_actions, masks, deterministic)[0]

        # 策略输出是一个(1,2)的张量,第一个元素是线性速度,第二个元素是角速度。
        # 将其转换为包含"angular"和"linear"键的字典,以便传递给Spot机器人。
        if self._done_initializing:
            # 如果初始化完成,使用正常的导航动作
            action_dict = {
                "angular": action[0][0].item(),  # 角速度(弧度/秒)
                "linear": action[0][1].item(),   # 线性速度(米/秒)
                "arm_yaw": -1,                   # 机械臂偏航角(-1表示不移动)
                "info": self._policy_info,       # 策略信息
            }
        else:
            # 如果还在初始化阶段,只移动机械臂,不移动底盘
            action_dict = {
                "angular": 0,                    # 底盘角速度为0
                "linear": 0,                     # 底盘线性速度为0
                "arm_yaw": action[0][0].item(),  # 使用动作的第一个值作为机械臂偏航角
                "info": self._policy_info,       # 策略信息
            }

        # 如果策略信息中包含rho_theta(极坐标),则添加到动作字典中
        if "rho_theta" in self._policy_info:
            action_dict["rho_theta"] = self._policy_info["rho_theta"]

        # 如果初始偏航角序列为空,则标记初始化完成
        self._done_initializing = len(self._initial_yaws) == 0

        return action_dict

    def get_action(self, observations: Dict[str, Any], masks: Tensor, deterministic: bool = True) -> Dict[str, Any]:
        """获取动作的简化接口,使用默认的RNN状态和前一步动作"""
        return self.act(observations, None, None, masks, deterministic=deterministic)

    def _reset(self: Union["RealityMixin", ITMPolicyV2]) -> None:
        """重置策略状态"""
        # 调用父类的重置方法
        parent_cls: ITMPolicyV2 = super()  # type: ignore
        parent_cls._reset()
        # 重新复制初始偏航角序列
        self._initial_yaws = INITIAL_ARM_YAWS.copy()
        # 重置初始化完成标志
        self._done_initializing = False

    def _initialize(self) -> Tensor:
        """获取下一个初始化偏航角"""
        # 从初始偏航角序列中弹出第一个角度
        yaw = self._initial_yaws.pop(0)
        # 返回作为张量的偏航角
        return torch.tensor([[yaw]], dtype=torch.float32)

    def _cache_observations(self: Union["RealityMixin", ITMPolicyV2], observations: Dict[str, Any]) -> None:
        """缓存当前时间步的RGB、深度和相机变换等观测数据

        Args:
           observations (Dict[str, Any]): 当前时间步的观测数据
        """
        # 如果观测缓存已有数据,则直接返回(避免重复缓存)
        if len(self._observations_cache) > 0:
            return

        # 获取障碍物地图对象
        self._obstacle_map: ObstacleMap
        # 处理除最后一个之外的所有障碍物地图深度数据
        for obs_map_data in observations["obstacle_map_depths"][:-1]:
            depth, tf, min_depth, max_depth, fx, fy, topdown_fov = obs_map_data
            # 更新障碍物地图,但不进行探索
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

        # 缓存观测数据
        self._observations_cache = {
            "frontier_sensor": frontiers,                            # 前沿传感器数据
            "nav_depth": nav_depth,                                 # 导航深度(用于PointNav)
            "robot_xy": observations["robot_xy"],                   # 机器人XY坐标(2维numpy数组)
            "robot_heading": observations["robot_heading"],         # 机器人朝向(弧度浮点数)
            "object_map_rgbd": observations["object_map_rgbd"],     # 对象地图RGBD数据
            "value_map_rgbd": observations["value_map_rgbd"],       # 价值地图RGBD数据
        }

    def _infer_depth(self, rgb: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        """从RGB图像推断深度图像

        Args:
            rgb (np.ndarray): 用于推断深度的RGB图像
            min_depth (float): 最小深度值
            max_depth (float): 最大深度值

        Returns:
            np.ndarray: 推断得到的深度图像(已归一化到0-1范围)
        """
        # 将numpy数组转换为PIL图像
        img_pil = Image.fromarray(rgb)
        # 使用推理模式(禁用梯度计算)来提高性能
        with torch.inference_mode():
            # 使用ZoeDepth模型推断深度
            depth = self._depth_model.infer_pil(img_pil)
        # 将深度值限制在指定范围内并归一化到0-1
        depth = (np.clip(depth, min_depth, max_depth)) / (max_depth - min_depth)
        return depth


@dataclass
class RealityConfig(DictConfig):
    """实机运行配置类"""
    policy: VLFMConfig = VLFMConfig()  # VLFM策略配置


class RealityITMPolicyV2(RealityMixin, ITMPolicyV2):
    """实机ITM策略版本2
    
    这个类通过多重继承组合了RealityMixin(提供实机运行能力)和ITMPolicyV2(提供ITM策略逻辑)。
    它是VLFM系统在Spot机器人上运行的具体策略实现。
    """
    pass
