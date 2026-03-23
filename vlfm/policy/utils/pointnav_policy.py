# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete
from torch import Tensor
'''
机器人需要从当前位置导航到目标位置
├── 输入：深度图像 + 目标相对于机器人的位置(rho, theta)
└── 输出：应该执行的动作（前进、左转、右转等）
'''

habitat_version = ""

try:
    import habitat
    from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy

    habitat_version = habitat.__version__
    # 定义适配的 PointNavResNetTensorOutputPolicy 类，统一act方法的返回格式

    if habitat_version == "0.1.5":
        print("Using habitat 0.1.5; assuming SemExp code is being used")

        class PointNavResNetTensorOutputPolicy(PointNavResNetPolicy): # 
            def act(self, *args: Any, **kwargs: Any) -> Tuple[Tensor, Tensor]:
                value, action, action_log_probs, rnn_hidden_states = super().act(*args, **kwargs)
                return action, rnn_hidden_states

    else:
        from habitat_baselines.common.tensor_dict import TensorDict
        from habitat_baselines.rl.ppo.policy import PolicyActionData

        class PointNavResNetTensorOutputPolicy(PointNavResNetPolicy):  # type: ignore
            def act(self, *args: Any, **kwargs: Any) -> Tuple[Tensor, Tensor]:
                policy_actions: "PolicyActionData" = super().act(*args, **kwargs)
                return policy_actions.actions, policy_actions.rnn_hidden_states

    HABITAT_BASELINES_AVAILABLE = True
except ModuleNotFoundError:
    from vlfm.policy.utils.non_habitat_policy.nh_pointnav_policy import (
        PointNavResNetPolicy,  # 从非habitat中导入PointNavResNetPolicy
    )

    class PointNavResNetTensorOutputPolicy(PointNavResNetPolicy):  # type: ignore
        """Already outputs a tensor, so no need to convert."""

        pass

    HABITAT_BASELINES_AVAILABLE = False


def discrete_action_to_continuous(action: Union[int, Tensor]) -> Tuple[float, float]:
    """将Habitat的离散动作转换为JetRacer的连续速度控制
    
    JetRacer控制说明：
    - linear: 驱动轮线速度（提供动力，小车才会移动）
    - angular: 前轮转向角度（仅控制方向，不提供动力）
    
    Habitat离散动作映射为JetRacer控制：
    - STOP = 0: 停止 (linear=0, angular=0)
    - FORWARD = 1: 直行 (linear=正值, angular=0)  
    - TURN_LEFT = 2: 前进左转 (linear=正值, angular=正值)
    - TURN_RIGHT = 3: 前进右转 (linear=正值, angular=负值)
    
    Args:
        action: Habitat的离散动作值 (0-3)
        
    Returns:
        Tuple[float, float]: (linear_vel, angular_vel) JetRacer的速度控制
    """
    if isinstance(action, Tensor):
        # 处理不同形状的action张量
        if action.numel() == 1:
            action = action.item()
        else:
            # 如果action有多个元素，取第一个元素（通常是batch维度）
            action = action.flatten()[0].item()
    
    # JetRacer动作参数
    FORWARD_SPEED = 0.4   # 前进线速度（调大速度）
    TURN_ANGLE = 0.4      # 转向角度
    
    if action == 0:  # STOP - 完全停止
        return 0.0, 0.0
    elif action == 1:  # FORWARD - 直线前进
        return FORWARD_SPEED, 0.0
    elif action == 2:  # TURN_LEFT - 前进并左转
        return FORWARD_SPEED, TURN_ANGLE
    elif action == 3:  # TURN_RIGHT - 前进并右转
        return FORWARD_SPEED, -TURN_ANGLE
    else:
        print(f"警告: 未知的动作值 {action}, 使用停止动作")
        return 0.0, 0.0


class WrappedPointNavResNetPolicy:
    """这是一个封装了 PointNavResNetPolicy 的包装器类，外部只需要调用 act(observations, masks)就能直接得到动作
    主要用于简化基于深度学习和循环神经网络的点目标导航策略的使用。
    Wrapper for the PointNavResNetPolicy that allows for easier usage, however it can
    only handle one environment at a time. Automatically updates the hidden state
    and previous action for the policy.
    """

    def __init__(
        self,
        ckpt_path: str, # 预训练模型检查点路径
        device: Union[str, torch.device] = "cuda", # 计算设备
    ):
        if isinstance(device, str):
            device = torch.device(device) # 将字符串设备名称转换为torch.device对象
            # torch.device 用于指定张量计算和存储的设备。它是一个对象，表示设备类型和设备序号
        self.policy = load_pointnav_policy(ckpt_path)
        # 将模型移动到指定设备
        self.policy.to(device)
        # 判断策略是连续的还是离散的动作
        self.discrete_actions = not hasattr(self.policy.action_distribution, "mu_maybe_std") 
        # hasattr(obj, name): 检查对象 obj 是否具有名为 name 的属性
        
        self.pointnav_test_recurrent_hidden_states = torch.zeros(
            1,  # The number of environments.
            self.policy.net.num_recurrent_layers,
            512,  # hidden state size
            device=device,
        )
        if self.discrete_actions:
            num_actions = 1
            action_dtype = torch.long
        else:
            num_actions = 2
            action_dtype = torch.float32
        self.pointnav_prev_actions = torch.zeros(
            1,  # number of environments
            num_actions,
            device=device,
            dtype=action_dtype,
        )
        self.device = device

    def act(
        self,
        observations: Union["TensorDict", Dict], # 输入观察
        masks: Tensor,
        deterministic: bool = False, # 控制选择方式，确定动作还是采样
    ) -> Tensor:
        """ 根据由深度信息得到的角度预测动作
        Infers action to take towards the given (rho, theta) based on depth vision.

        Args:
            observations (Union["TensorDict", Dict]): A dictionary containing (at least)
                the following:
                    - "depth" (torch.float32): Depth image tensor (N, H, W, 1).
                    - "pointgoal_with_gps_compass" (torch.float32):
                        PointGoalWithGPSCompassSensor tensor representing a rho and
                        theta w.r.t. to the agent's current pose (N, 2).
            masks (torch.bool): Tensor of masks, with a value of 1 for any step after
                the first in an episode; has 0 for first step.
            deterministic (bool): Whether to select a logit action deterministically.

        Returns:
            Tensor: A tensor denoting the action to take.
        """
        # Convert numpy arrays to torch tensors for each dict value
        observations = move_obs_to_device(observations, self.device) # 将 numpy → torch，并放到 device 上：
        # 调用pointnav_policy.act
        pointnav_action, rnn_hidden_states = self.policy.act(
            observations, # 观测数据
            self.pointnav_test_recurrent_hidden_states,  # 历史hidden states 
            self.pointnav_prev_actions,  # 前一步动作
            masks, # 掩码
            deterministic=deterministic, # 控制选择方式
        )
        # 更新内部状态
        self.pointnav_prev_actions = pointnav_action.clone()
        self.pointnav_test_recurrent_hidden_states = rnn_hidden_states
        return pointnav_action

    def act_continuous(
        self,
        observations: Union["TensorDict", Dict],
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[float, float]:
        """获取连续速度控制命令，用于JetRacer控制
        
        Args:
            observations: 观测数据
            masks: 掩码张量
            deterministic: 是否确定性选择动作
            
        Returns:
            Tuple[float, float]: (linear_vel, angular_vel) 适用于JetRacer的速度控制
        """
        # 获取原始动作
        pointnav_action = self.act(observations, masks, deterministic)
        
        # 如果是离散动作，转换为连续速度
        if self.discrete_actions:
            linear_vel, angular_vel = discrete_action_to_continuous(pointnav_action)
            return linear_vel, angular_vel
        else:
            # 如果是连续动作，直接返回（假设格式为[angular, linear]）
            if pointnav_action.numel() >= 2:
                angular_vel = pointnav_action[0, 0].item()
                linear_vel = pointnav_action[0, 1].item()
                return linear_vel, angular_vel
            else:
                print("警告: 连续动作维度不足，使用停止动作")
                return 0.0, 0.0

    def reset(self) -> None:  # 重置 hidden states 和 prev actions（通常在新一轮导航开始时用）
        """
        Resets the hidden state and previous action for the policy.
        """
        self.pointnav_test_recurrent_hidden_states = torch.zeros_like(self.pointnav_test_recurrent_hidden_states)
        self.pointnav_prev_actions = torch.zeros_like(self.pointnav_prev_actions)


def load_pointnav_policy(file_path: str):
    """ 从pth文件中加载PointNavResNetPolicy
    返回：已经加载好权重的策略
    Loads a PointNavResNetPolicy policy from a .pth file.

    Args:
        file_path (str): The path to the trained weights of the pointnav policy.
    Returns:
        PointNavResNetTensorOutputPolicy: The policy.
    """
    # 先检查是否是连续动作
    ckpt_dict = torch.load(file_path, map_location="cpu")
    has_continuous_action = "action_distribution.mu_maybe_std.weight" in ckpt_dict
    
    if HABITAT_BASELINES_AVAILABLE: # 如果habitat可用，优先使用habitat版本（无论离散还是连续）
        # 使用全局定义的 PointNavResNetTensorOutputPolicy 类
        PolicyClass = PointNavResNetTensorOutputPolicy
        
        obs_space = SpaceDict(  # 定义观察空间和动作空间 
            {
                "depth": spaces.Box(low=0.0, high=1.0, shape=(224, 224, 1), dtype=np.float32),
                # 深度图像，是一个形状为 (224, 224, 1) 的浮点型张量，值域在 [0.0, 1.0] 之间
                "pointgoal_with_gps_compass": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
                # 表示带有 GPS 指南针的目标点，是一个包含 2 个元素的浮点型向量
                # 表示相对于代理当前位置的距离 (rho) 和角度 (theta)
            }
        )
        
        # 动作空间已经在函数开头根据checkpoint确定
        if has_continuous_action:
            action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) # 连续动作 [angular, linear]
        else:
            action_space = Discrete(4) # 离散动作
            
        if habitat_version == "0.1.5": # 针对0.1.5版本特殊处理
            pointnav_policy = PolicyClass(
                obs_space,  # 输入观测定义  
                action_space, # 输出动作空间
                hidden_size=512, # RNN 隐藏层大小
                num_recurrent_layers=2, # LSTM堆叠两层
                rnn_type="LSTM", # 循环网络类型
                resnet_baseplanes=32, # 
                backbone="resnet18", # 特征提取器
                normalize_visual_inputs=False,
                obs_transform=None,
            )
            # Need to overwrite the visual encoder because it uses an older version of
            # ResNet that calculates the compression size differently
            # 这里覆盖了 pointnav_policy 的视觉网络，因为 habitat 0.1.5 用的是旧版 ResNet
            # 计算压缩特征大小的方式不同，必须替换为项目自定义的 PointNavResNetNet。
            from vlfm.policy.utils.non_habitat_policy.nh_pointnav_policy import (
                PointNavResNetNet,
            )

            # print(pointnav_policy)
            pointnav_policy.net = PointNavResNetNet(discrete_actions=True, no_fwd_dict=True)
            state_dict = torch.load(file_path + ".state_dict", map_location="cpu")
        else:
            # ckpt_dict已经在前面加载了
            if "config" in ckpt_dict and "state_dict" in ckpt_dict:
                # 标准checkpoint格式：有config和state_dict
                pointnav_policy = PolicyClass.from_config(ckpt_dict["config"], obs_space, action_space)
                state_dict = ckpt_dict["state_dict"]
            else:
                # 只有权重的checkpoint格式：使用habitat策略
                pointnav_policy = PolicyClass(
                    obs_space,
                    action_space,
                    hidden_size=512,
                    num_recurrent_layers=2,
                    rnn_type="LSTM",
                    resnet_baseplanes=32,
                    backbone="resnet18",
                    normalize_visual_inputs=False,
                    obs_transform=None,
                )
                state_dict = ckpt_dict
        pointnav_policy.load_state_dict(state_dict)  # 把权重加载进模型
        return pointnav_policy

    else:
        # 使用非habitat版本的策略
        from vlfm.policy.utils.non_habitat_policy.nh_pointnav_policy import (
            PointNavResNetPolicy as NonHabitatPointNavResNetPolicy,
        )
        
        class LocalPointNavResNetTensorOutputPolicy(NonHabitatPointNavResNetPolicy):
            """Non-habitat wrapper that outputs tensors."""
            pass
        
        pointnav_policy = LocalPointNavResNetTensorOutputPolicy()
        current_state_dict = pointnav_policy.state_dict()
        # Let old checkpoints work with new code
        # 如果检查点文件中没有新版本的连续动作嵌入层参数，则使用旧版本的参数替代
        if "net.prev_action_embedding_cont.bias" not in ckpt_dict.keys():
            ckpt_dict["net.prev_action_embedding_cont.bias"] = ckpt_dict["net.prev_action_embedding.bias"]
        if "net.prev_action_embedding_cont.weights" not in ckpt_dict.keys():
            ckpt_dict["net.prev_action_embedding_cont.weight"] = ckpt_dict["net.prev_action_embedding.weight"]
        # 将检查点文件中的参数加载到当前模型中，仅加载当前模型中存在的参数，忽略不匹配的部分
        pointnav_policy.load_state_dict({k: v for k, v in ckpt_dict.items() if k in current_state_dict})
        # 检查检查点中的参数是否有未被加载的部分，即当前模型中不存在的参数
        unused_keys = [k for k in ckpt_dict.keys() if k not in current_state_dict]
        print(f"The following unused keys were not loaded when loading the pointnav policy: {unused_keys}")
        return pointnav_policy


def move_obs_to_device(
    observations: Dict[str, Any],
    device: torch.device,
    unsqueeze: bool = False,
) -> Dict[str, Tensor]:
    """ 将观察数据移动到给定的设备，将numpy数组转换成torch张量
    Moves observations to the given device, converts numpy arrays to torch tensors.

    Args:
        observations (Dict[str, Union[Tensor, np.ndarray]]): The observations.
        device (torch.device): The device to move the observations to.
        unsqueeze (bool): Whether to unsqueeze the tensors or not.
    Returns:
        Dict[str, Tensor]: The observations on the given device as torch tensors.
    """
    # Convert numpy arrays to torch tensors for each dict value
    for k, v in observations.items():
        if isinstance(v, np.ndarray):
            tensor_dtype = torch.uint8 if v.dtype == np.uint8 else torch.float32
            observations[k] = torch.from_numpy(v).to(device=device, dtype=tensor_dtype)
            if unsqueeze:
                observations[k] = observations[k].unsqueeze(0)

    return observations


if __name__ == "__main__":
    import argparse  # 使用 argparse 解析命令行参数

    parser = argparse.ArgumentParser("Load a checkpoint file for PointNavResNetPolicy") # 
    parser.add_argument("ckpt_path", help="path to checkpoint file")
    # ckpt_path：表示一个“checkpoint 文件的路径”，就是模型保存的文件路径。
    # help="path to checkpoint file"：是对这个参数的解释，会出现在 --help 的帮助信息里。
    args = parser.parse_args() # 开始解析命令行输入，并把结果存到 args 里面

    policy = load_pointnav_policy(args.ckpt_path) # 加载预训练的pointNav模型
    print("Loaded model from checkpoint successfully!")
    mask = torch.zeros(1, 1, device=torch.device("cuda"), dtype=torch.bool)  #创建一个形状为 (1, 1) 的布尔型张量，用于表示 episode 的步骤掩码
    
    observations = {
        "depth": torch.zeros(1, 224, 224, 1, device=torch.device("cuda")),  
        "pointgoal_with_gps_compass": torch.zeros(1, 2, device=torch.device("cuda")),
    }

    # 模型参数移动到GPU
    policy.to(torch.device("cuda"))
    action = policy.act(
        observations,
        torch.zeros(1, 4, 512, device=torch.device("cuda"), dtype=torch.float32),
        torch.zeros(1, 1, device=torch.device("cuda"), dtype=torch.long),
        mask,
    ) # 调用策略的 act 方法执行一次推理
    print("Forward pass successful!")
