# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
'''
这个文件实现了一个极简的、可以被 habitat-baselines （PPO trainer）识别并用于评估的“占位”策略（dummy policy）。
目的不是训练一个真实的智能体，而是满足基类/框架对策略对象的接口要求
文件其实就是提供了：注册（registry）→ 类实现（满足接口）→ 一个在命令行直接运行时会生成 dummy_policy.pth 的入口
'''
from typing import Any, Generator

import torch
from habitat import get_config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict # 导入张量字典类，用于处理观测数据
from habitat_baselines.rl.ppo import Policy
from habitat_baselines.rl.ppo.policy import PolicyActionData


@baseline_registry.register_policy # 装饰器将类注册到 Habitat 的策略注册表中
class BasePolicy(Policy):
    """The bare minimum needed to load a policy for evaluation using ppo_trainer.py"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    @property
    def should_load_agent_state(self) -> bool:
        return False

    @classmethod
    def from_config(cls, *args: Any, **kwargs: Any) -> Any:
        return cls()

    def act(
        self,
        observations: TensorDict,
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = False,
    ) -> PolicyActionData:
        # Just moves forwards
        num_envs = observations["rgb"].shape[0]
        action = torch.ones(num_envs, 1, dtype=torch.long)
        return PolicyActionData(actions=action, rnn_hidden_states=rnn_hidden_states)

    # used in ppo_trainer.py eval:

    def to(self, *args: Any, **kwargs: Any) -> None:
        return

    def eval(self) -> None:
        return

    def parameters(self) -> Generator:
        yield torch.zeros(1)


if __name__ == "__main__":
    #使用 `torch.save` 保存一个虚拟的 `state_dict`
    # 这很有用，可以生成一个 `.pth` 文件，用于加载其他甚至不从检查点读取的策略，尽管 Habitat 要求必须加载一个检查点
    # Save a dummy state_dict using torch.save. This is useful for generating a pth file
    # that can be used to load other policies that don't even read from checkpoints,
    # even though habitat requires a checkpoint to be loaded.
    config = get_config("habitat-lab/habitat-baselines/habitat_baselines/config/pointnav/ppo_pointnav_example.yaml") # 获取 Habitat 配置
    dummy_dict = { # 创建虚拟字典，包含配置、额外状态和空的状态字典
        "config": config,
        "extra_state": {"step": 0},
        "state_dict": {},
    }

    torch.save(dummy_dict, "dummy_policy.pth") # 使用 PyTorch 保存虚拟字典为 .pth 文件
