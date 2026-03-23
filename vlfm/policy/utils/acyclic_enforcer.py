# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
# Modifications Copyright (c) 2026 Yikang.

from typing import Any, Set

import numpy as np


class StateAction:
    # 初始化方法，接收位置、动作和其他可选参数
    def __init__(self, position: np.ndarray, action: Any, other: Any = None):
        self.position = position
        self.action = action
        self.other = other

    def __hash__(self) -> int:
        # 定义哈希方法，使StateAction对象可以作为集合或字典的键
        string_repr = f"{self.position}_{self.action}_{self.other}"
        # 将位置、动作和其他信息拼接成字符串
        return hash(string_repr)  # 返回字符串的哈希值


class AcyclicEnforcer:
    # 定义类，用于防止循环行为
    history: Set[StateAction] = set()
    # 存储历史状态-动作对的集合，初始为空集

    def check_cyclic(self, position: np.ndarray, action: Any, other: Any = None) -> bool:
        # 检查给定的状态-动作是否在历史记录中，并返回结果
        state_action = StateAction(position, action, other)
        cyclic = state_action in self.history  # 检查该状态-动作对是否已在历史记录中
        return cyclic

    def add_state_action(self, position: np.ndarray, action: Any, other: Any = None) -> None:
        state_action = StateAction(position, action, other)  # 将状态-动作对添加到历史记录中
        self.history.add(state_action)  # 将该对象添加到历史记录集合中
