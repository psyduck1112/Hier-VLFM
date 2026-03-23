# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
# Modifications Copyright (c) 2026 Yikang.

import os
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from torch import Tensor

from vlfm.mapping.frontier_map import FrontierMap
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.utils.geometry_utils import closest_point_within_threshold
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.detections import ObjectDetections
from vlfm.vlm.ultralytics_yoloworld import UltralyticsYOLOWorldITMClient

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

PROMPT_SEPARATOR = "|"


class BaseITMPolicy(BaseObjectNavPolicy):
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5
    _last_value: float = float("-inf")  # 上一个前沿的价值
    _last_frontier: np.ndarray = np.zeros(2)  # 上一个前沿点

    @staticmethod  # 静态方法，不能访问任何类变量和实例变量
    # 用于可视化时将多通道数组压缩为单通道，取最大值
    def _vis_reduce_fn(i: np.ndarray) -> np.ndarray:
        return np.max(i, axis=-1)

    def __init__(
        self,
        text_prompt: str,  # 提示语
        discrete_actions: bool = False,  # 是否离散动作
        use_max_confidence: bool = True,  # 是否使用最大置信度
        sync_explored_areas: bool = False,
        use_ultralytics_yoloworld: bool = True,  # 是否使用Ultralytics YOLO-World
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        # 选择使用Ultralytics YOLO-World或BLIP2进行图像文本匹配
        if use_ultralytics_yoloworld:
            self._itm = UltralyticsYOLOWorldITMClient(
                port=int(os.environ.get("ULTRALYTICS_YOLOWORLD_ITM_PORT", "12187"))
            )
        else:
            self._itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._text_prompt = text_prompt
        self._value_map: ValueMap = ValueMap(
            value_channels=len(text_prompt.split(PROMPT_SEPARATOR)),  # 提示语确定价值通道
            size=1500,  # 统一地图尺寸为1500，与ObstacleMap一致
            use_max_confidence=use_max_confidence,  # 是否使用最大置信度
            obstacle_map=self._obstacle_map if sync_explored_areas else None,
        )  # 创建value_map实例
        self._acyclic_enforcer = AcyclicEnforcer()  # 创建防循环实例

    def _reset(self) -> None:
        super()._reset()
        self._value_map.reset()
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        """
        选择最佳前沿点并生成导航动作
        """
        frontiers = self._observations_cache["frontier_sensor"]  # 观察缓存获取前沿点

        # 注释掉详细的frontier调试信息
        # print(f"🔍 Frontier调试:")
        # print(f"   - frontiers类型: {type(frontiers)}")
        # print(f"   - frontiers形状: {frontiers.shape if hasattr(frontiers, 'shape') else 'No shape'}")
        # print(f"   - frontiers内容: {frontiers}")
        # print(f"   - 是否等于zeros: {np.array_equal(frontiers, np.zeros((1, 2)))}")
        # print(f"   - 长度: {len(frontiers)}")

        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:  # 如果没有前沿点
            print("❌ No frontiers found during exploration, stopping.")
            self._policy_info["stop_reason"] = "exploration_complete"  # 标记停止原因：探索完成
            return self._stop_action  # 停止
        best_frontier, best_value = self._get_best_frontier(observations, frontiers)  # 获取最佳前沿点
        os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"  # 设置调试信息
        print(f"Best value: {best_value*100:.2f}%")
        pointnav_action = self._pointnav(best_frontier, stop=False)  # 调用base_objbav中的_pointnav方法获取动作

        return pointnav_action  # 返回动作

    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        基于self._value_map返回最佳前沿点及其价值
        Returns the best frontier and its value based on self._value_map.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from
                the environment.
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """
        # The points and values will be sorted in descending order
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)  # 根据前沿点价值排序
        robot_xy = self._observations_cache["robot_xy"]
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])  # 前沿点价值前2

        os.environ["DEBUG_INFO"] = ""  # 清空调试信息
        # If there is a last point pursued, then we consider sticking to pursuing it
        # if it is still in the list of frontiers and its current value is not much
        # worse than self._last_value.
        if not np.array_equal(self._last_frontier, np.zeros(2)):  # 如果上一个前沿点不为零
            curr_index = None  # 初始化当前索引为None，用于存储上一个前沿点在当前前沿点列表中的位置

            # 查找上一个前沿点在当前前沿点列表中的位置
            for idx, p in enumerate(sorted_pts):  # 遍历前沿点
                if np.array_equal(p, self._last_frontier):  # 如果前沿点等于上一个前沿点
                    # Last point is still in the list of frontiers
                    curr_index = idx  # 记录索引
                    break

            if curr_index is None:  # 如果列表中没有上一个前沿点
                # 查找与上一个前沿点在阈值范围内最近的点
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier, threshold=0.5)
                #
                if closest_index != -1:  # 如果存在
                    # There is a point close to the last point pursued
                    # 如果存在，将这个最近索引作为当前索引
                    curr_index = closest_index

            if curr_index is not None:  # 如果列表中有上一个前沿点
                curr_value = sorted_values[curr_index]  # 取当前值的价值
                if curr_value + 0.01 > self._last_value:  # 如果当前价值不低于上一个前沿点的价值(允许有0.01的误差)
                    # The last point pursued is still in the list of frontiers and its
                    # value is not much worse than self._last_value
                    print("Sticking to last point.")  # 继续选择该点
                    os.environ["DEBUG_INFO"] += "Sticking to last point. "
                    best_frontier_idx = curr_index  # 当前索引即为最佳

        # If there is no last point pursued, then just take the best point, given that
        # it is not cyclic.
        if best_frontier_idx is None:  # 如果没有上一个前沿点或不符合条件
            for idx, frontier in enumerate(sorted_pts):  # 遍历排序后的点
                cyclic = self._acyclic_enforcer.check_cyclic(
                    robot_xy, frontier, top_two_values
                )  # 检查该点是否会导致循环行为
                if cyclic:  # 如果会
                    print("Suppressed cyclic frontier.")
                    continue  # 跳过该点
                best_frontier_idx = idx  # 设置最佳前沿点索引为当前索引
                break

        if best_frontier_idx is None:  # 如果没有找到最佳前沿点
            print("All frontiers are cyclic. Just choosing the closest one.")  # 所有前沿点是循环的，就选最远的
            os.environ["DEBUG_INFO"] += "All frontiers are cyclic. "
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]  # 取得最佳前沿
        best_value = sorted_values[best_frontier_idx]  # 获取最佳前沿对应的价值
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)  # 将选择记录到防循环器中
        self._last_value = best_value  # 更新上一个前沿点的价值
        self._last_frontier = best_frontier  # 更新上一个前沿点的坐标
        os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%"

        return best_frontier, best_value  # 返回最佳前沿点坐标和价值

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)

        if not self._visualize:
            return policy_info

        markers = []

        # Draw frontiers on to the cost map
        frontiers = self._observations_cache["frontier_sensor"]
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            markers.append((frontier[:2], marker_kwargs))

        if not np.array_equal(self._last_goal, np.zeros(2)):
            # Draw the pointnav goal on to the cost map
            if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                color = self._selected__frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal, marker_kwargs))
        policy_info["value_map"] = cv2.cvtColor(
            self._value_map.visualize(markers, reduce_fn=self._vis_reduce_fn),
            cv2.COLOR_BGR2RGB,
        )

        return policy_info

    def _update_value_map(self) -> None:  # 更新价值图
        all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
        # 从self._observations_cache["value_map_rgbd"]这个列表中提取每个元素的第一个子元素（索引为0），并组成一个新的列表

        # 检查是否使用YOLO-World
        if isinstance(self._itm, UltralyticsYOLOWorldITMClient):
            # DEBUG: 调试目标对象传递
            # 注释掉详细日志
            # print(f"🎯 ITM._update_value_map: _target_object = '{self._target_object}'")
            # print(f"📸 ITM._update_value_map: RGB count = {len(all_rgb)}")

            # 对于YOLO-World，直接使用target_object
            cosines = [[self._itm.cosine(rgb, self._target_object)] for rgb in all_rgb]  # 遍历所有RGB图像
        else:
            # 对于BLIP2，保持原有逻辑
            cosines = [
                [
                    self._itm.cosine(
                        rgb,
                        p.replace("target_object", self._target_object.replace("|", "/")),
                        # 将目标物体名称中的"|"替换为"/"
                        # 将子提示中的"target_object"占位符换成当前目标对象名称
                    )
                    for p in self._text_prompt.split(PROMPT_SEPARATOR)  # 遍历得到的每个提示子字符串
                ]
                for rgb in all_rgb  # 遍历所有RGB图像
            ]
        for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            cosines, self._observations_cache["value_map_rgbd"]  # 将两个可迭代对象配对组合
        ):  # 遍历所有cosines与观察rgbd数据
            self._value_map.update_map(np.array(cosine), depth, tf, min_depth, max_depth, fov)  # 更新价值图

        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        """抽象方法，由子类实现前沿点排序逻辑"""
        raise NotImplementedError


class ITMPolicy(BaseITMPolicy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frontier_map: FrontierMap = FrontierMap()

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        self._pre_step(observations, masks)
        if self._visualize:  # 如果可视化，更新价值图
            self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _reset(self) -> None:
        super()._reset()
        self._frontier_map.reset()

    def _sort_frontiers_by_value(  # 根据价值对前沿点进行排序
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        rgb = self._observations_cache["object_map_rgbd"][0][0]  # 读取RGB图像
        text = self._text_prompt.replace("target_object", self._target_object)  # 处理语句
        self._frontier_map.update(frontiers, rgb, text)  # 更新边界地图
        return self._frontier_map.sort_waypoints()


class ITMPolicyV2(BaseITMPolicy):
    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        self._pre_step(observations, masks)
        self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5)
        return sorted_frontiers, sorted_values


class ITMPolicyV3(ITMPolicyV2):
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._exploration_thresh = exploration_thresh

        def visualize_value_map(arr: np.ndarray) -> np.ndarray:
            # Get the values in the first channel
            first_channel = arr[:, :, 0]
            # Get the max values across the two channels
            max_values = np.max(arr, axis=2)
            # Create a boolean mask where the first channel is above the threshold
            mask = first_channel > exploration_thresh
            # Use the mask to select from the first channel or max values
            result = np.where(mask, first_channel, max_values)

            return result

        self._vis_reduce_fn = visualize_value_map  # type: ignore

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5, reduce_fn=self._reduce_values)

        return sorted_frontiers, sorted_values

    def _reduce_values(self, values: List[Tuple[float, float]]) -> List[float]:
        """
        Reduce the values to a single value per frontier

        Args:
            values: A list of tuples of the form (target_value, exploration_value). If
                the highest target_value of all the value tuples is below the threshold,
                then we return the second element (exploration_value) of each tuple.
                Otherwise, we return the first element (target_value) of each tuple.

        Returns:
            A list of values, one per frontier.
        """
        target_values = [v[0] for v in values]
        max_target_value = max(target_values)

        if max_target_value < self._exploration_thresh:
            explore_values = [v[1] for v in values]
            return explore_values
        else:
            return [v[0] for v in values]
