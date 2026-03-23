# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
# Modifications Copyright (c) 2026 Yikang.

from typing import Any, List, Union

import cv2
import numpy as np


class TrajectoryVisualizer:
    _num_drawn_points: int = 1  # 内部计数器，记录已经绘制了多少个路径点，默认1
    _cached_path_mask: Union[np.ndarray, None] = (
        None  # 缓存的路径掩膜，这个变量可以是一个 NumPy 数组，也可以是 None，并且初始值是 None
    )
    _origin_in_img: Union[np.ndarray, None] = None  # 原点在图像中的像素坐标
    _pixels_per_meter: Union[float, None] = None  # 缩放比例（1米对应多少像素）
    agent_line_length: int = 10  # 机器人朝向线的长度
    agent_line_thickness: int = 3  # 朝向线的粗细
    path_color: tuple = (0, 255, 0)  # 路径颜色（B,G,R）
    path_thickness: int = 3  # 路径线条粗细
    scale_factor: float = 1.0  # 缩放比例因子，可以整体调整绘制尺寸

    # 初始化时传入原点位置和像素/米的比例
    def __init__(self, origin_in_img: np.ndarray, pixels_per_meter: float):
        self._origin_in_img = origin_in_img
        self._pixels_per_meter = pixels_per_meter

    # 重置轨迹绘制
    def reset(self) -> None:
        self._num_drawn_points = 1  # 已绘制点恢复为1
        self._cached_path_mask = None  # 路径缓存清空

    # 绘制完整轨迹（包括路径和当前机器人位置+朝向）
    def draw_trajectory(
        self,
        img: np.ndarray,
        camera_positions: Union[np.ndarray, List[np.ndarray]],
        camera_yaw: float,  # 朝向角度rad
    ) -> np.ndarray:
        """Draws the trajectory on the image and returns it"""
        img = self._draw_path(img, camera_positions)  # 绘制路径
        img = self._draw_agent(img, camera_positions[-1], camera_yaw)  # 绘制机器人位置或朝向
        return img

    def _draw_path(self, img: np.ndarray, camera_positions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Draws the path on the image and returns it"""
        if len(camera_positions) < 2:
            return img
        # 点太少无法画路径，直接返回

        if self._cached_path_mask is not None:
            path_mask = self._cached_path_mask.copy()
            # 有缓存掩膜，复制一份继续用
        else:
            path_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            # 无缓存就新建一张黑色掩膜（单通道）

        for i in range(self._num_drawn_points - 1, len(camera_positions) - 1):
            path_mask = self._draw_line(path_mask, camera_positions[i], camera_positions[i + 1])
            # 从上次绘制点开始，依次画出相邻点之间的线段

        img[path_mask == 255] = self.path_color
        # 将掩膜中的白色区域用绿色涂色，显示路径
        self._cached_path_mask = path_mask
        # 缓存最新路径掩膜
        self._num_drawn_points = len(camera_positions)
        # 更新已绘制点数
        return img

    def _draw_line(self, img: np.ndarray, pt_a: np.ndarray, pt_b: np.ndarray) -> np.ndarray:
        """Draws a line between two points and returns it"""
        # Convert metric coordinates to pixel coordinates
        px_a = self._metric_to_pixel(pt_a)
        px_b = self._metric_to_pixel(pt_b)

        if np.array_equal(px_a, px_b):
            return img
        # 两点重合不画

        cv2.line(
            img,
            tuple(px_a[::-1]),
            tuple(px_b[::-1]),
            255,
            int(self.path_thickness * self.scale_factor),
        )
        # 在掩膜上画白线
        # 注意坐标翻转，图像索引是行（y），列（x）

        return img

    def _draw_agent(self, img: np.ndarray, camera_position: np.ndarray, camera_yaw: float) -> np.ndarray:
        """Draws the agent on the image and returns it"""
        px_position = self._metric_to_pixel(camera_position)
        # 转换机器人位置到像素坐标
        cv2.circle(
            img,
            tuple(px_position[::-1]),
            int(8 * self.scale_factor),
            (255, 192, 15),
            -1,
        )
        # 画机器人当前位置的圆点（黄色实心）
        # 角度已在里程计层面修正，这里直接使用
        heading_end_pt = (
            int(px_position[0] - self.agent_line_length * self.scale_factor * np.cos(camera_yaw)),
            int(px_position[1] - self.agent_line_length * self.scale_factor * np.sin(camera_yaw)),
        )
        # 计算机器人朝向线终点坐标
        cv2.line(
            img,
            tuple(px_position[::-1]),
            tuple(heading_end_pt[::-1]),
            (0, 0, 0),
            int(self.agent_line_thickness * self.scale_factor),
        )
        # 机器人朝向线终点坐标
        return img

    def draw_circle(self, img: np.ndarray, position: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Draws the point as a circle on the image and returns it"""
        px_position = self._metric_to_pixel(position)
        # 转换坐标
        cv2.circle(img, tuple(px_position[::-1]), **kwargs)
        # 画圆
        return img

    def _metric_to_pixel(self, pt: np.ndarray) -> np.ndarray:
        """Converts a metric coordinate to a pixel coordinate"""
        # Need to flip y-axis because pixel coordinates start from top left
        # 只翻转Y轴，保持X轴方向一致，避免左右翻转
        px = pt * self._pixels_per_meter * np.array([1, -1]) + self._origin_in_img
        # px = pt * self._pixels_per_meter + self._origin_in_img
        # 米坐标乘比例缩放后反转y轴方向，再加上图像原点坐标，得到像素坐标
        px = px.astype(np.int32)
        # 转为整型（像素索引必须是整数）
        return px
