# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, List

import numpy as np

from vlfm.mapping.traj_visualizer import TrajectoryVisualizer


class BaseMap: #全局，中心为原点的地图
    '''
    初始化，重置，更新地图、机器人轨迹，坐标系转化
    '''
    _camera_positions: List[np.ndarray] = [] #存储相机位置历史
    _last_camera_yaw: float = 0.0 #记录最后的相机偏航 
    _map_dtype: np.dtype = np.dtype(np.float32) #定义地图的数据类型为32位浮点数

    def __init__(self, size: int = 1000, pixels_per_meter: int = 20, *args: Any, **kwargs: Any):
        """
        Args:
            size: The size of the map in pixels.
        """
        self.pixels_per_meter = pixels_per_meter #每米对应的像素数
        self.size = size #地图的边长（像素）
        self._map = np.zeros((size, size), dtype=self._map_dtype) #创建全0的地图矩阵
        self._episode_pixel_origin = np.array([size // 2, size // 2]) #设置地图的中心点
        self._traj_vis = TrajectoryVisualizer(self._episode_pixel_origin, self.pixels_per_meter) #初始化轨迹可视化工具

    def reset(self) -> None: #重置地图状态
        self._map.fill(0) #将地图矩阵清零
        self._camera_positions = [] #清空相机位置历史
        self._traj_vis = TrajectoryVisualizer(self._episode_pixel_origin, self.pixels_per_meter) #重新创建轨迹可视化工具实例

    def update_agent_traj(self, robot_xy: np.ndarray, robot_heading: float) -> None: #更新机器人轨迹
        self._camera_positions.append(robot_xy)  #机器人当前位置（x,y）添加到历史列表
        self._last_camera_yaw = robot_heading #更新最后的机器人当前朝向

        # 世界坐标系：x向右，y向上，原点在地图中心
        # 像素坐标系：x向右，y向下，原点在左上角
    def _xy_to_px(self, points: np.ndarray) -> np.ndarray: #坐标转换方法 真实转像素
        """Converts an array of (x, y) coordinates to pixel coordinates.

        Args:
            points: The array of (x, y) coordinates to convert.

        Returns:
            The array of (x, y) pixel coordinates.
        """
        px = np.rint(points[:, ::-1] * self.pixels_per_meter) + self._episode_pixel_origin
        # [start : stop : step]
        # points[:, ::-1]: 交换x,y列(因为numpy数组是行优先) [::-1] 就是从头到尾，步长为 -1，即把序列倒序
        # * self.pixels_per_meter: 按比例缩放为像素值
        # + self._episode_pixel_origin: 加上地图中心偏移
        px[:, 0] = self._map.shape[0] - px[:, 0] # 翻转y轴(图像坐标系y轴向下)
        return px.astype(int)

    def _px_to_xy(self, px: np.ndarray) -> np.ndarray: #像素转真实
        """Converts an array of pixel coordinates to (x, y) coordinates.

        Args:
            px: The array of pixel coordinates to convert.

        Returns:
            The array of (x, y) coordinates.
        """
        px_copy = px.copy() # 创建副本防止修改原数据
        px_copy[:, 0] = self._map.shape[0] - px_copy[:, 0] # 翻转y轴(与_xy_to_px对应)
        points = (px_copy - self._episode_pixel_origin) / self.pixels_per_meter # 减去中心偏移并缩放为世界坐标
        return points[:, ::-1] # 交换x,y列返回
