# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
# Modifications Copyright (c) 2026 Yikang.

from typing import Any, Union

import cv2
import numpy as np
from frontier_exploration.frontier_detection import detect_frontier_waypoints
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war

from vlfm.mapping.base_map import BaseMap
from vlfm.utils.geometry_utils import extract_yaw, get_point_cloud, transform_points
from vlfm.utils.img_utils import fill_small_holes


class ObstacleMap(BaseMap):
    """生成两张地图：_navigable_map，explored_area
    Generates two maps; one representing the area that the robot has explored so far,
    and another representing the obstacles that the robot has seen so far.
    """

    _map_dtype: np.dtype = np.dtype(bool)  # 地图类型，布尔值
    _frontiers_px: np.ndarray = np.array([])  # px边界点
    frontiers: np.ndarray = np.array([])  # 显示边界点
    radius_padding_color: tuple = (100, 100, 100)  # 机器人半径颜色

    def __init__(
        self,
        min_height: float,  # 高度范围
        max_height: float,
        agent_radius: float,  # 机器人碰撞半径
        area_thresh: float = 3.0,  # square meters 有效边界的最小面积，过滤狭窄缝隙
        hole_area_thresh: int = 100000,  # square pixels 深度图空洞填充阈值，填充小孔洞
        size: int = 1000,
        pixels_per_meter: int = 20,
    ):
        super().__init__(size, pixels_per_meter)
        self.explored_area = np.zeros((size, size), dtype=bool)  # 已探索区域 地图2
        self._map = np.zeros((size, size), dtype=bool)  # 主障碍物地图
        self._navigable_map = np.zeros((size, size), dtype=bool)  # 可通行地图 地图1
        self._min_height = min_height
        self._max_height = max_height
        self._area_thresh_in_pixels = area_thresh * (self.pixels_per_meter**2)
        self._hole_area_thresh = hole_area_thresh
        kernel_size = self.pixels_per_meter * agent_radius * 2
        # round kernel_size to nearest odd number
        kernel_size = int(kernel_size) + (int(kernel_size) % 2 == 0)
        self._navigable_kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def reset(self) -> None:
        super().reset()  # 继承父类重置
        self._navigable_map.fill(0)  # 地图矩阵清零
        self.explored_area.fill(0)  # 已探索区域清零
        self._frontiers_px = np.array([])  # 边界点缓存清零
        self.frontiers = np.array([])

    def update_map(
        self,
        depth: Union[np.ndarray, Any],  # 深度图像（归一化））
        tf_camera_to_episodic: np.ndarray,  # 相机到现实坐标变换矩阵
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,  # fx/fy 相机焦距
        topdown_fov: float,  # 视野角度
        explore: bool = True,
        update_obstacles: bool = True,
    ) -> None:
        """
        将现在视野中所有的障碍物加入地图，同时更新机器人已经探索过的区域
        Adds all obstacles from the current view to the map. Also updates the area
        that the robot has explored so far.

        Args:
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).

            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.
            topdown_fov (float): The field of view of the depth camera projected onto
                the topdown map.
            explore (bool): Whether to update the explored area.
            update_obstacles (bool): Whether to update the obstacle map.
        """
        if update_obstacles:
            if self._hole_area_thresh == -1:  # 如果空洞填充阈值为-1
                filled_depth = depth.copy()  # 复制深度
                filled_depth[depth == 0] = 1.0  # 将所有深度为0的区域设为1
            else:
                filled_depth = fill_small_holes(depth, self._hole_area_thresh)  # 填充小的空洞区域
            scaled_depth = filled_depth * (max_depth - min_depth) + min_depth  # 深度图数值恢复实际
            mask = scaled_depth < max_depth  # 创建掩膜，过滤过大像素点
            point_cloud_camera_frame = get_point_cloud(
                scaled_depth, mask, fx, fy
            )  # 根据深度图和相机内参生成相机坐标系下的3D点云
            point_cloud_episodic_frame = transform_points(
                tf_camera_to_episodic, point_cloud_camera_frame
            )  # 将相机坐标系下的点云转换到全局坐标系（episodic frame）
            obstacle_cloud = filter_points_by_height(
                point_cloud_episodic_frame, self._min_height, self._max_height
            )  # 根据高度过滤点

            # Populate topdown map with obstacle locations
            xy_points = obstacle_cloud[:, :2]  # 提取障碍物点云的x、y坐标
            pixel_points = self._xy_to_px(xy_points)  # 转换为像素坐标

            # 边界检查：过滤超出地图范围的像素点
            valid_mask = (
                (pixel_points[:, 0] >= 0)
                & (pixel_points[:, 0] < self._map.shape[1])
                & (pixel_points[:, 1] >= 0)
                & (pixel_points[:, 1] < self._map.shape[0])
            )
            valid_pixel_points = pixel_points[valid_mask]

            if len(valid_pixel_points) > 0:
                self._map[valid_pixel_points[:, 1], valid_pixel_points[:, 0]] = 1  # 在地图上标记障碍物位置为1

            # 注释掉边界过滤调试信息
            # if len(pixel_points) != len(valid_pixel_points):
            #     print(f"⚠️ 边界过滤: 总点数{len(pixel_points)}, 有效点数{len(valid_pixel_points)}")

            # Update the navigable area, which is an inverse of the obstacle map after a
            # dilation operation to accommodate the robot's radius.
            # 通过膨胀操作扩展障碍物区域（考虑机器人半径）
            # 然后取反得到可导航区域
            self._navigable_map = 1 - cv2.dilate(
                self._map.astype(np.uint8),  # bool转换成数字，后续处理
                self._navigable_kernel,
                iterations=1,  # 迭代一次
            ).astype(
                bool
            )  # 转化为bool

        if not explore:
            print("⚠️ explore=False，跳过探索区域更新")
            return

        # Update the explored area
        agent_xy_location = tf_camera_to_episodic[:2, 3]  # 从变换矩阵中提取智能体的x,y坐标(忽略z轴和旋转部分)
        agent_pixel_location = self._xy_to_px(agent_xy_location.reshape(1, 2))[
            0
        ]  ## 将世界坐标系中的xy坐标转换为像素坐标系中的坐标

        # 注释掉详细探索日志
        # print(f"🔍 探索更新: ({agent_xy_location[0]:.2f}, {agent_xy_location[1]:.2f}), FOV: {topdown_fov}°")

        # 函数返回的是一个 NumPy 数组（二进制掩码），其形状与输入的地图完全相同。
        # 值为1的像素：代表在当前时刻，智能体的视野中刚刚首次看到的区域。这些区域在上一帧还被迷雾笼罩，而在这一帧变成了可见。
        current_angle = -extract_yaw(tf_camera_to_episodic)
        max_line_len_px = max_depth * self.pixels_per_meter
        # topdown_fov已经是度数，不需要转换
        fov_degrees = topdown_fov

        # 记录视野参数
        # print(f"视野: 角度{current_angle:.1f}°, FOV{fov_degrees:.0f}°, 深度{max_line_len_px:.0f}px")

        new_explored_area = reveal_fog_of_war(  # 计算当前视野范围内的新探索区域
            top_down_map=self._navigable_map.astype(np.uint8),  # 可导航地图转换为数字
            current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),  # 当前掩码，可导地图形状，全零
            current_point=agent_pixel_location[::-1],  # 当前智能体位置，反转坐标顺序
            current_angle=current_angle,  # 当前角度（取变换矩阵的yaw角负值）
            fov=fov_degrees,  # 视野角度（弧度转换为度）
            # FOV参数用于：创建一个扇形或锥形的可见区域，与max_line_len一起定义可见区域的边界
            max_line_len=max_line_len_px,  # 最大视线长度（转为像素）
        )

        # 简化的探索结果（注释掉详细日志）
        # new_pixels = np.sum(new_explored_area)
        # if new_pixels > 0:
        #     print(f"   新探索: {new_pixels}px")

        new_explored_area = cv2.dilate(
            new_explored_area, np.ones((3, 3), np.uint8), iterations=1
        )  # 使用3x3卷积核对新探索区域进行膨胀，填充小孔洞与间隙
        self.explored_area[new_explored_area > 0] = 1  # 将新探索的区域合并到总探索的区域中
        # 将self.explored_area对应位置的值设置为1，两数组形状必须一致
        self.explored_area[self._navigable_map == 0] = 0  # 确保不可导航区域不会标记为已探索
        # 查找探索区域的轮廓（只检测最外层轮廓，简单近似）
        contours, _ = cv2.findContours(  # 函数返回两个值，第一个是轮廓列表，第二个是层次结构信息，_表示忽略复杂层次信息
            self.explored_area.astype(np.uint8),
            cv2.RETR_EXTERNAL,  # 轮廓检索模式
            cv2.CHAIN_APPROX_SIMPLE,  # 轮廓近似方法
        )
        # contours是一个列表，其中每个元素都是一个轮廓。每个轮廓本身又是一个包含众多点（坐标）的NumPy数组，这些点连接起来就形成了一个封闭的区域
        if len(contours) > 1:  # 当找到的轮廓数量大于1时
            min_dist = np.inf  # 初始化变量，记录找到的最小距离
            best_idx = 0  # 初始化变量，记录最小距离对应的轮廓索引
            for idx, cnt in enumerate(contours):  # 遍历所有找到的轮廓（idx是索引，cnt是具体的轮廓数据）
                # 计算当前轮廓cnt与智能体位置agent_pixel_location的距离
                dist = cv2.pointPolygonTest(cnt, tuple([int(i) for i in agent_pixel_location]), True)
                if dist >= 0:  # 情况1：智能体在当前轮廓内部或边界上
                    best_idx = idx  # 选中轮廓
                    break  # 跳出循环 已经找到最佳
                # 情况2：智能体在轮廓外部
                elif abs(dist) < min_dist:  # 当前轮廓到智能体的距离比之前记录的最小距离还要小
                    min_dist = abs(dist)  # 更新最小距离
                    best_idx = idx  # 更新最佳轮廓索引
            new_area = np.zeros_like(
                self.explored_area, dtype=np.uint8
            )  # 创建一个和原始探索区域形状、数据类型完全相同的全零数组
            cv2.drawContours(new_area, contours, best_idx, 1, -1)  # type: ignore 在黑色画布上，把best_idx对应的轮廓区域全部涂成白色
            self.explored_area = new_area.astype(
                bool
            )  # 将填充好的uint8类型图像转换回布尔类型,赋值给self.explored_area。

        # Compute frontier locations
        self._frontiers_px = self._get_frontiers()  # 获取边界点像素坐标
        if len(self._frontiers_px) == 0:
            self.frontiers = np.array([])
        else:
            self.frontiers = self._px_to_xy(self._frontiers_px)  # 将像素坐标转化为现实

    def _get_frontiers(self) -> np.ndarray:
        """Returns the frontiers of the map."""
        # Dilate the explored area slightly to prevent small gaps between the explored
        # area and the unnavigable area from being detected as frontiers.
        # 防止已探索区域和不可导航区域之间的小缝隙被误检测为边界
        explored_area = cv2.dilate(
            self.explored_area.astype(np.uint8),
            np.ones((5, 5), np.uint8),  # 5x5的结构元素（核）
            iterations=1,
        )
        frontiers = detect_frontier_waypoints(  # 进行边界检测
            self._navigable_map.astype(np.uint8),
            explored_area,
            self._area_thresh_in_pixels,  # 面积阈值（像素），忽略太小的区域
        )
        return frontiers  # 返回边界点坐标组

    def visualize(self) -> np.ndarray:
        """Visualizes the map."""
        vis_img = np.ones((*self._map.shape[:2], 3), dtype=np.uint8) * 255  # 创建与地图尺寸相同的rgb
        # Draw explored area in light green
        vis_img[self.explored_area == 1] = (200, 255, 200)
        # Draw unnavigable areas in gray
        vis_img[self._navigable_map == 0] = self.radius_padding_color  # 配置的灰色值
        # Draw obstacles in black
        vis_img[self._map == 1] = (0, 0, 0)
        # Draw frontiers in blue (200, 0, 0)
        for frontier in self._frontiers_px:
            cv2.circle(vis_img, tuple([int(i) for i in frontier]), 5, (200, 0, 0), 2)

        vis_img = cv2.flip(vis_img, 0)  # 垂直翻转图像 沿水平轴翻转（上下）图像系与地图系y轴方向相反

        if len(self._camera_positions) > 0:  # 如果有相机位置记录，绘制运动轨迹
            self._traj_vis.draw_trajectory(
                vis_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

        return vis_img


def filter_points_by_height(points: np.ndarray, min_height: float, max_height: float) -> np.ndarray:  # 按照高度筛选点
    return points[(points[:, 2] >= min_height) & (points[:, 2] <= max_height)]
