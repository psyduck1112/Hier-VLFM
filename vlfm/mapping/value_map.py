# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
# Modifications Copyright (c) 2026 Yikang.

import glob
import json
import os
import os.path as osp
import shutil
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from vlfm.mapping.base_map import BaseMap
from vlfm.utils.geometry_utils import extract_yaw, get_rotation_matrix
from vlfm.utils.img_utils import (
    monochannel_to_inferno_rgb,
    pixel_value_within_radius,
    place_img_in_img,
    rotate_image,
)

DEBUG = False
SAVE_VISUALIZATIONS = False
RECORDING = os.environ.get("RECORD_VALUE_MAP", "0") == "1"
PLAYING = os.environ.get("PLAY_VALUE_MAP", "0") == "1"
RECORDING_DIR = "value_map_recordings"
JSON_PATH = osp.join(RECORDING_DIR, "data.json")
KWARGS_JSON = osp.join(RECORDING_DIR, "kwargs.json")


class ValueMap(BaseMap):
    """
    生成一个地图，用于表示已探索环境中各个区域对于寻找和导航到目标物体的价值高低
    Generates a map representing how valuable explored regions of the environment
    are with respect to finding and navigating to the target object."""

    _confidence_masks: Dict[Tuple[float, float], np.ndarray] = {}  # 缓存置信度掩码
    _camera_positions: List[np.ndarray] = []
    _last_camera_yaw: float = 0.0
    _min_confidence: float = 0.25
    _decision_threshold: float = 0.35  # 决策阈值
    _map: np.ndarray  # 从BaseMap继承的基础地图

    def __init__(
        self,
        value_channels: int,  # 价值通道数
        size: int = 1000,  # 地图大小(像素)
        use_max_confidence: bool = True,
        fusion_type: str = "default",
        obstacle_map: Optional["ObstacleMap"] = None,  # type: ignore # noqa: F821
    ) -> None:
        """
        Args:
            value_channels: The number of channels in the value map.
            size: The size of the value map in pixels.
            use_max_confidence: Whether to use the maximum confidence value in the value
                map or a weighted average confidence value.
            fusion_type: The type of fusion to use when combining the value map with the
                obstacle map.
            obstacle_map: An optional obstacle map to use for overriding the occluded
                areas of the FOV 可选 覆盖FOV区域
        """
        if PLAYING:  # 回放模式
            size = 2000  # 将地图大小设置为2000像素
        super().__init__(size)  # 传入地图大小参数
        self._value_map = np.zeros((size, size, value_channels), np.float32)  # 创建一个三维的NumPy数组作为价值地图
        self._value_channels = value_channels
        self._use_max_confidence = use_max_confidence
        self._fusion_type = fusion_type
        self._obstacle_map = obstacle_map
        if self._obstacle_map is not None:  # 确保价值地图尺度与障碍物一致
            assert self._obstacle_map.pixels_per_meter == self.pixels_per_meter
            assert self._obstacle_map.size == self.size
        if os.environ.get("MAP_FUSION_TYPE", "") != "":  # 检查环境变量是否设置了地图融合类型
            self._fusion_type = os.environ["MAP_FUSION_TYPE"]  # 覆盖构造函数中传入的值

        if RECORDING:  # 记录模式、
            # 删除已有路径，为新的腾空间
            if osp.isdir(RECORDING_DIR):  # 如果记录路径存在
                warnings.warn(f"Recording directory {RECORDING_DIR} already exists. Deleting it.")
                shutil.rmtree(RECORDING_DIR)  # 递归删除整个目录，包括里面的所有文件和子文件夹。
            os.mkdir(RECORDING_DIR)
            # Dump all args to a file
            # 保存参数到文件
            with open(KWARGS_JSON, "w") as f:
                json.dump(
                    {
                        "value_channels": value_channels,
                        "size": size,
                        "use_max_confidence": use_max_confidence,
                    },
                    f,
                )
            # Create a blank .json file inside for now
            with open(JSON_PATH, "w") as f:
                f.write("{}")

    def reset(self) -> None:
        super().reset()
        self._value_map.fill(0)

    def update_map(
        self,
        values: np.ndarray,  # 价值数组
        depth: np.ndarray,  # 深度数组（已归一化）
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,  # 最小深度m
        max_depth: float,  # 最大深度m
        fov: float,  # 视野rad
    ) -> None:
        """Updates the value map with the given depth image, pose, and value to use.

        Args:
            values: The value to use for updating the map.
            depth: The depth image to use for updating the map; expected to be already
                normalized to the range [0, 1].
            tf_camera_to_episodic: The transformation matrix from the episodic frame to
                the camera frame.
            min_depth: The minimum depth value in meters.
            max_depth: The maximum depth value in meters.
            fov: The field of view of the camera in RADIANS.
        """
        assert (
            len(values) == self._value_channels
        ), f"Incorrect number of values given ({len(values)}). Expected {self._value_channels}."

        curr_map = self._localize_new_data(depth, tf_camera_to_episodic, min_depth, max_depth, fov)

        # Fuse the new data with the existing data
        self._fuse_new_data(curr_map, values)

        if RECORDING:
            idx = len(glob.glob(osp.join(RECORDING_DIR, "*.png")))
            img_path = osp.join(RECORDING_DIR, f"{idx:04d}.png")
            cv2.imwrite(img_path, (depth * 255).astype(np.uint8))
            with open(JSON_PATH, "r") as f:
                data = json.load(f)
            data[img_path] = {
                "values": values.tolist(),
                "tf_camera_to_episodic": tf_camera_to_episodic.tolist(),
                "min_depth": min_depth,
                "max_depth": max_depth,
                "fov": fov,
            }
            with open(JSON_PATH, "w") as f:
                json.dump(data, f)

    def sort_waypoints(
        self, waypoints: np.ndarray, radius: float, reduce_fn: Optional[Callable] = None
    ) -> Tuple[np.ndarray, List[float]]:
        # reduce_fn:可选的可调用函数（Callable），用于将多通道价值评估结果压缩为单个值
        """
        在给定范围内选择最佳的航点
        Selects the best waypoint from the given list of waypoints.

        Args:
            waypoints (np.ndarray): An array of 2D waypoints to choose from.
            radius (float): The radius in meters to use for selecting the best waypoint.
            reduce_fn (Callable, optional): The function to use for reducing the values
                within the given radius. Defaults to np.max.

        Returns:
            Tuple[np.ndarray, List[float]]: A tuple of the sorted waypoints and
                their corresponding values.
        """
        radius_px = int(radius * self.pixels_per_meter)  #  计算半径的像素数

        def get_value(point: np.ndarray) -> Union[float, Tuple[float, ...]]:
            # 从地图获取给给定点周围区域的评估值
            x, y = point
            # 将世界坐标系中的x坐标转换为像素坐标，并加上原点偏移
            px = int(-x * self.pixels_per_meter) + self._episode_pixel_origin[0]
            py = int(-y * self.pixels_per_meter) + self._episode_pixel_origin[1]
            point_px = (self._value_map.shape[0] - px, py)
            # 对每个价值通道计算在给定半径内的值
            all_values = [
                pixel_value_within_radius(self._value_map[..., c], point_px, radius_px)
                for c in range(self._value_channels)
            ]
            # 如果只有一个价值通道，直接返回该通道的值
            if len(all_values) == 1:
                return all_values[0]
            # 否则返回包含所有通道值的元组
            return tuple(all_values)

        # 对每个航点计算价值
        values = [get_value(point) for point in waypoints]

        # 如果有多个价值通道，使用提供的归约函数合并每个航点的多个评估值
        if self._value_channels > 1:
            # 确保提供了reduce_fn
            assert reduce_fn is not None, "Must provide a reduction function when using multiple value channels."
            values = reduce_fn(values)

        # Use np.argsort to get the indices of the sorted values
        # 使用np.argsort对值进行降序排序，得到索引
        sorted_inds = np.argsort([-v for v in values])  # type: ignore
        # 根据排序后的索引获取对应的评估值
        sorted_values = [values[i] for i in sorted_inds]
        # 根据排序后的索引获取对应的航点
        sorted_frontiers = np.array([waypoints[i] for i in sorted_inds])

        # 返回排序后的航点和对应的评估值
        return sorted_frontiers, sorted_values

    def visualize(
        self,
        markers: Optional[List[Tuple[np.ndarray, Dict[str, Any]]]] = None,
        reduce_fn: Callable = lambda i: np.max(i, axis=-1),
        obstacle_map: Optional["ObstacleMap"] = None,  # type: ignore # noqa: F821
    ) -> np.ndarray:
        """Return an image representation of the map"""
        # Must negate the y values to get the correct orientation
        reduced_map = reduce_fn(self._value_map).copy()
        if obstacle_map is not None:
            reduced_map[obstacle_map.explored_area == 0] = 0
        map_img = np.flipud(reduced_map)
        # Make all 0s in the value map equal to the max value, so they don't throw off
        # the color mapping (will revert later)
        zero_mask = map_img == 0
        map_img[zero_mask] = np.max(map_img)
        map_img = monochannel_to_inferno_rgb(map_img)
        # Revert all values that were originally zero to white
        map_img[zero_mask] = (255, 255, 255)
        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                map_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

            if markers is not None:
                for pos, marker_kwargs in markers:
                    map_img = self._traj_vis.draw_circle(map_img, pos, **marker_kwargs)

        return map_img

    def _process_local_data(self, depth: np.ndarray, fov: float, min_depth: float, max_depth: float) -> np.ndarray:
        """Using the FOV and depth, return the visible portion of the FOV.
            使用FOV和深度，返回FOV的可见部分
        Args:
            depth: The depth image to use for determining the visible portion of the
                FOV.
        Returns:
            A mask of the visible portion of the FOV.
        """
        # Squeeze out the channel dimension if depth is a 3D array
        if len(depth.shape) == 3:  # 有的深度图可能是 (H, W, 1)
            depth = depth.squeeze(2)  # 去掉多余维度
        # Squash depth image into one row with the max depth value for each column
        # 每一列对应一个水平角度
        depth_row = np.max(depth, axis=0) * (max_depth - min_depth) + min_depth  # 提取每列最大深度，反归一化
        # 每个像素列只保留一个「最远能看到的深度」

        # Create a linspace of the same length as the depth row from -fov/2 to fov/2
        # 创建角度序列
        angles = np.linspace(-fov / 2, fov / 2, len(depth_row))  # 每个角度对应一列

        # Assign each value in the row with an x, y coordinate depending on 'angles'
        # and the max depth value for that column
        x = depth_row  # x 表示深度距离（前方)
        y = depth_row * np.tan(angles)  # 表示左右偏移（用三角函数计算）

        # Get blank cone mask 获取置信度掩码
        cone_mask = self._get_confidence_mask(fov, max_depth)  # 创建一个锥形视野（FOV cone），中心区域置信度更高

        # Convert the x, y coordinates to pixel coordinates
        # 把 (x, y) 世界坐标转换到图像像素坐标
        x = (x * self.pixels_per_meter + cone_mask.shape[0] / 2).astype(int)
        y = (y * self.pixels_per_meter + cone_mask.shape[1] / 2).astype(int)

        # Create a contour from the x, y coordinates, with the top left and right
        # corners of the image as the first two points
        # 拼接轮廓点, 底部左/右点 (start, end), 中间是 (y, x) 轨迹
        last_row = cone_mask.shape[0] - 1
        last_col = cone_mask.shape[1] - 1
        start = np.array([[0, last_col]])  #
        end = np.array([[last_row, last_col]])
        contour = np.concatenate((start, np.stack((y, x), axis=1), end), axis=0)  # 拼接轮廓点

        # Draw the contour onto the cone mask, in filled-in black
        visible_mask = cv2.drawContours(cone_mask, [contour], -1, 0, -1)  # 绘制可见边界
        # type: ignore #

        if DEBUG:
            vis = cv2.cvtColor((cone_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            cv2.drawContours(vis, [contour], -1, (0, 0, 255), -1)
            for point in contour:
                vis[point[1], point[0]] = (0, 255, 0)
            if SAVE_VISUALIZATIONS:
                # Create visualizations directory if it doesn't exist
                if not os.path.exists("visualizations"):
                    os.makedirs("visualizations")
                # Expand the depth_row back into a full image
                depth_row_full = np.repeat(depth_row.reshape(1, -1), depth.shape[0], axis=0)
                # Stack the depth images with the visible mask
                depth_rgb = cv2.cvtColor((depth * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                depth_row_full = cv2.cvtColor((depth_row_full * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                vis = np.flipud(vis)
                new_width = int(vis.shape[1] * (depth_rgb.shape[0] / vis.shape[0]))
                vis_resized = cv2.resize(vis, (new_width, depth_rgb.shape[0]))
                vis = np.hstack((depth_rgb, depth_row_full, vis_resized))
                time_id = int(time.time() * 1000)
                cv2.imwrite(f"visualizations/{time_id}.png", vis)
            else:
                cv2.imshow("obstacle mask", vis)
                cv2.waitKey(0)

        return visible_mask  # 返回可见掩码

    def _localize_new_data(
        self,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fov: float,
    ) -> np.ndarray:
        """
        把相机观测得到的小地图“转换、旋转、定位”到全局地图坐标系中，得到可直接融合的局部地图
        """
        # Get new portion of the map
        curr_data = self._process_local_data(depth, fov, min_depth, max_depth)

        # Rotate this new data to match the camera's orientation
        yaw = extract_yaw(tf_camera_to_episodic)  # 根据偏航角旋转数据
        if PLAYING:
            if yaw > 0:
                yaw = 0
            else:
                yaw = np.deg2rad(30)
        curr_data = rotate_image(curr_data, -yaw)

        # Determine where this mask should be overlaid
        cam_x, cam_y = tf_camera_to_episodic[:2, 3] / tf_camera_to_episodic[3, 3]

        # Convert to pixel units
        px = int(cam_x * self.pixels_per_meter) + self._episode_pixel_origin[0]
        py = int(-cam_y * self.pixels_per_meter) + self._episode_pixel_origin[1]

        # 添加边界检查防止像素坐标越界
        if not (0 <= px < self._map.shape[0] and 0 <= py < self._map.shape[1]):
            # 坐标超出地图边界，返回空地图避免崩溃
            return np.zeros_like(self._map)

        # Overlay the new data onto the map
        curr_map = np.zeros_like(self._map)
        curr_map = place_img_in_img(curr_map, curr_data, px, py)

        return curr_map

    def _get_blank_cone_mask(self, fov: float, max_depth: float) -> np.ndarray:
        """
        生成一个不考虑任何障碍物的FOV视锥掩码

        Args:
            fov: 视野角度（弧度）
            max_depth: 最大深度（米）

        Returns:
            np.ndarray: 表示FOV视锥的二维数组，视锥区域为1，其他区域为0
        """
        # 计算最大深度对应的像素大小
        size = int(max_depth * self.pixels_per_meter)
        # 创建正方形掩码图像，确保视锥能完整放入，中心点在正中心
        cone_mask = np.zeros((size * 2 + 1, size * 2 + 1))
        # 绘制椭圆形状的视锥区域
        cone_mask = cv2.ellipse(
            cone_mask,
            (size, size),  # 中心像素坐标
            (size, size),  # 长轴和短轴长度（这里相等）
            0,  # 椭圆旋转角度
            -np.rad2deg(fov) / 2 + 90,  # 起始角度
            np.rad2deg(fov) / 2 + 90,  # 结束角度
            1,  # 颜色值
            -1,  # 填充形状
        )
        return cone_mask

    def _get_confidence_mask(self, fov: float, max_depth: float) -> np.ndarray:
        """
        生成一个考虑中心区域权重更高的FOV视锥置信度掩码

        Args:
            fov: 视野角度（弧度）
            max_depth: 最大深度（米）

        Returns:
            np.ndarray: 浮点型二维数组，视锥内根据与中心的角度分布赋予置信度值
        """
        # 如果已生成过相同参数的置信度掩码，直接返回副本，避免重复计算
        if (fov, max_depth) in self._confidence_masks:
            return self._confidence_masks[(fov, max_depth)].copy()

        # 生成基础的视锥掩码
        cone_mask = self._get_blank_cone_mask(fov, max_depth)
        # 创建与掩码相同形状的浮点型数组，用于存放置信度值
        adjusted_mask = np.zeros_like(cone_mask).astype(np.float32)

        # 遍历掩码中的每一个像素，计算置信度值
        for row in range(adjusted_mask.shape[0]):
            for col in range(adjusted_mask.shape[1]):
                # 计算当前像素相对于掩码中心的偏移
                horizontal = abs(row - adjusted_mask.shape[0] // 2)
                vertical = abs(col - adjusted_mask.shape[1] // 2)
                # 计算点相对于中心的极角
                angle = np.arctan2(vertical, horizontal)
                # 将角度从[0, fov/2]映射到[0, π/2]，用于后续权重计算
                angle = remap(angle, 0, fov / 2, 0, np.pi / 2)

                # 使用cos²(angle)作为置信度函数，越接近中心值越大，越接近边缘值越小
                confidence = np.cos(angle) ** 2
                # 将置信度值缩放到[self._min_confidence, 1]区间，避免边缘直接变为零
                confidence = remap(confidence, 0, 1, self._min_confidence, 1)
                # 将计算得到的置信度值存入掩码
                adjusted_mask[row, col] = confidence

        # 保证只有视场范围内的像素保留置信度，其他区域为0
        adjusted_mask = adjusted_mask * cone_mask
        # 保存结果到缓存字典，便于下次快速取用
        self._confidence_masks[(fov, max_depth)] = adjusted_mask.copy()

        return adjusted_mask

    def _fuse_new_data(self, new_map: np.ndarray, values: np.ndarray) -> None:
        """
        将新观测数据融合到现有的置信度图和数值图中

        融合方式可配置（替换、均值、加权平均、最大置信度），并考虑障碍物约束与置信度阈值

        Args:
            new_map: 新的置信度图（同维度numpy数组，值在[0,1]范围内）
            values: 新观测对应的属性值（长度必须等于self._value_channels）
        """
        # 检查传入的值数量是否与通道数匹配
        assert (
            len(values) == self._value_channels
        ), f"Incorrect number of values given ({len(values)}). Expected {self._value_channels}."

        # 如果提供了障碍物地图，则使用它来屏蔽新地图中的区域
        if self._obstacle_map is not None:
            explored_area = self._obstacle_map.explored_area
            new_map[explored_area == 0] = 0
            self._map[explored_area == 0] = 0
            self._value_map[explored_area == 0] *= 0

        # 根据融合类型进行不同的处理
        if self._fusion_type == "replace":
            # 替换模式：当前观测的值将覆盖任何现有值
            new_value_map = np.zeros_like(self._value_map)
            new_value_map[new_map > 0] = values
            self._map[new_map > 0] = new_map[new_map > 0]
            self._value_map[new_map > 0] = new_value_map[new_map > 0]
            return
        elif self._fusion_type == "equal_weighting":
            # 等权重模式：更新值始终是当前值和新值的平均值
            self._map[self._map > 0] = 1
            new_map[new_map > 0] = 1
        else:
            assert self._fusion_type == "default", f"Unknown fusion type {self._fusion_type}"

        # 将置信度低于决策阈值且低于现有地图中对应值的区域置为0
        new_map_mask = np.logical_and(new_map < self._decision_threshold, new_map < self._map)
        new_map[new_map_mask] = 0

        if self._use_max_confidence:
            # 使用最大置信度：对于新地图中置信度高于现有地图的每个像素，用新值替换现有值
            higher_new_map_mask = new_map > self._map
            self._value_map[higher_new_map_mask] = values
            # 更新置信度地图
            self._map[higher_new_map_mask] = new_map[higher_new_map_mask]
        else:
            # 使用加权平均：每个像素用现有值和新值的加权平均更新
            confidence_denominator = self._map + new_map
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                # 计算权重
                weight_1 = self._map / confidence_denominator
                weight_2 = new_map / confidence_denominator

            # 扩展权重维度以匹配多通道
            weight_1_channeled = np.repeat(np.expand_dims(weight_1, axis=2), self._value_channels, axis=2)
            weight_2_channeled = np.repeat(np.expand_dims(weight_2, axis=2), self._value_channels, axis=2)

            # 使用加权平均更新数值地图和置信度地图
            self._value_map = self._value_map * weight_1_channeled + values * weight_2_channeled
            self._map = self._map * weight_1 + new_map * weight_2

            # 将置信度分母为0导致的NaN值替换为0
            self._value_map = np.nan_to_num(self._value_map)
            self._map = np.nan_to_num(self._map)


def remap(value: float, from_low: float, from_high: float, to_low: float, to_high: float) -> float:
    """区间缩放
    Maps a value from one range to another.

    Args:
        value (float): The value to be mapped.
        from_low (float): The lower bound of the input range.
        from_high (float): The upper bound of the input range.
        to_low (float): The lower bound of the output range.
        to_high (float): The upper bound of the output range.

    Returns:
        float: The mapped value.
    """
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low


def replay_from_dir() -> None:
    with open(KWARGS_JSON, "r") as f:
        kwargs = json.load(f)
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    v = ValueMap(**kwargs)

    sorted_keys = sorted(list(data.keys()))

    for img_path in sorted_keys:
        tf_camera_to_episodic = np.array(data[img_path]["tf_camera_to_episodic"])
        values = np.array(data[img_path]["values"])
        depth = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        v.update_map(
            values,
            depth,
            tf_camera_to_episodic,
            float(data[img_path]["min_depth"]),
            float(data[img_path]["max_depth"]),
            float(data[img_path]["fov"]),
        )

        img = v.visualize()
        cv2.imshow("img", img)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break


if __name__ == "__main__":
    if PLAYING:
        replay_from_dir()
        quit()

    v = ValueMap(value_channels=1)
    depth = cv2.imread("depth.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    img = v._process_local_data(
        depth=depth,
        fov=np.deg2rad(79),
        min_depth=0.5,
        max_depth=5.0,
    )
    cv2.imshow("img", (img * 255).astype(np.uint8))
    cv2.waitKey(0)

    num_points = 20

    x = [0, 10, 10, 0]
    y = [0, 0, 10, 10]
    angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

    points = np.stack((x, y), axis=1)

    for pt, angle in zip(points, angles):
        tf = np.eye(4)
        tf[:2, 3] = pt
        tf[:2, :2] = get_rotation_matrix(angle)
        v.update_map(
            np.array([1]),
            depth,
            tf,
            min_depth=0.5,
            max_depth=5.0,
            fov=np.deg2rad(79),
        )
        img = v.visualize()
        cv2.imshow("img", img)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
