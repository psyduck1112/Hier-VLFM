#!/usr/bin/env python3
"""
JetRacer VLFM可视化器
参考VLFM的HabitatVis和BaseObjectNavPolicy可视化架构
实现实时导航可视化和视频录制
"""

import cv2
import numpy as np
import rospy
from typing import Dict, List, Optional, Tuple, Any
import torch
from pathlib import Path
import time


class JetRacerVLFMVisualizer:
    """
    JetRacer VLFM可视化器
    参考VLFM BaseObjectNavPolicy和HabitatVis的设计
    """
    
    def __init__(self, map_size: int = 1000, map_scale: float = 20.0):
        """
        初始化可视化器
        
        Args:
            map_size: 地图像素大小
            map_scale: 像素/米比例
        """
        self.map_size = map_size
        self.map_scale = map_scale
        
        # BaseMap兼容的坐标转换参数
        self._episode_pixel_origin = np.array([map_size // 2, map_size // 2])
        
        # 数据存储（参考HabitatVis架构）
        self.rgb: List[np.ndarray] = []
        self.depth: List[np.ndarray] = []
        self.maps: List[np.ndarray] = []
        self.vis_maps: List[List[np.ndarray]] = []
        self.texts: List[List[str]] = []
        self.actions: List[Dict[str, float]] = []
        
        # 轨迹记录
        self.trajectory_points = []
        
        # 状态标志
        self.frame_count = 0
        self.episode_count = 0
        
        # 颜色定义（BGR格式，与VLFM一致）
        self.colors = {
            'robot': (0, 255, 0),          # 绿色
            'goal': (0, 0, 255),           # 红色
            'obstacle': (128, 128, 128),   # 灰色
            'frontier': (0, 0, 200),       # 深红色（前沿点）
            'explored': (200, 255, 200),   # 浅绿色（已探索）
            'trajectory': (255, 0, 255),   # 紫色
            'target_object': (0, 165, 255), # 橙色
            'navigable': (255, 255, 255),  # 白色（可通行）
        }
        
    def collect_data(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        robot_xy: np.ndarray,
        robot_heading: float,
        action_info: Dict[str, Any],
        policy_info: Optional[Dict[str, Any]] = None,
        target_object: str = "",
        obstacle_map: Optional[np.ndarray] = None,
        explored_area: Optional[np.ndarray] = None,
        frontiers: Optional[np.ndarray] = None,
        cosine_similarity: Optional[float] = None
    ) -> None:
        """
        收集单步数据用于可视化（参考HabitatVis.collect_data）
        
        Args:
            rgb_image: RGB观测图像
            depth_image: 深度观测图像
            robot_xy: 机器人位置
            robot_heading: 机器人朝向
            action_info: 动作信息
            policy_info: 策略信息
            target_object: 目标物体名称
            obstacle_map: 障碍物地图(_navigable_map)
            explored_area: 已探索区域地图
            frontiers: 前沿点
            cosine_similarity: 余弦相似度值
        """
        
        # ================== RGB图像处理 ==================
        # _current_rgb是RGB格式，cv2所有函数（imshow/VideoWriter）期望BGR，需转换
        annotated_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        annotated_rgb = self._overlay_status_info(
            annotated_rgb, action_info, target_object, cosine_similarity
        )
        self.rgb.append(annotated_rgb)

        # ================== 深度图像处理 ==================
        # 始终从原始深度图生成可视化，确保每帧独立
        depth_vis = self._process_depth_image(depth_image)
        self.depth.append(depth_vis)
        
        # ================== 地图可视化 ==================
        # 只保留obstacle_map主地图，放大3倍
        map_vis = self._create_map_visualization(
            obstacle_map, explored_area, frontiers, robot_xy, robot_heading
        )
        self.maps.append(map_vis)

        # 其他地图暂时注释掉
        # vis_map_imgs = []
        # if obstacle_map is not None:
        #     vis_map_imgs.append(self._create_obstacle_map_vis(obstacle_map, robot_xy, robot_heading))
        # if policy_info and "value_map" in policy_info:
        #     vis_map_imgs.append(policy_info["value_map"])
        # self.vis_maps.append(vis_map_imgs)
        self.vis_maps.append([])
        
        # ================== 文本信息收集 ==================
        # 参考HabitatVis.collect_data()的文本处理
        text_info = []
        
        # 基本信息
        text_info.append(f"目标: {target_object}")
        text_info.append(f"位置: ({robot_xy[0]:.2f}, {robot_xy[1]:.2f})")
        text_info.append(f"朝向: {robot_heading:.2f}rad")
        text_info.append(f"帧数: {self.frame_count}")
        
        # 动作信息
        if action_info:
            text_info.append(f"线速度: {action_info.get('linear', 0):.3f}")
            text_info.append(f"角速度: {action_info.get('angular', 0):.3f}")
            
        # 余弦相似度
        if cosine_similarity is not None:
            text_info.append(f"余弦相似度: {cosine_similarity:.3f}")
            
        # 前沿点信息
        if frontiers is not None:
            text_info.append(f"前沿点数量: {len(frontiers)}")
            
        self.texts.append(text_info)
        
        # ================== 动作记录 ==================
        self.actions.append(action_info.copy() if action_info else {})
        
        # ================== 轨迹更新 ==================
        # 只有当机器人真正移动时才添加轨迹点
        should_add_point = True
        if len(self.trajectory_points) > 0:
            last_point = self.trajectory_points[-1]
            distance = np.linalg.norm(robot_xy - last_point)
            # 只有移动超过2cm才记录新的轨迹点
            if distance < 0.02:  # 2cm阈值
                should_add_point = False
        
        if should_add_point:
            self.trajectory_points.append(robot_xy.copy())
            # 注释掉轨迹添加日志  
            # rospy.loginfo_throttle(1, f"🛤️ 轨迹点已添加: ({robot_xy[0]:.3f}, {robot_xy[1]:.3f})")
            
        if len(self.trajectory_points) > 200:  # 限制轨迹长度
            self.trajectory_points = self.trajectory_points[-200:]
            
        self.frame_count += 1
        
    def flush_frames(self, failure_cause: str = "Episode Complete") -> List[np.ndarray]:
        """
        输出所有累积帧并返回完整可视化视频序列
        参考HabitatVis.flush_frames()
        
        Args:
            failure_cause: episode结束原因
            
        Returns:
            完整的视频帧序列
        """
        counts = {
            "rgb": len(self.rgb),
            "depth": len(self.depth),
            "maps": len(self.maps),
            "vis_maps": len(self.vis_maps),
            "texts": len(self.texts),
            "actions": len(self.actions),
        }
        rospy.loginfo(f"生成视频帧序列: {counts['rgb']} 帧")
        
        frames = []
        # 防止 collect_data 中途异常导致各缓存长度不一致
        num_frames = min(counts["rgb"], counts["depth"], counts["maps"])
        if len({counts["rgb"], counts["depth"], counts["maps"]}) != 1:
            rospy.logwarn(
                "可视化缓存长度不一致，将按最短长度导出视频: "
                f"rgb={counts['rgb']}, depth={counts['depth']}, maps={counts['maps']}, "
                f"vis_maps={counts['vis_maps']}, texts={counts['texts']}, actions={counts['actions']}"
            )
        if num_frames == 0:
            self.reset()
            return []
        
        for i in range(num_frames):
            try:
                # 创建组合帧（参考HabitatVis._create_frame）
                frame = self._create_combined_frame(
                    depth=self.depth[i],
                    rgb=self.rgb[i],
                    map_img=self.maps[i],
                    vis_map_imgs=self.vis_maps[i] if i < len(self.vis_maps) else [],
                    text_info=self.texts[i] if i < len(self.texts) else []
                )

                # 添加episode状态信息
                frame = self._add_text_overlay(frame, failure_cause, top=True)
                frames.append(frame)
            except Exception as e:
                rospy.logwarn_throttle(5, f"跳过损坏可视化帧 idx={i}: {e}")
                continue
            
        # 统一帧尺寸（参考HabitatVis的pad_images处理）
        if len(frames) > 0:
            frames = self._pad_frames(frames)
            
        # 调整为标准尺寸（参考HabitatVis的resize处理）
        frames = [self._resize_frame(f, target_width=1440) for f in frames]  # 从960增加到1440(1.5倍)
        
        # 重置状态
        self.reset()
        
        return frames
        
    def reset(self) -> None:
        """重置可视化器状态（参考HabitatVis.reset）"""
        self.rgb = []
        self.depth = []
        self.maps = []
        self.vis_maps = []
        self.texts = []
        self.actions = []
        self.frame_count = 0
        self.episode_count += 1
        
    def _process_depth_image(self, depth_image: np.ndarray) -> np.ndarray:
        """
        处理深度图像为可视化格式
        参考BaseObjectNavPolicy的深度图像处理
        """
        if depth_image is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
            
        # 归一化深度值到0-255并转换为RGB
        depth_normalized = np.clip(depth_image, 0, 5.0)  # 限制在5米
        depth_vis = (depth_normalized / 5.0 * 255).astype(np.uint8)
        depth_rgb = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2RGB)
        
        return depth_rgb
        
    def _create_map_visualization(
        self, 
        obstacle_map: Optional[np.ndarray],
        explored_area: Optional[np.ndarray],
        frontiers: Optional[np.ndarray], 
        robot_xy: np.ndarray,
        robot_heading: float
    ) -> np.ndarray:
        """
        创建地图可视化
        参考ObstacleMap.visualize()的实现
        注意：正确处理ObstacleMap的坐标系统和翻转
        """
        # 创建基础地图，初始为白色
        map_vis = np.ones((self.map_size, self.map_size, 3), dtype=np.uint8) * 255
        
        # 绘制地图
        # 注意：ObstacleMap内部用 _map[px[:,1], px[:,0]] 存储，行列与标准图像坐标转置
        # 需要对数组做 .T 才能与 _xy_to_px 的 overlay 坐标对齐
        if obstacle_map is not None and explored_area is not None:
            try:
                obs_T = obstacle_map.T
                exp_T = explored_area.T
                h, w = obs_T.shape[:2]
                if h > 0 and w > 0:
                    resized_obs = cv2.resize(obs_T.astype(np.uint8),
                                            (self.map_size, self.map_size),
                                            interpolation=cv2.INTER_NEAREST)
                    resized_exp = cv2.resize(exp_T.astype(np.uint8),
                                            (self.map_size, self.map_size),
                                            interpolation=cv2.INTER_NEAREST)

                    # 1. 已探索区域 → 浅绿色
                    map_vis[resized_exp > 0] = (200, 255, 200)
                    # 2. 不可通行区域 → 灰色（_navigable_map: 0=障碍物/不可通行）
                    map_vis[resized_obs == 0] = self.colors['obstacle']

            except Exception as e:
                rospy.logwarn_throttle(10, f"地图可视化错误: {e}")
        elif obstacle_map is not None:
            try:
                obs_T = obstacle_map.T
                h, w = obs_T.shape[:2]
                if h > 0 and w > 0:
                    resized_obs = cv2.resize(obs_T.astype(np.uint8),
                                            (self.map_size, self.map_size),
                                            interpolation=cv2.INTER_NEAREST)
                    map_vis[resized_obs == 0] = self.colors['obstacle']

            except Exception as e:
                rospy.logwarn_throttle(10, f"简化地图可视化错误: {e}")
        
        # 绘制前沿点（使用正确的BaseMap坐标转换）
        if frontiers is not None and len(frontiers) > 0:
            try:
                # 确保frontiers是正确的形状 (N, 2) - [x, y]格式
                frontiers_array = np.array(frontiers)
                if frontiers_array.ndim == 1:
                    frontiers_array = frontiers_array.reshape(1, -1)
                
                # 使用BaseMap标准的坐标转换方法
                frontier_px = self._xy_to_px(frontiers_array)
                
                rospy.loginfo_throttle(10, 
                    f"🎯 Frontier转换: {len(frontiers_array)} 个点, "
                    f"world范围: x[{frontiers_array[:, 0].min():.2f}, {frontiers_array[:, 0].max():.2f}] "
                    f"y[{frontiers_array[:, 1].min():.2f}, {frontiers_array[:, 1].max():.2f}]")
                
                # 绘制每个frontier点
                for i, (px_x, px_y) in enumerate(frontier_px):
                    px_x, px_y = int(px_x), int(px_y)
                    if 0 <= px_x < self.map_size and 0 <= px_y < self.map_size:
                        cv2.circle(map_vis, (px_x, px_y), 6, self.colors['frontier'], -1)
                        # 添加点标号便于调试
                        cv2.putText(map_vis, str(i), (px_x + 8, px_y - 8), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    else:
                        rospy.logwarn_throttle(5, f"⚠️ Frontier {i} 超出范围: ({px_x}, {px_y})")
                        
            except Exception as e:
                rospy.logwarn(f"Frontier绘制错误: {e}")
        
        # 绘制机器人位置和朝向（使用BaseMap标准坐标转换）
        robot_points = robot_xy.reshape(1, -1)  # 转换为 (1, 2) 形状
        robot_px = self._xy_to_px(robot_points)[0]  # 获取第一个(也是唯一一个)点
        robot_px_x, robot_px_y = int(robot_px[0]), int(robot_px[1])
        
        # 添加调试信息
        rospy.loginfo_throttle(5, 
            f"🗺️ 坐标转换: world({robot_xy[0]:.2f}, {robot_xy[1]:.2f}) -> "
            f"pixel({robot_px_x}, {robot_px_y}) -> "
            f"map_size={self.map_size}, scale={self.map_scale}")
        
        if 0 <= robot_px_x < self.map_size and 0 <= robot_px_y < self.map_size:
            # 机器人位置
            cv2.circle(map_vis, (robot_px_x, robot_px_y), 4, self.colors['robot'], -1)
            rospy.loginfo_throttle(5, f"✅ 机器人位置绘制成功: ({robot_px_x}, {robot_px_y})")
            
        else:
            rospy.logwarn_throttle(5, f"⚠️ 机器人位置超出地图范围: ({robot_px_x}, {robot_px_y})")
        
        # 绘制轨迹（使用BaseMap标准坐标转换）
        if len(self.trajectory_points) > 1:
            try:
                # 将轨迹点转换为numpy数组
                traj_array = np.array(self.trajectory_points)
                traj_px = self._xy_to_px(traj_array)
                
                # 绘制轨迹线段
                for i in range(1, len(traj_px)):
                    x1, y1 = int(traj_px[i-1][0]), int(traj_px[i-1][1])
                    x2, y2 = int(traj_px[i][0]), int(traj_px[i][1])
                    
                    if (0 <= x1 < self.map_size and 0 <= y1 < self.map_size and
                        0 <= x2 < self.map_size and 0 <= y2 < self.map_size):
                        cv2.line(map_vis, (x1, y1), (x2, y2), 
                               self.colors['trajectory'], 2)
            except Exception as e:
                rospy.logwarn(f"轨迹绘制错误: {e}")
        
        # 逆时针旋转90°：让前进方向(x+)从"右"变成"上"，robot_left(row-)变成"左"
        # 注意：不需要垂直flip，_xy_to_px已经处理了y轴方向
        # 中间像素坐标系：col+=前进, row+=机器人右侧(y-)，旋转后：上=前进, 左=机器人左侧
        map_vis = cv2.rotate(map_vis, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return map_vis
        
    def _create_obstacle_map_vis(
        self, 
        obstacle_map: np.ndarray,
        robot_xy: np.ndarray, 
        robot_heading: float
    ) -> np.ndarray:
        """
        创建障碍物地图的专门可视化
        参考ObstacleMap.visualize()
        """
        # 这里可以根据实际的ObstacleMap结构进行更精确的可视化
        return self._create_map_visualization(obstacle_map, None, None, robot_xy, robot_heading)
        
    def _overlay_status_info(
        self,
        image: np.ndarray,
        action_info: Dict[str, Any],
        target_object: str,
        cosine_similarity: Optional[float]
    ) -> np.ndarray:
        """
        在图像上叠加状态信息
        参考BaseObjectNavPolicy的文本叠加和overlay_frame功能
        """
        img_with_text = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)  # 白色文字
        thickness = 2
        
        # 添加黑色背景使文字更清晰
        y_offset = 30
        
        # 目标物体
        text = f"Target: {target_object}"
        cv2.putText(img_with_text, text, (10, y_offset), font, font_scale, (0, 0, 0), thickness+2)
        cv2.putText(img_with_text, text, (10, y_offset), font, font_scale, color, thickness)
        y_offset += 25
        
        # 动作信息
        if action_info:
            linear = action_info.get('linear', 0)
            angular = action_info.get('angular', 0)
            
            text = f"Linear: {linear:.3f} m/s"
            cv2.putText(img_with_text, text, (10, y_offset), font, font_scale, (0, 0, 0), thickness+2)
            cv2.putText(img_with_text, text, (10, y_offset), font, font_scale, color, thickness)
            y_offset += 25
            
            text = f"Angular: {angular:.3f} rad/s"
            cv2.putText(img_with_text, text, (10, y_offset), font, font_scale, (0, 0, 0), thickness+2)
            cv2.putText(img_with_text, text, (10, y_offset), font, font_scale, color, thickness)
            y_offset += 25
        
        # 余弦相似度
        if cosine_similarity is not None:
            text = f"Cosine Sim: {cosine_similarity:.3f}"
            cv2.putText(img_with_text, text, (10, y_offset), font, font_scale, (0, 0, 0), thickness+2)
            cv2.putText(img_with_text, text, (10, y_offset), font, font_scale, color, thickness)
        
        return img_with_text
        
    def _xy_to_px(self, points: np.ndarray) -> np.ndarray:
        """
        BaseMap兼容的坐标转换方法：世界坐标 -> 像素坐标
        参考BaseMap._xy_to_px的精确实现
        
        Args:
            points: (N, 2) 数组，每行为 [x, y] 世界坐标
            
        Returns:
            (N, 2) 数组，每行为 [px_x, px_y] 像素坐标
        """
        if points.size == 0:
            return np.array([])
            
        # BaseMap._xy_to_px的精确实现:
        # px = np.rint(points[:, ::-1] * self.pixels_per_meter) + self._episode_pixel_origin
        # px[:, 0] = self._map.shape[0] - px[:, 0]
        
        # 1. 交换x,y列: points[:, ::-1] 变成 [y, x]
        points_yx = points[:, ::-1]
        
        # 2. 缩放到像素并加上中心偏移
        px = np.rint(points_yx * self.map_scale) + self._episode_pixel_origin
        
        # 3. 翻转y轴 (图像坐标系y轴向下)
        px[:, 0] = self.map_size - px[:, 0]
        
        # 4. BaseMap返回的是 [px_y, px_x]，但我们需要 [px_x, px_y] 用于OpenCV
        px_opencv = px[:, ::-1]  # 交换回来变成 [px_x, px_y]
        
        return px_opencv.astype(int)
        
    def _px_to_xy(self, px: np.ndarray) -> np.ndarray:
        """
        BaseMap兼容的坐标转换方法：像素坐标 -> 世界坐标
        参考BaseMap._px_to_xy的精确实现
        
        Args:
            px: (N, 2) 数组，每行为 [px_x, px_y] 像素坐标
            
        Returns:
            (N, 2) 数组，每行为 [x, y] 世界坐标
        """
        if px.size == 0:
            return np.array([])
            
        # 输入是OpenCV格式 [px_x, px_y]，需要转换为BaseMap格式 [px_y, px_x]
        px_basemap = px[:, ::-1].copy()  # 交换为 [px_y, px_x]
        
        # BaseMap._px_to_xy的精确实现:
        # px_copy[:, 0] = self._map.shape[0] - px_copy[:, 0]
        # points = (px_copy - self._episode_pixel_origin) / self.pixels_per_meter
        # return points[:, ::-1]
        
        # 1. 翻转y轴
        px_basemap[:, 0] = self.map_size - px_basemap[:, 0]
        
        # 2. 减去中心偏移并缩放为世界坐标
        points_yx = (px_basemap - self._episode_pixel_origin) / self.map_scale
        
        # 3. 交换x,y列返回 [x, y]
        points_xy = points_yx[:, ::-1]
        
        return points_xy
        
    def _create_combined_frame(
        self,
        depth: np.ndarray,
        rgb: np.ndarray,
        map_img: np.ndarray,
        vis_map_imgs: List[np.ndarray],
        text_info: List[str]
    ) -> np.ndarray:
        """
        创建组合帧
        参考HabitatVis._create_frame()的布局策略
        """
        # 左侧：深度图和RGB图垂直堆叠
        try:
            depth_resized = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))
            left_panel = np.vstack((depth_resized, rgb))
        except Exception as e:
            rospy.logwarn(f"图像拼接错误: {e}")
            left_panel = rgb

        # 右侧：单张地图（只含frontier、obstacle、轨迹）
        right_panel = map_img

        try:
            # 将左右面板统一到相同高度后水平拼接
            target_h = max(left_panel.shape[0], right_panel.shape[0])
            left_panel_resized = cv2.resize(left_panel, (
                int(left_panel.shape[1] * target_h / left_panel.shape[0]), target_h))
            right_panel_resized = cv2.resize(right_panel, (
                int(right_panel.shape[1] * target_h / right_panel.shape[0]), target_h))

            combined_frame = np.hstack((left_panel_resized, right_panel_resized))

        except Exception as e:
            rospy.logwarn(f"地图组合错误: {e}")
            combined_frame = left_panel

        return combined_frame
        
    def _resize_images_to_same_height(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """调整图像到相同高度"""
        if not images:
            return images
            
        target_height = min(img.shape[0] for img in images)
        resized = []
        for img in images:
            aspect_ratio = img.shape[1] / img.shape[0]
            new_width = int(target_height * aspect_ratio)
            resized.append(cv2.resize(img, (new_width, target_height)))
        return resized
        
    def _resize_images_to_same_width(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """调整图像到相同宽度"""
        if not images:
            return images
            
        target_width = min(img.shape[1] for img in images)
        resized = []
        for img in images:
            aspect_ratio = img.shape[0] / img.shape[1]
            new_height = int(target_width * aspect_ratio)
            resized.append(cv2.resize(img, (target_width, new_height)))
        return resized
        
    def _add_text_overlay(self, image: np.ndarray, text: str, top: bool = False) -> np.ndarray:
        """
        添加文本叠加
        参考vlfm.utils.visualization.add_text_to_image
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        
        # 计算文本尺寸
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_height = text_size[1] + 20
        
        # 创建文本图像
        text_img = np.ones((text_height, image.shape[1], 3), dtype=np.uint8) * 255
        cv2.putText(text_img, text, (10, text_size[1] + 10), font, font_scale, 
                   (0, 0, 0), font_thickness)
        
        if top:
            return np.vstack([text_img, image])
        else:
            return np.vstack([image, text_img])
            
    def _pad_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        统一帧尺寸
        参考vlfm.utils.visualization.pad_images
        """
        if not frames:
            return frames
            
        max_height = max(f.shape[0] for f in frames)
        max_width = max(f.shape[1] for f in frames)
        
        padded_frames = []
        for frame in frames:
            height_diff = max_height - frame.shape[0]
            width_diff = max_width - frame.shape[1]
            
            padded_frame = np.pad(
                frame,
                ((0, height_diff), (0, width_diff), (0, 0)),
                mode='constant',
                constant_values=255
            )
            padded_frames.append(padded_frame)
            
        return padded_frames
        
    def _resize_frame(self, frame: np.ndarray, target_width: int = 1440) -> np.ndarray:
        """调整帧到目标宽度，保持宽高比"""
        height, width = frame.shape[:2]
        aspect_ratio = height / width
        target_height = int(target_width * aspect_ratio)
        
        return cv2.resize(frame, (target_width, target_height))


class JetRacerVideoRecorder:
    """
    JetRacer视频录制器
    参考VLFMVideoRecorder和habitat-baselines的generate_video功能
    """
    
    def __init__(self, output_dir: str = "/tmp/vlfm_videos", fps: int = 10):
        """
        初始化视频录制器
        
        Args:
            output_dir: 视频输出目录
            fps: 帧率
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.fps = fps
        self.writer = None
        self.current_episode = 0
        
    def save_episode_video(
        self, 
        frames: List[np.ndarray], 
        episode_id: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """
        保存episode视频
        参考habitat-baselines的generate_video
        
        Args:
            frames: 视频帧序列
            episode_id: episode标识
            metrics: episode指标
            
        Returns:
            保存的视频文件路径
        """
        if not frames:
            rospy.logwarn("没有帧数据，跳过视频保存")
            return ""
            
        # 生成文件名
        timestamp = int(time.time())
        if episode_id:
            filename = f"jetracer_episode_{episode_id}_{timestamp}.mp4"
        else:
            filename = f"jetracer_episode_{self.current_episode:03d}_{timestamp}.mp4"
            self.current_episode += 1
            
        # 添加指标到文件名
        if metrics:
            metric_str = "_".join([f"{k}_{v:.2f}" for k, v in metrics.items()])
            filename = filename.replace('.mp4', f'_{metric_str}.mp4')
            
        output_path = self.output_dir / filename
        
        try:
            # 确保所有帧尺寸一致（取第一帧尺寸为准）
            height, width = frames[0].shape[:2]

            # 创建视频写入器，优先使用mp4v，失败则回退XVID
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))
            if not writer.isOpened():
                rospy.logwarn("mp4v codec不可用，尝试XVID")
                output_path = output_path.with_suffix('.avi')
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))

            if not writer.isOpened():
                rospy.logerr("无法创建VideoWriter，检查codec和路径")
                return ""

            # 写入帧（跳过尺寸不匹配的帧）
            written = 0
            for frame in frames:
                if frame.shape[:2] == (height, width):
                    writer.write(frame)
                    written += 1
                else:
                    frame_resized = cv2.resize(frame, (width, height))
                    writer.write(frame_resized)
                    written += 1

            writer.release()

            rospy.loginfo(f"视频已保存: {output_path} ({written}/{len(frames)} 帧)")
            return str(output_path)

        except Exception as e:
            rospy.logerr(f"视频保存失败: {e}")
            return ""
