# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
# Modifications Copyright (c) 2026 Yikang.
# JetRacer ObjectNav Environment - Reality Architecture Compatible

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import rospy

from vlfm.reality.jetracer_base_env import JetRacerBaseEnv
from vlfm.utils.geometry_utils import wrap_heading

# JetRacer specific constants - 单相机配置
LEFT_CROP = 40   # JetRacer图像裁剪
RIGHT_CROP = 40
TARGET_WIDTH = 640
TARGET_HEIGHT = 480


class JetRacerObjectNavEnv(JetRacerBaseEnv):
    """
    JetRacer ObjectNav环境 - 完全遵循Reality架构
    
    继承JetRacerBaseEnv，适配JetRacer的单RGB-D相机硬件配置
    保持与Spot ObjectNavEnv相同的接口，但简化为单相机实现，不依赖Spot组件
    """

    # Reality架构要求的类变量
    tf_episodic_to_global: np.ndarray = np.eye(4)  # must be set in reset()
    tf_global_to_episodic: np.ndarray = np.eye(4)  # must be set in reset()
    episodic_start_yaw: float = float("inf")       # must be set in reset()
    target_object: str = ""                        # must be set in reset()

    def __init__(
        self,
        robot,  # JetRacerRobot实例
        max_body_cam_depth: float = 3.0,      # JetRacer主相机最大深度
        max_gripper_cam_depth: float = 1.5,   # 保持兼容性（JetRacer无gripper）
        max_lin_dist: float = 0.3,            # JetRacer最大线速度
        max_ang_dist: float = np.deg2rad(30), # JetRacer最大角速度
        time_step: float = 1.0,               # JetRacer时间步长（较慢）
        **kwargs,
    ):
        """
        初始化JetRacer ObjectNav环境
        
        Args:
            robot: JetRacerRobot实例
            max_body_cam_depth: 主相机最大深度范围
            max_gripper_cam_depth: 兼容参数（JetRacer无gripper）
            max_lin_dist: 最大线速度
            max_ang_dist: 最大角速度
            time_step: 步长时间间隔
        """
        self.max_body_cam_depth = max_body_cam_depth
        self.max_gripper_cam_depth = max_gripper_cam_depth  # 保持兼容性
        
        # 初始化ROS节点（如果需要）
        if not rospy.get_node_uri():
            rospy.init_node('jetracer_objectnav_env', anonymous=True)
        
        # 调用父类初始化
        super().__init__(
            robot=robot,
            max_lin_dist=max_lin_dist,
            max_ang_dist=max_ang_dist,
            time_step=time_step,
            **kwargs,
        )
        
        rospy.loginfo("JetRacer ObjectNav环境初始化完成")

    def reset(self, goal: str) -> Dict[str, Any]:
        """
        重置环境 - 完全遵循Reality架构的reset模式
        
        Args:
            goal: 目标物体名称（如"chair", "table"）
            
        Returns:
            初始观测字典
        """
        rospy.loginfo(f"🔄 重置JetRacer ObjectNav环境，目标: {goal}")
        
        # 设置目标物体
        self.target_object = goal
        
        # 获取机器人当前位置作为episodic坐标系原点
        robot_xy, robot_yaw = self.robot.xy_yaw
        self.episodic_start_yaw = robot_yaw
        
        # 构建episodic坐标变换矩阵
        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)
        
        self.tf_episodic_to_global = np.array([
            [cos_yaw, -sin_yaw, 0, robot_xy[0]],
            [sin_yaw,  cos_yaw, 0, robot_xy[1]],
            [0,        0,       1, 0],
            [0,        0,       0, 1]
        ])
        
        self.tf_global_to_episodic = np.linalg.inv(self.tf_episodic_to_global)
        
        # 调用父类reset获取基础观测
        observations = super().reset()
        
        # 添加ObjectNav特定的观测
        observations.update({
            "objectgoal": goal,
            "target_object": goal,
        })
        
        # 获取并处理初始图像观测
        observations.update(self._get_vision_observations())
        
        rospy.loginfo(f"✅ 环境重置完成，目标: {goal}")
        return observations

    def _get_vision_observations(self) -> Dict[str, Any]:
        """
        获取视觉观测数据 - 适配JetRacer单相机
        
        Returns:
            包含处理后图像的观测字典
        """
        vision_obs = {}
        
        try:
            # 从JetRacer获取RGB-D数据
            rgb_image, depth_image = self.robot.get_rgbd()
            
            if rgb_image is not None and depth_image is not None:
                # 处理RGB图像
                processed_rgb = self._process_rgb_image(rgb_image)
                
                # 处理深度图像
                processed_depth = self._process_depth_image(depth_image)
                
                # 构建图像观测字典
                vision_obs.update({
                    "rgb": processed_rgb,
                    "depth": processed_depth,
                    "camera_transform": self.robot.get_camera_transform(),
                    "camera_intrinsics": self.robot.get_camera_intrinsics(),
                })
                
                rospy.logdebug(f"视觉观测获取成功: RGB {processed_rgb.shape}, Depth {processed_depth.shape}")
            
            else:
                rospy.logwarn("⚠️ 无法获取RGB-D数据，使用默认观测")
                vision_obs.update({
                    "rgb": np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8),
                    "depth": np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.float32),
                })
        
        except Exception as e:
            rospy.logerr(f"❌ 视觉观测处理失败: {e}")
            vision_obs.update({
                "rgb": np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8),
                "depth": np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.float32),
            })
        
        return vision_obs

    def _process_rgb_image(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        处理RGB图像 - 裁剪和缩放
        
        Args:
            rgb_image: 原始RGB图像
            
        Returns:
            处理后的RGB图像
        """
        # 确保图像格式正确
        if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
            rospy.logwarn(f"异常RGB图像格式: {rgb_image.shape}")
            return np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        
        # 应用左右裁剪
        h, w = rgb_image.shape[:2]
        if LEFT_CROP + RIGHT_CROP < w:
            rgb_cropped = rgb_image[:, LEFT_CROP:w-RIGHT_CROP]
        else:
            rgb_cropped = rgb_image
        
        # 调整大小到目标尺寸
        rgb_resized = cv2.resize(rgb_cropped, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # 确保数据类型
        return rgb_resized.astype(np.uint8)

    def _process_depth_image(self, depth_image: np.ndarray) -> np.ndarray:
        """
        处理深度图像 - 裁剪、缩放和过滤
        
        Args:
            depth_image: 原始深度图像
            
        Returns:
            处理后的深度图像
        """
        # 应用深度范围限制
        depth_filtered = np.clip(depth_image, 0.1, self.max_body_cam_depth)
        
        # 过滤无效值
        depth_filtered[depth_image <= 0] = 0.0
        depth_filtered[depth_image > self.max_body_cam_depth] = 0.0
        
        # 应用左右裁剪
        h, w = depth_filtered.shape[:2]
        if LEFT_CROP + RIGHT_CROP < w:
            depth_cropped = depth_filtered[:, LEFT_CROP:w-RIGHT_CROP]
        else:
            depth_cropped = depth_filtered
        
        # 调整大小到目标尺寸
        depth_resized = cv2.resize(depth_cropped, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # 确保数据类型
        return depth_resized.astype(np.float32)

    def get_observations(self) -> Dict[str, Any]:
        """
        获取当前观测 - Reality架构标准接口
        
        Returns:
            完整的观测字典
        """
        # 获取基础观测（位置、传感器等）
        observations = super().get_observations()
        
        # 添加视觉观测
        vision_obs = self._get_vision_observations()
        observations.update(vision_obs)
        
        # 添加ObjectNav特定信息
        observations.update({
            "objectgoal": self.target_object,
            "target_object": self.target_object,
            "episodic_gps": self._get_episodic_gps(),
            "episodic_compass": self._get_episodic_compass(),
        })
        
        return observations

    def _get_episodic_gps(self) -> np.ndarray:
        """获取episodic坐标系下的GPS"""
        robot_xy, _ = self.robot.xy_yaw
        global_pos = np.array([robot_xy[0], robot_xy[1], 0.0, 1.0])
        episodic_pos = self.tf_global_to_episodic @ global_pos
        return episodic_pos[:2]  # 返回[x, y]

    def _get_episodic_compass(self) -> float:
        """获取episodic坐标系下的compass"""
        _, robot_yaw = self.robot.xy_yaw
        return wrap_heading(robot_yaw - self.episodic_start_yaw)

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        执行一步 - Reality架构标准接口
        
        Args:
            action: 动作字典
            
        Returns:
            (observations, reward, done, info)
        """
        try:
            # 执行动作
            observations, reward, done, info = super().step(action)
            
            # 更新观测
            vision_obs = self._get_vision_observations()
            observations.update(vision_obs)
            
            # 更新ObjectNav信息
            observations.update({
                "objectgoal": self.target_object,
                "target_object": self.target_object,
            })
            
            # 更新info
            info.update({
                "target_object": self.target_object,
                "robot_position": self.robot.pose,
            })
            
            return observations, reward, done, info
        
        except Exception as e:
            rospy.logerr(f"❌ 环境step执行失败: {e}")
            # 返回安全的默认值
            observations = self.get_observations()
            return observations, -1.0, True, {"error": str(e)}

    def close(self):
        """关闭环境 - Reality架构标准接口"""
        rospy.loginfo("🔄 关闭JetRacer ObjectNav环境")
        
        # 停止机器人
        if hasattr(self.robot, 'stop'):
            self.robot.stop()
        
        # 调用父类关闭
        super().close()
        
        rospy.loginfo("✅ 环境已关闭")
