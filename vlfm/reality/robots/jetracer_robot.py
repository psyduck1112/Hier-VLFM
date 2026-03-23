# JetRacer Robot - Reality Architecture Compatible
# 专为JetRacer设计，不继承BaseRobot（那是为Spot设计的）

import math
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu

from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix


class JetRacerRobot:
    """
    JetRacer机器人类 - 专用Reality架构实现
    
    不继承BaseRobot（那是为Spot的多相机、机械臂设计的）
    而是提供适合JetRacer的简洁接口：单RGB-D相机 + 地面移动
    """
    
    def __init__(
        self,
        namespace: str = "",
        rgb_topic: str = "/camera/color/image_raw",
        depth_topic: str = "/camera/depth/image_rect_raw", 
        cmd_topic: str = "/cmd_vel",
        odom_topic: str = "/odom",
        imu_topic: str = "/imu",
    ):
        """
        初始化JetRacer机器人
        
        Args:
            namespace: ROS命名空间
            rgb_topic: RGB图像话题
            depth_topic: 深度图像话题
            cmd_topic: 速度控制话题
            odom_topic: 里程计话题
            imu_topic: IMU话题
        """
        self.namespace = namespace
        self.cv_bridge = CvBridge()
        
        # 传感器数据缓存
        self._current_rgb: Optional[np.ndarray] = None
        self._current_depth: Optional[np.ndarray] = None
        self._current_pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        
        # 时间戳
        self._last_rgb_time = rospy.Time.now()
        self._last_depth_time = rospy.Time.now()
        
        # 相机参数（RealSense D435i典型值）
        self.camera_params = {
            "fx": 615.0, "fy": 615.0,
            "cx": 320.0, "cy": 240.0,
            "width": 640, "height": 480,
            "depth_scale": 0.001,
        }
        
        # 创建ROS接口
        self._setup_ros_interface(rgb_topic, depth_topic, cmd_topic, odom_topic, imu_topic)
        
        rospy.loginfo("JetRacer机器人初始化完成")
    
    def _setup_ros_interface(self, rgb_topic: str, depth_topic: str, cmd_topic: str, 
                           odom_topic: str, imu_topic: str):
        """设置ROS订阅和发布"""
        # 订阅传感器数据
        rospy.Subscriber(rgb_topic, Image, self._rgb_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber(depth_topic, Image, self._depth_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber(odom_topic, Odometry, self._odom_callback, queue_size=1)
        rospy.Subscriber(imu_topic, Imu, self._imu_callback, queue_size=1)
        
        # 创建控制发布者
        self._cmd_pub = rospy.Publisher(cmd_topic, Twist, queue_size=1)
    
    def _rgb_callback(self, msg: Image):
        """RGB图像回调"""
        try:
            if msg.encoding == "bgr8":
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
                self._current_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            elif msg.encoding == "rgb8":
                self._current_rgb = self.cv_bridge.imgmsg_to_cv2(msg, "rgb8")
            else:
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
                if len(cv_image.shape) == 3:
                    self._current_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                else:
                    self._current_rgb = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
            
            self._last_rgb_time = rospy.Time.now()
            
        except Exception as e:
            rospy.logerr(f"RGB处理错误: {e}")
    
    def _depth_callback(self, msg: Image):
        """深度图像回调"""
        try:
            if msg.encoding == "16UC1":
                depth_image = self.cv_bridge.imgmsg_to_cv2(msg, "16UC1")
                self._current_depth = depth_image.astype(np.float32) * self.camera_params["depth_scale"]
            elif msg.encoding == "32FC1":
                self._current_depth = self.cv_bridge.imgmsg_to_cv2(msg, "32FC1")
            else:
                depth_raw = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
                if depth_raw.dtype == np.uint16:
                    self._current_depth = depth_raw.astype(np.float32) * self.camera_params["depth_scale"]
                else:
                    self._current_depth = depth_raw.astype(np.float32)
            
            # 过滤异常深度值
            if self._current_depth is not None:
                valid_mask = (self._current_depth > 0.1) & (self._current_depth < 10.0)
                self._current_depth = np.where(valid_mask, self._current_depth, 0.0)
                
            self._last_depth_time = rospy.Time.now()
            
        except Exception as e:
            rospy.logerr(f"深度处理错误: {e}")
    
    def _odom_callback(self, msg: Odometry):
        """里程计回调"""
        try:
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            
            yaw = math.atan2(
                2 * (ori.w * ori.z + ori.x * ori.y),
                1 - 2 * (ori.y * ori.y + ori.z * ori.z)
            )
            
            self._current_pose = {"x": pos.x, "y": pos.y, "yaw": yaw}
            
        except Exception as e:
            rospy.logerr(f"里程计处理错误: {e}")
    
    def _imu_callback(self, msg: Imu):
        """IMU回调 - 可以用于融合定位"""
        pass  # JetRacer主要依赖里程计
    
    # ===== Reality架构要求的核心接口 =====
    
    @property
    def pose(self) -> Tuple[float, float, float]:
        """获取机器人位置 - Reality架构核心接口"""
        return self._current_pose["x"], self._current_pose["y"], self._current_pose["yaw"]
    
    @property
    def xy_yaw(self) -> Tuple[np.ndarray, float]:
        """获取位置和朝向 - 兼容性接口"""
        pose = self.pose
        return np.array([pose[0], pose[1]]), pose[2]
    
    def get_rgbd(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        获取RGB-D数据 - JetRacer核心接口
        
        Returns:
            Tuple[rgb_image, depth_image]: RGB和深度图像
        """
        if self._current_rgb is None or self._current_depth is None:
            return None, None
        
        return self._current_rgb.copy(), self._current_depth.copy()
    
    def move(self, linear_vel: float, angular_vel: float):
        """
        移动机器人 - JetRacer核心接口
        
        Args:
            linear_vel: 线速度 (m/s)
            angular_vel: 角速度 (rad/s)
        """
        cmd = Twist()
        cmd.linear.x = np.clip(linear_vel, -0.5, 0.5)
        cmd.angular.z = np.clip(angular_vel, -1.0, 1.0)
        self._cmd_pub.publish(cmd)
    
    def stop(self):
        """停止机器人"""
        self.move(0.0, 0.0)
    
    def get_camera_transform(self) -> np.ndarray:
        """
        获取相机变换矩阵（相对于机器人基座）
        
        Returns:
            4x4变换矩阵
        """
        # JetRacer相机前向安装，稍微向上倾斜
        return xyz_yaw_to_tf_matrix(0.1, 0.0, 0.15, 0.0)  # 10cm前，15cm高
    
    def get_camera_intrinsics(self) -> np.ndarray:
        """
        获取相机内参矩阵
        
        Returns:
            3x3内参矩阵
        """
        fx, fy = self.camera_params["fx"], self.camera_params["fy"]
        cx, cy = self.camera_params["cx"], self.camera_params["cy"]
        
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    
    def get_robot_transform(self) -> np.ndarray:
        """
        获取机器人基座变换矩阵
        
        Returns:
            4x4变换矩阵
        """
        pose = self.pose
        return xyz_yaw_to_tf_matrix(pose[0], pose[1], 0.0, pose[2])
    
    def is_ready(self) -> bool:
        """检查机器人是否准备就绪"""
        return (self._current_rgb is not None and 
                self._current_depth is not None and
                rospy.Time.now() - self._last_rgb_time < rospy.Duration(2.0) and
                rospy.Time.now() - self._last_depth_time < rospy.Duration(2.0))
    
    def disconnect(self):
        """断开连接"""
        self.stop()
        rospy.loginfo("JetRacer机器人断开连接")