#!/usr/bin/env python3
# 第1行：Shebang行，告诉系统用python3解释器执行此脚本
# /usr/bin/env python3 会自动找到系统中的python3路径

"""
多行文档字符串（docstring）
描述整个程序的功能和用途
VLFM JetRacer控制器 - 主机端
直接使用Waveshare JetRacer现有ROS接口
"""

# ===== 导入标准库模块 =====
import rospy          # ROS的Python API，用于创建节点、订阅发布话题
import numpy as np    # NumPy数值计算库，用于数组和矩阵操作
import torch          # PyTorch深度学习框架，用于张量运算和神经网络
import math           # 数学函数库，提供三角函数、对数等数学运算
import time            # 计时
import threading       # 多线程支持
from typing import Dict, Any, Optional  # 类型注解，Dict字典类型，Any任意类型，Optional可选类型
import cv2            # OpenCV计算机视觉库，用于图像处理
from cv_bridge import CvBridge  # ROS与OpenCV之间的图像格式转换工具

# ===== 导入ROS消息类型 =====
from sensor_msgs.msg import Image, Imu    # ROS传感器消息：Image图像消息，Imu惯导消息
from nav_msgs.msg import Odometry         # ROS导航消息：里程计消息，包含位置和速度
from geometry_msgs.msg import Twist       # ROS几何消息：速度指令，包含线速度和角速度
try:
    from message_filters import ApproximateTimeSynchronizer, Subscriber as MfSubscriber
    MESSAGE_FILTERS_AVAILABLE = True
except ImportError:
    ApproximateTimeSynchronizer = None  # type: ignore
    MfSubscriber = None  # type: ignore
    MESSAGE_FILTERS_AVAILABLE = False

# ===== 添加模块路径 =====
import sys
import os
# 添加vlfm项目根目录到Python路径，确保能够导入vlfm模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ===== 导入VLFM策略模块 =====
from vlfm.policy.jetracer_ros_policy import JetRacerROSITMPolicyV2  # VLFM的JetRacer ROS策略V2版本
from vlfm.mapping.obstacle_map import ObstacleMap           # VLFM障碍物地图构建模块
from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix  # VLFM几何工具：坐标变换矩阵计算
from vlfm.utils.jetracer_vlfm_visualizer import JetRacerVLFMVisualizer, JetRacerVideoRecorder  # VLFM风格可视化
from omegaconf import OmegaConf                            # 配置管理库：OmegaConf配置工厂


class VLFMJetRacerController:
    """
    类文档字符串：描述这个类的功能
    VLFM JetRacer控制器类
    功能：连接到Waveshare JetRacer的ROS话题，运行VLFM导航算法
    """
    
    def __init__(self, jetracer_namespace=""):
        """
        构造函数（初始化方法）
        参数：
            jetracer_namespace: str类型，JetRacer话题的命名空间，默认为空字符串
        """
        # 初始化ROS节点，节点名称为'vlfm_jetracer_controller'
        # anonymous=True 表示允许多个同名节点同时运行（会自动添加随机后缀）
        rospy.init_node('vlfm_jetracer_controller', anonymous=True)
        
        # 实例属性赋值：存储命名空间参数到对象属性中
        self.namespace = jetracer_namespace
        
        # 创建CV Bridge对象，用于ROS Image消息与OpenCV图像格式的相互转换
        self.cv_bridge = CvBridge()
        
        # ===== 传感器数据缓存区 =====
        # 使用私有属性（下划线开头）存储当前时刻的传感器数据
        
        # RGB图像缓存：Optional[np.ndarray] 表示可以是NumPy数组或None
        self._current_rgb: Optional[np.ndarray] = None
        
        # 深度图像缓存：同样可以是NumPy数组或None
        self._current_depth: Optional[np.ndarray] = None
        
        # 里程计数据缓存：使用字典存储x坐标、y坐标和yaw角度（偏航角）
        self._current_odom = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        
        # IMU数据缓存：目前只存储yaw角度信息
        self._current_imu = {"yaw": 0.0}
        
        # 数据时间戳记录
        self._last_rgb_timestamp = 0.0
        self._last_depth_timestamp = 0.0
        self._last_odom_timestamp = 0.0
        
        
        # ===== VLFM算法组件 =====
        # VLFM策略对象：Optional[JetRacerROSITMPolicyV2] 表示可以是JetRacerROSITMPolicyV2对象或None
        # 初始化时设为None，后续在_initialize_vlfm()中创建
        self.policy: Optional[JetRacerROSITMPolicyV2] = None
        
        # 障碍物地图对象：在_initialize_vlfm()中从策略获取引用
        self.obstacle_map: Optional[ObstacleMap] = None
        
        # ===== VLFM风格可视化组件 =====
        self.visualizer = JetRacerVLFMVisualizer(map_size=1500, map_scale=20.0)  # 统一地图大小为1500
        self.video_recorder = JetRacerVideoRecorder(output_dir="/tmp/vlfm_jetracer_videos", fps=10)
        self.enable_visualization = True
        self.save_video = True
        self.current_episode_frames = []
        self.episode_start_time = None
        self._episode_recording_active = False
        self._episode_video_saved = False
        
        # 目标物体名称：机器人要寻找的目标，默认为"chair"（椅子）
        self.target_object = "chair"
        
        # ===== Intel RealSense D435i相机参数字典 =====
        self.camera_params = {
            # fx, fy: 焦距参数（像素单位），需要通过相机标定获得准确值
            "fx": 616.0, "fy": 616.0, 
            # cx, cy: 主点坐标（像素单位），通常在图像中心附近
            "cx": 320.0, "cy": 240.0,  
            # width, height: 图像分辨率（像素）
            "width": 640, "height": 480,  
            # depth_scale: 深度数据单位转换因子，RealSense输出毫米，转换为米
            "depth_scale": 0.001  
        }
        # 俯视FOV（度）：与相机内参一致，避免固定90度造成探索扇区失真
        self._topdown_fov_deg = float(
            np.degrees(2.0 * np.arctan(self.camera_params["width"] / (2.0 * self.camera_params["fx"])))
        )
        
        
        # 调用方法设置ROS通信接口（订阅者和发布者）
        self._setup_ros_interface()
        
        # 调用方法初始化VLFM策略和相关组件
        self._initialize_vlfm()
        rospy.on_shutdown(self._on_ros_shutdown)
        
        # 使用ROS日志系统输出信息级别的消息
        rospy.loginfo("VLFM JetRacer控制器启动完成")

    def _on_ros_shutdown(self) -> None:
        """ROS关闭时兜底保存视频，避免Ctrl+C路径漏保存。"""
        try:
            self._cmd_pub.publish(Twist())
        except Exception:
            pass

        try:
            if self._episode_recording_active and not self._episode_video_saved:
                rospy.loginfo("ROS shutdown触发，执行兜底视频保存...")
                self.end_episode_recording(success=False)
        except Exception as e:
            rospy.logwarn(f"shutdown兜底视频保存失败: {e}")

    def _setup_ros_interface(self):
        """
        私有方法：设置ROS通信接口
        功能：创建订阅者（接收数据）和发布者（发送数据）
        """
        
        # ===== RGB/Depth同步订阅（按时间戳近似对齐）=====
        possible_rgb_topics = [
            '/camera/color/image_raw',
            '/csi_cam_0/image_raw',
            '/camera/image_raw',
            '/usb_cam/image_raw',
            '/csi_cam/image_raw',
        ]
        possible_depth_topics = [
            '/camera/aligned_depth_to_color/image_raw',
            '/camera/depth/aligned_depth_to_color/image_raw',
            '/camera/depth/image_rect_raw',
        ]

        rgb_topic = self._select_topic(possible_rgb_topics, '/camera/color/image_raw')
        depth_topic = self._select_topic(possible_depth_topics, '/camera/depth/image_rect_raw')
        rospy.loginfo(f"RGB话题: {rgb_topic}, Depth话题: {depth_topic}")

        # 不使用时间戳同步：JetRacer速度慢(<0.3m/s)，RGB/Depth几十ms的偏差
        # 对地图精度影响可忽略，且同步会在计算阻塞期间大量丢帧
        self._rgb_sub = rospy.Subscriber(
            rgb_topic,
            Image,
            self._rgb_callback,
            queue_size=2,
            buff_size=2**24,
        )
        self._depth_sub = rospy.Subscriber(
            depth_topic,
            Image,
            self._depth_callback,
            queue_size=2,
            buff_size=2**24,
        )
        rospy.loginfo("✅ 使用独立RGB/Depth回调（无时间戳同步）")
        
        # 订阅IMU惯性测量单元数据 - 使用JetRacer的IMU话题
        rospy.Subscriber(
            '/imu',                             # JetRacer IMU话题
            Imu,                                # 消息类型：IMU消息，包含姿态和加速度
            self._imu_callback,                 # IMU数据回调函数
            queue_size=10                       # 队列大小：1（IMU数据较小，不需要大缓冲区）
        )
        
        # ===== 订阅JetRacer底盘数据 =====
        # 订阅里程计数据（机器人位置和速度信息）- 使用实际存在的odom话题
        rospy.Subscriber(
            '/odom',                            # 使用JetRacer里程计话题
            Odometry,                           # 消息类型：里程计消息
            self._odom_callback,                # 里程计回调函数
            queue_size=10                       # 队列大小：1
        )
        
        # ===== 发布控制指令到JetRacer =====
        # 创建发布者，用于发送速度控制指令 - 使用实际存在的cmd_vel话题
        self._cmd_pub = rospy.Publisher(
            '/cmd_vel',                         # 直接使用/cmd_vel话题（不需要命名空间）
            Twist,                              # 消息类型：Twist包含线速度和角速度
            queue_size=10                       # 队列大小：1条消息
        )
    
    def _select_topic(self, candidates, default_topic: str) -> str:
        """从候选话题中选择当前可用的话题。"""
        try:
            topic_names = {name for name, _ in rospy.get_published_topics()}
            for topic in candidates:
                if topic in topic_names:
                    return topic
        except Exception:
            pass
        rospy.logwarn(f"未找到候选话题，使用默认: {default_topic}")
        return default_topic

    def _initialize_vlfm(self):
        """
        私有方法：初始化VLFM策略和相关组件
        功能：创建VLFM策略实例和障碍物地图
        """
        try:
            # 输出初始化开始的信息
            rospy.loginfo("初始化VLFM策略...")
            
            # ===== 创建VLFM策略配置对象 =====
            # OmegaConf.create() 创建配置对象，支持嵌套字典结构
            config = OmegaConf.create({
                "policy": {  # 策略配置子项
                    # 图像预处理方式：中心裁剪并调整大小
                    "obs_transform": "center_crop_resize",
                    # RGB图像输入尺寸：224x224（ViT Vision Transformer标准输入）
                    "rgb_image_size": 224,
                    # 深度图像输入尺寸：224x224（与PointNav模型训练时一致）
                    "depth_image_size": 224,
                    # 是否对深度数据进行归一化处理
                    "normalize_depth": True,
                    # COCO数据集中物体的检测置信度阈值（较严格）
                    "coco_threshold": 0.8,
                    # 非COCO物体的检测置信度阈值（较宽松）
                    "non_coco_threshold": 0.4,
                    # PointNav底层导航策略的预训练权重文件路径
                    "pointnav_policy_path": "data/pointnav_weights.pth",
                    # 深度图像的PointNav处理尺寸 [高度, 宽度] - 与训练时一致
                    "depth_image_shape": [224, 224],
                    # PointNav导航到目标时的停止距离（米）
                    "pointnav_stop_radius": 1.0,
                    # 对象地图的形态学腐蚀核大小（像素）
                    "object_map_erosion_size": 3.0,
                    # 启用可视化以获取policy_info
                    "visualize": True,
                    # 明确设置连续动作
                    "discrete_actions": False,
                    # JetRacer车身半径约6cm，影响障碍物膨胀核大小(3x3)
                    "agent_radius": 0.06,
                }
            })
            
            # ===== 创建VLFM策略实例 =====
            # 控制器主导模式（对应Habitat架构）：
            #   控制器 = Environment（负责所有ROS通信：传感器订阅、地图构建、指令发布）
            #   策略   = Policy（只负责导航算法：前沿选择、VLM评分、动作计算）
            self.policy = JetRacerROSITMPolicyV2(
                text_prompt="Seems like there is a target_object ahead.",
                **config.policy
            )
            
            # ===== 直接使用策略内部的障碍物地图 =====
            # 策略的ValueMap引用了self.policy._obstacle_map来同步explored_area
            # 控制器必须更新同一个对象，否则ValueMap的explored_area全为0导致value被清零
            self.obstacle_map = self.policy._obstacle_map
            # JetRacer很矮(~10cm车身)，可从桌椅下穿过
            # 只检测低矮障碍物(鞋子、墙脚、椅子腿等)，过滤高处物体
            self.obstacle_map._min_height = 0.05  # 5cm以上算障碍
            self.obstacle_map._max_height = 0.25  # 25cm以上的物体可从下方穿过
            rospy.loginfo(f"[ObstacleMap] kernel_size={self.obstacle_map._navigable_kernel.shape[0]}, "
                          f"min_h={self.obstacle_map._min_height}, max_h={self.obstacle_map._max_height}")

            # 输出初始化成功的信息
            rospy.loginfo("VLFM策略初始化成功")
            
        # 异常处理：捕获初始化过程中的任何错误
        except Exception as e:
            rospy.logerr(f"❌ VLFM初始化失败: {e}")
            import traceback
            rospy.logerr(f"详细错误: {traceback.format_exc()}")
            # 继续运行，但使用安全模式
            rospy.logwarn("继续运行基本功能（安全模式）")
            self.policy = None
            rospy.logwarn("VLFM初始化失败，将尝试基础导航模式")

    def _decode_rgb_image(self, msg: Image) -> np.ndarray:
        """将ROS RGB消息转换为RGB numpy数组。"""
        if msg.encoding == "bgr8":
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        if msg.encoding == "rgb8":
            return self.cv_bridge.imgmsg_to_cv2(msg, "rgb8")
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
        if len(cv_image.shape) == 3:
            return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)

    def _decode_depth_image(self, msg: Image) -> np.ndarray:
        """将ROS深度消息统一转换为米单位float32。"""
        if msg.encoding == "16UC1":
            depth_image = self.cv_bridge.imgmsg_to_cv2(msg, "16UC1")
            return depth_image.astype(np.float32) * self.camera_params["depth_scale"]
        if msg.encoding == "32FC1":
            return self.cv_bridge.imgmsg_to_cv2(msg, "32FC1")
        depth_raw = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
        if depth_raw.dtype == np.uint16:
            return depth_raw.astype(np.float32) * self.camera_params["depth_scale"]
        return depth_raw.astype(np.float32)

    def _rgbd_callback(self, rgb_msg: Image, depth_msg: Image):
        """时间同步后的RGB-D回调。"""
        try:
            self._current_rgb = self._decode_rgb_image(rgb_msg)
            self._current_depth = self._decode_depth_image(depth_msg)
            self._last_rgb_timestamp = rgb_msg.header.stamp.to_sec()
            self._last_depth_timestamp = depth_msg.header.stamp.to_sec()

            if self._current_rgb is not None and not hasattr(self, '_rgb_status_logged'):
                rospy.loginfo(f"📸 RGB相机连接成功: {self._current_rgb.shape}")
                self._rgb_status_logged = True
            if self._current_depth is not None and not hasattr(self, '_depth_status_logged'):
                rospy.loginfo(f"📐 深度传感器连接成功: {self._current_depth.shape}")
                self._depth_status_logged = True
        except Exception as e:
            rospy.logerr(f"同步RGB-D处理错误: {e}")

    def _rgb_callback(self, msg: Image):
        """
        RGB图像数据回调函数
        参数：msg - ROS Image消息对象，包含图像数据和元信息
        功能：将ROS图像消息转换为OpenCV格式并存储，支持多种图像编码格式
        """
        try:
            self._current_rgb = self._decode_rgb_image(msg)
            
            # 调试：检查图像内容
            if self._current_rgb is not None:
                # 记录RGB数据接收时间戳
                self._last_rgb_timestamp = msg.header.stamp.to_sec()
                
                # 记录图像接收状态
                if not hasattr(self, '_rgb_status_logged'):
                    rospy.loginfo(f"📸 RGB相机连接成功: {self._current_rgb.shape}")
                    self._rgb_status_logged = True
            
                
        except Exception as e:
            rospy.logerr(f"RGB处理错误: {e}")
            import traceback
            rospy.logerr(f"详细错误: {traceback.format_exc()}")

    def _depth_callback(self, msg: Image):
        """
        深度图像数据回调函数
        参数：msg - ROS Image消息，包含深度数据
        功能：处理不同编码格式的深度数据并转换为统一的米单位浮点格式
        """
        try:
            self._current_depth = self._decode_depth_image(msg)
            
            # ===== 深度数据验证和调试 =====
            if self._current_depth is not None:
                # 记录深度数据接收时间戳
                self._last_depth_timestamp = msg.header.stamp.to_sec()
                
                # 记录深度传感器状态
                if not hasattr(self, '_depth_status_logged'):
                    rospy.loginfo(f"📐 深度传感器连接成功: {self._current_depth.shape}")
                    self._depth_status_logged = True
            
        except Exception as e:
            rospy.logerr(f"深度处理错误: {e}")

    def _odom_callback(self, msg: Odometry):
        """里程计回调"""
        try:
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            
            # 四元数转yaw角
            yaw = math.atan2(
                2 * (ori.w * ori.z + ori.x * ori.y),
                1 - 2 * (ori.y * ori.y + ori.z * ori.z)
            )
            
            self._current_odom = {"x": pos.x, "y": pos.y, "yaw": yaw}
            # 某些里程计消息时间戳可能为0，回退到本地时间
            if msg.header.stamp.to_sec() > 0:
                self._last_odom_timestamp = msg.header.stamp.to_sec()
            else:
                self._last_odom_timestamp = rospy.Time.now().to_sec()
            
        except Exception as e:
            rospy.logerr(f"里程计处理错误: {e}")

    def _imu_callback(self, msg: Imu):
        """IMU回调"""
        try:
            # 简单提取yaw角
            ori = msg.orientation
            yaw = math.atan2(
                2 * (ori.w * ori.z + ori.x * ori.y),
                1 - 2 * (ori.y * ori.y + ori.z * ori.z)
            )
            self._current_imu = {"yaw": yaw}
            
        except Exception as e:
            rospy.logerr(f"IMU处理错误: {e}")

    def _create_vlfm_observations(self, robot_xy: np.ndarray, robot_heading: float) -> Optional[Dict[str, Any]]:
        """创建VLFM观测数据"""
        # 原子快照：确保整步计算使用同一对 RGB/Depth，防止回调中途更新
        rgb_snap = self._current_rgb
        depth_snap = self._current_depth
        rgb_ts = self._last_rgb_timestamp
        depth_ts = self._last_depth_timestamp

        if rgb_snap is None or depth_snap is None:
            rospy.logwarn_throttle(5, "传感器数据缺失: RGB或深度图像为None")
            return None

        # 检查传感器数据可用性
        current_time = rospy.Time.now().to_sec()
        rgb_age = current_time - rgb_ts
        depth_age = current_time - depth_ts

        if rgb_age > 1.0 or depth_age > 1.0:
            rospy.logwarn(f"传感器数据过期")

        # 检查 RGB 与 Depth 时间差（机器人停车期间采集，差值通常很小）
        ts_diff = abs(rgb_ts - depth_ts)
        if ts_diff > 0.2:
            rospy.logwarn_throttle(2, f"RGB/Depth时间差 {ts_diff*1000:.0f}ms")
        
        try:
            # robot_heading 由 _run_vlfm_step 传入，已经是统一后的episodic heading
            rospy.loginfo_throttle(1, f"[ANGLE] episodic_yaw={np.degrees(robot_heading):.1f}°")
            # tf_matrix 必须与 pointnav 使用同一heading语义，避免地图坐标系与控制坐标系不一致
            camera_height = 0.20  # JetRacer上RealSense的安装高度，需根据实际测量调整
            tf_matrix = xyz_yaw_to_tf_matrix(
                np.array([robot_xy[0], robot_xy[1], camera_height]),
                robot_heading,
            )
            
            # 归一化到 [0, 1]：与 update_map 内部反归一化公式一致
            # update_map: scaled = normalized * (max - min) + min
            # 正确逆变换: normalized = (depth - min) / (max - min)，无效像素(0)保持0
            depth_normalized = np.where(
                self._current_depth > 0,
                (np.clip(self._current_depth, self.DEPTH_MIN, self.DEPTH_MAX) - self.DEPTH_MIN)
                / (self.DEPTH_MAX - self.DEPTH_MIN),
                0.0,
            )

            # 更新障碍物地图
            if self.obstacle_map is not None:
                self.obstacle_map.update_map(
                    depth_normalized,
                    tf_matrix,
                    min_depth=self.DEPTH_MIN,
                    max_depth=self.DEPTH_MAX,
                    fx=self.camera_params["fx"],
                    fy=self.camera_params["fy"],
                    topdown_fov=self._topdown_fov_deg,
                    explore=True,
                    update_obstacles=True,
                )

                self.obstacle_map.update_agent_traj(robot_xy, robot_heading)
                frontiers = self.obstacle_map.frontiers

            else:
                frontiers = np.array([[0, 0]])

            # 创建相机到世界坐标系的变换矩阵
            tf_camera_to_episodic = tf_matrix

            # 构造VLFM观测
            observations = {
                "objectgoal": self.target_object,
                "rgb": torch.from_numpy(self._current_rgb).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0,
                "depth": torch.from_numpy(self._current_depth).unsqueeze(0).unsqueeze(0),
                "robot_xy": robot_xy,
                "robot_heading": robot_heading,
                "frontier_sensor": frontiers,
                "object_map_rgbd": [(
                    self._current_rgb,
                    depth_normalized,
                    tf_camera_to_episodic,
                    self.DEPTH_MIN,
                    self.DEPTH_MAX,
                    self.camera_params["fx"],
                    self.camera_params["fy"],
                )],
                "value_map_rgbd": [(
                    self._current_rgb,
                    depth_normalized,
                    tf_camera_to_episodic,
                    self.DEPTH_MIN,
                    self.DEPTH_MAX,
                    math.radians(self._topdown_fov_deg)  # fov (弧度)
                )]
            }
            
            
            return observations
            
        except Exception as e:
            import traceback
            rospy.logerr(f"观测创建错误: {e}")
            rospy.logerr(f"详细错误: {traceback.format_exc()}")
            return None

    # JetRacer深度范围（米）：远处噪声大，缩短可减少假障碍物
    DEPTH_MAX = 1.5
    DEPTH_MIN = 0.1

    def _run_vlfm_step(self) -> Optional[Twist]:
        """运行VLFM一步"""
        _step_t0 = time.time()
        # 提前定义机器人位置变量，确保在整个函数作用域可用
        raw_robot_xy = np.array([self._current_odom["x"], self._current_odom["y"]])
        robot_heading = self._current_odom["yaw"]

        # 统一使用归零后的坐标和朝向（避免value map越界）
        if hasattr(self, '_imu_offset'):
            # 显示和VLFM内部都使用归零后的坐标
            robot_xy = np.array([
                raw_robot_xy[0] - self._imu_offset["x"],
                raw_robot_xy[1] - self._imu_offset["y"]
            ])
            display_robot_xy = robot_xy
            # 朝向也归零（与xy一致）
            robot_heading = robot_heading - self._imu_offset["yaw"]
            robot_heading = np.arctan2(np.sin(robot_heading), np.cos(robot_heading))
            # 🔍 调试IMU偏移效果
            rospy.logdebug(f"IMU偏移调试:")
            rospy.logdebug(f"   原始坐标: ({raw_robot_xy[0]:.3f}, {raw_robot_xy[1]:.3f})")
            rospy.logdebug(f"   IMU偏移: ({self._imu_offset['x']:.3f}, {self._imu_offset['y']:.3f})")  
            rospy.logdebug(f"   归零坐标: ({robot_xy[0]:.3f}, {robot_xy[1]:.3f})")
        else:
            robot_xy = raw_robot_xy
            display_robot_xy = raw_robot_xy
            rospy.logdebug(f"IMU未初始化，使用原始坐标: ({robot_xy[0]:.3f}, {robot_xy[1]:.3f})")
        
        try:
            if self.policy is None:
                # 安全模式：返回简单的探索动作
                cmd = Twist()
                cmd.linear.x = 0.1  # 缓慢前进
                cmd.angular.z = 0.0
                rospy.loginfo_throttle(5, "安全模式：VLFM未初始化，使用简单动作")
                return cmd
            
            observations = self._create_vlfm_observations(robot_xy, robot_heading)
            if observations is None:
                return None
            
            # 第一次调用使用False mask来触发重置和目标设置
            if not hasattr(self, '_first_call_done'):
                masks = torch.zeros(1, 1, dtype=torch.bool)  # False触发重置
                self._first_call_done = True
                rospy.loginfo(f"开始导航，目标: {self.target_object}")
            else:
                masks = torch.ones(1, 1, dtype=torch.bool)  # 后续调用正常
            
            action_result = self.policy.act(
                observations=observations,
                rnn_hidden_states=None,
                prev_actions=None,
                masks=masks,
                deterministic=True
            )

            # # 诊断：每5步保存obstacle_map截图（暂时关闭排查问题）
            # if not hasattr(self, '_diag_step'):
            #     self._diag_step = 0
            #     os.makedirs('/tmp/vlfm_diag', exist_ok=True)
            # self._diag_step += 1
            # if self._diag_step % 5 == 0:
            #     try:
            #         if self.obstacle_map:
            #             obs_img = self.obstacle_map.visualize()
            #             cv2.imwrite(f'/tmp/vlfm_diag/obstacle_map_{self._diag_step:04d}.png', obs_img)
            #         rospy.loginfo_throttle(5, f"[DIAG] 截图已保存: step {self._diag_step}")
            #     except Exception as e:
            #         rospy.logwarn_throttle(10, f"截图保存失败: {e}")

            # JetRacerROSMixin.act返回字典格式
            if isinstance(action_result, dict):
                angular_vel = float(action_result["angular"])
                linear_vel = float(action_result["linear"])
            else:
                # 备用处理（如果是元组格式）
                action = action_result[0]
                angular_vel = float(action[0].item())
                linear_vel = float(action[1].item())
            
            # 转换为JetRacer格式
            # 注意：JetRacer可能有不同的速度范围，需要根据文档调整
            cmd = Twist()
            cmd.linear.x = np.clip(linear_vel, -0.8, 0.8)  # 限制线速度在±0.8m/s
            cmd.angular.z = np.clip(angular_vel, -0.5, 0.5)  # 限制角速度在±0.5rad/s
            
            # 输出关键导航信息
            cos_sim = action_result.get('cosine_similarity', 'N/A')
            best_value = action_result.get('best_value', 'N/A')
            
            # 格式化显示余弦相似度和最佳值
            cos_str = f", cos={cos_sim:.3f}" if isinstance(cos_sim, (int, float)) else ""
            value_str = f", best_value={best_value:.3f}" if isinstance(best_value, (int, float)) else ""
            
            # 获取策略内部的导航模式和停止原因
            policy_mode = "unknown"
            stop_reason = None
            if hasattr(self.policy, '_policy_info'):
                policy_info = self.policy._policy_info
                if 'mode' in policy_info:
                    policy_mode = policy_info['mode']
                if 'stop_reason' in policy_info:
                    stop_reason = policy_info['stop_reason']
            
            # 检测停止状态
            is_stopped = (cmd.linear.x == 0.0 and cmd.angular.z == 0.0)
            
            # 根据策略模式和停止原因显示对应的状态标志
            current_mode = getattr(self, '_current_mode', '')
            current_stop_reason = getattr(self, '_current_stop_reason', '')
            
            if current_mode != policy_mode:
                if policy_mode == "initialize":
                    rospy.loginfo("🔧 VLFM初始化中")
                elif policy_mode == "explore":
                    rospy.loginfo("🔍 探索模式：寻找目标物体")
                elif policy_mode == "navigate":
                    rospy.loginfo(f"🎯 导航模式：前往{self.target_object}")
                
                self._current_mode = policy_mode
            
            # 检测停止原因变化
            if stop_reason and current_stop_reason != stop_reason:
                if stop_reason == "target_reached":
                    rospy.loginfo(f"✅ 任务完成！成功到达{self.target_object}")
                    # 可以在这里添加episode结束逻辑
                    self.end_episode_recording(success=True)
                elif stop_reason == "exploration_complete":
                    rospy.loginfo(f"❌ 探索完成但未找到{self.target_object}")
                    # 可以在这里添加episode失败处理
                    self.end_episode_recording(success=False)
                
                self._current_stop_reason = stop_reason
            
            # 显示模式和动作信息，包含停止状态
            status_str = f" [STOPPED: {stop_reason}]" if is_stopped and stop_reason else ""
            rospy.loginfo(f"[{policy_mode}]{status_str} Action: linear={cmd.linear.x:.3f}, angular={cmd.angular.z:.3f}{cos_str}{value_str}")
            
            # ===== VLFM风格可视化处理 =====
            # 首先确认可视化条件
            vis_conditions = [
                ("enable_visualization", self.enable_visualization),
                ("current_rgb", self._current_rgb is not None),
                ("current_depth", self._current_depth is not None),
                ("action_result", action_result is not None),
                ("robot_xy", 'robot_xy' in locals()),
                ("robot_heading", 'robot_heading' in locals()),
                ("observations", 'observations' in locals())
            ]
            
            all_conditions_met = all(condition[1] for condition in vis_conditions)
            if not all_conditions_met:
                missing = [name for name, met in vis_conditions if not met]
                rospy.logwarn_throttle(10, f"可视化条件不满足: {missing}")
            
            if all_conditions_met:
                try:
                    self._update_vlfm_visualization(action_result, display_robot_xy, robot_heading, observations)
                except Exception as vis_e:
                    rospy.logwarn_throttle(5, f"❌ 可视化处理失败: {vis_e}")

            _step_dt = time.time() - _step_t0
            rospy.loginfo_throttle(3, f"[PERF] step耗时: {_step_dt*1000:.0f}ms")
            return cmd
            
        except Exception as e:
            import traceback
            rospy.logerr(f"VLFM运行错误: {e}")
            rospy.logerr(f"详细错误: {traceback.format_exc()}")
            
            # 即使出错也尝试收集可视化数据
            if (self.enable_visualization and 
                self._current_rgb is not None and 
                self._current_depth is not None):
                try:
                    # 创建一个默认的action_result用于可视化
                    default_action = {"linear": 0.0, "angular": 0.0}
                    # 创建默认观测数据
                    default_observations = {"objectgoal": self.target_object}
                    self._update_vlfm_visualization(default_action, robot_xy, robot_heading, default_observations)
                except Exception as vis_e:
                    rospy.logwarn_throttle(10, f"错误时可视化处理失败: {vis_e}")
            
            # 返回停止指令
            return Twist()

    def set_target(self, target: str):
        """设置目标对象"""
        self.target_object = target
        rospy.loginfo(f"目标设置为: {target}")

    def run(self):
        """运行主控制循环"""
        rate = rospy.Rate(5)  # 5Hz, 与测试版本一致，更稳定
        
        rospy.loginfo(f"🚀 开始VLFM导航，目标: {self.target_object}")
        rospy.loginfo("📷 等待传感器数据...")
        
        # 开始episode记录
        self.start_episode_recording()
        
        # 等待传感器数据就绪
        sensor_ready = False
        wait_count = 0
        while not rospy.is_shutdown() and not sensor_ready and wait_count < 50:  # 最多等待10秒
            if (
                self._current_rgb is not None
                and self._current_depth is not None
                and self._last_odom_timestamp > 0.0
            ):
                sensor_ready = True
                rospy.loginfo("传感器数据就绪！")
                # 记录并重置IMU参考原点
                initial_x = self._current_odom["x"]
                initial_y = self._current_odom["y"] 
                initial_yaw = self._current_odom["yaw"]
                
                # 设置IMU偏移，让项目运行时的起始位置显示为(0, 0)
                self._imu_offset = {"x": initial_x, "y": initial_y, "yaw": initial_yaw}
                
                rospy.loginfo(f"小车开机累积位置: x={initial_x:.6f}, y={initial_y:.6f}, yaw={initial_yaw:.6f}")
                rospy.loginfo(f"项目运行起始位置已重置为: (0.000, 0.000), yaw=0.0°")
                rospy.loginfo(f"🚀 IMU参考原点设置完成")
                
                # 取消强制地图中心设置，使用BaseMap的默认逻辑
                # IMU已经归零，直接使用原始坐标即可
                rospy.loginfo(f"📍 使用BaseMap默认地图中心，IMU已归零")
                rospy.loginfo(f"使用标准地图坐标系，开始绘制地图")
            else:
                wait_count += 1
                rospy.sleep(0.2)
        
        if not sensor_ready:
            rospy.logwarn("传感器数据等待超时，继续运行...")
        
        try:
            while not rospy.is_shutdown():
                try:
                    if self._current_rgb is None:
                        rospy.logwarn_throttle(5, "等待RGB相机数据...")
                        rate.sleep()
                        continue

                    # 初始化阶段：连续执行（转满一圈）
                    # 探索/导航阶段：脉冲式执行（停车→计算→跑0.3s→停车）
                    is_initializing = not getattr(self.policy, '_done_initializing', False)

                    if not is_initializing:
                        self._cmd_pub.publish(Twist())  # 计算前停车

                    cmd = self._run_vlfm_step()

                    if getattr(self, '_current_stop_reason', None) == "target_reached":
                        self._cmd_pub.publish(Twist())   # 确保发布停止指令
                        break                            

                    if cmd is not None:
                        self._cmd_pub.publish(cmd)
                        if is_initializing:
                            rate.sleep()  # 初始化：保持原来的连续转圈节奏
                        else:
                            rospy.sleep(0.3)  # 探索/导航：跑0.3s后停车
                            self._cmd_pub.publish(Twist())

                except Exception as e:
                    rospy.logerr(f"主循环错误: {e}")
                    import traceback
                    rospy.logerr(f"详细错误: {traceback.format_exc()}")
                    # 发送停止指令并尝试恢复
                    self._cmd_pub.publish(Twist())
                    rospy.logwarn("尝试从错误中恢复...")
                    rate.sleep()
        except KeyboardInterrupt:
            rospy.loginfo("收到Ctrl+C，正在保存视频并退出...")
        finally:
            # 停止机器人
            self._cmd_pub.publish(Twist())

            # 结束episode记录（无论正常退出还是Ctrl+C都会执行）
            self.end_episode_recording(success=False)

            # 清理资源
            cv2.destroyAllWindows()
            rospy.loginfo("清理资源完成")

    def _update_vlfm_visualization(
        self, 
        action_result: Dict[str, Any], 
        robot_xy: np.ndarray, 
        robot_heading: float,
        observations: Dict[str, Any]
    ):
        """更新VLFM风格的可视化显示"""
        try:
            
            # 获取障碍物地图和前沿点（需要在策略信息获取之前）
            obstacle_map_data = None
            explored_area_data = None
            frontiers = []
            if self.obstacle_map:
                # 获取可导航地图和已探索区域信息
                obstacle_map_data = self.obstacle_map._navigable_map
                explored_area_data = self.obstacle_map.explored_area
                
                # 获取frontiers并统一应用IMU偏移
                raw_frontiers = self.obstacle_map.frontiers
                if len(raw_frontiers) > 0 and hasattr(self, '_imu_offset'):
                    # VLFM内部和显示都使用归零后的frontier坐标
                    frontiers = []
                    for frontier in raw_frontiers:
                        offset_frontier = np.array([
                            frontier[0] - self._imu_offset["x"],
                            frontier[1] - self._imu_offset["y"]
                        ])
                        frontiers.append(offset_frontier)
                    frontiers = np.array(frontiers)
                    display_frontiers = frontiers
                else:
                    frontiers = raw_frontiers
                    display_frontiers = frontiers
            
            # 获取策略信息（包含可视化数据）
            # act()内部已调用_get_policy_info()并存入_policy_info，直接使用action_result["info"]
            policy_info = action_result.get("info", {})
            rospy.loginfo_throttle(5, f"[VIS_DIAG] policy_info keys: {list(policy_info.keys())}")
            
            # 获取余弦相似度（如果有的话）
            cosine_similarity = action_result.get('cosine_similarity', None)
            
            # 确保坐标系一致性：都使用归零后的坐标系
            # robot_xy已经是归零坐标，frontiers也应该是归零坐标
            # 但地图数据是基于原始坐标构建的，需要匹配
            
            # 收集可视化数据（参考HabitatVis.collect_data）
            self.visualizer.collect_data(
                rgb_image=self._current_rgb,
                depth_image=self._current_depth,
                robot_xy=robot_xy,  # 归零坐标系
                robot_heading=robot_heading,
                action_info=action_result,
                policy_info=policy_info,
                target_object=self.target_object,
                obstacle_map=obstacle_map_data,  # 原始坐标系的地图
                explored_area=explored_area_data,  # 原始坐标系的地图
                frontiers=display_frontiers,  # 归零坐标系的frontiers
                cosine_similarity=cosine_similarity
            )
            
            # 实时显示已关闭

        except Exception as e:
            import traceback
            rospy.logwarn_throttle(10, f"VLFM可视化更新失败: {e}\n{traceback.format_exc()}")
    
    def _show_realtime_visualization(self):
        """显示实时可视化（可选功能）"""
        try:
            # 创建一个简单的实时预览
            if len(self.visualizer.rgb) > 0:
                latest_rgb = self.visualizer.rgb[-1]
                cv2.imshow('VLFM JetRacer Navigation', latest_rgb)
                cv2.waitKey(1)
        except Exception as e:
            rospy.logwarn_throttle(30, f"实时显示失败: {e}")
    
    def start_episode_recording(self):
        """开始新episode的记录"""
        self.episode_start_time = rospy.Time.now()
        self.visualizer.reset()
        self._episode_recording_active = True
        self._episode_video_saved = False
        rospy.loginfo(f"开始记录新episode: {self.target_object}")
    
    def end_episode_recording(self, success: bool = False):
        """结束episode并保存视频"""
        if not self._episode_recording_active:
            rospy.loginfo("Episode未处于录制状态，跳过保存")
            return

        if self._episode_video_saved:
            rospy.loginfo("Episode视频已保存过，跳过重复保存")
            self._episode_recording_active = False
            return

        rospy.loginfo(f"📹 结束Episode录制，总共收集了 {self.visualizer.frame_count} 帧数据")
        
        if self.save_video:
            try:
                # 确定episode结果
                failure_cause = "Success" if success else "Episode Complete"

                # 兜底：若异常路径导致未收集任何帧，补一帧保证可落盘
                if self.visualizer.frame_count == 0 and self._current_rgb is not None and self._current_depth is not None:
                    rospy.logwarn("未收集到可视化帧，使用当前传感器数据补1帧")
                    self.visualizer.collect_data(
                        rgb_image=self._current_rgb,
                        depth_image=self._current_depth,
                        robot_xy=np.array([self._current_odom["x"], self._current_odom["y"]]),
                        robot_heading=float(self._current_odom["yaw"]),
                        action_info={"linear": 0.0, "angular": 0.0},
                        policy_info={},
                        target_object=self.target_object,
                        obstacle_map=self.obstacle_map._navigable_map if self.obstacle_map is not None else None,
                        explored_area=self.obstacle_map.explored_area if self.obstacle_map is not None else None,
                        frontiers=self.obstacle_map.frontiers if self.obstacle_map is not None else np.array([]),
                        cosine_similarity=None,
                    )
                
                # 生成视频帧
                frames = self.visualizer.flush_frames(failure_cause)
                
                if frames and len(frames) > 0:
                    # 计算episode指标
                    episode_duration = (rospy.Time.now() - self.episode_start_time).to_sec() if self.episode_start_time else 0
                    metrics = {
                        'duration': episode_duration,
                        'frames': len(frames),
                        'success': 1.0 if success else 0.0
                    }
                    
                    # 保存视频
                    episode_id = f"{self.target_object}_{int(rospy.Time.now().to_sec())}"
                    video_path = self.video_recorder.save_episode_video(
                        frames=frames,
                        episode_id=episode_id,
                        metrics=metrics
                    )
                    
                    if video_path:
                        rospy.loginfo(f"Episode视频已保存: {video_path}")
                        self._episode_video_saved = True
                    else:
                        rospy.logwarn("❌ 视频保存失败：无文件路径返回")
                else:
                    rospy.logwarn(f"❌ 没有视频帧可保存 (frames={len(frames) if frames else 0})")
                        
            except Exception as e:
                import traceback
                rospy.logerr(f"❌ Episode视频保存失败: {e}")
                rospy.logerr(f"详细错误: {traceback.format_exc()}")
        else:
            rospy.loginfo("📹 视频保存已禁用，跳过保存")

        self._episode_recording_active = False

    def stop(self):
        """停止机器人"""
        self._cmd_pub.publish(Twist())
        cv2.destroyAllWindows()
        rospy.loginfo("机器人已停止")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="chair", help="目标对象")
    parser.add_argument("--namespace", default="", help="JetRacer命名空间")
    args = parser.parse_args()
    
    try:
        controller = VLFMJetRacerController(jetracer_namespace=args.namespace)
        
        if args.target:
            controller.set_target(args.target)
        
        controller.run()
        
    except KeyboardInterrupt:
        rospy.loginfo("用户中断")
    except Exception as e:
        rospy.logerr(f"程序错误: {e}")


if __name__ == '__main__':
    main()
