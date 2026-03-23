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
import rospy  # ROS的Python API，用于创建节点、订阅发布话题
import numpy as np  # NumPy数值计算库，用于数组和矩阵操作
import torch  # PyTorch深度学习框架，用于张量运算和神经网络
import math  # 数学函数库，提供三角函数、对数等数学运算
from typing import Dict, Any, Optional  # 类型注解，Dict字典类型，Any任意类型，Optional可选类型
import cv2  # OpenCV计算机视觉库，用于图像处理
from cv_bridge import CvBridge  # ROS与OpenCV之间的图像格式转换工具

# ===== 导入ROS消息类型 =====
from sensor_msgs.msg import Image, Imu  # ROS传感器消息：Image图像消息，Imu惯导消息
from nav_msgs.msg import Odometry  # ROS导航消息：里程计消息，包含位置和速度
from geometry_msgs.msg import Twist  # ROS几何消息：速度指令，包含线速度和角速度

# ===== 添加模块路径 =====
import sys
import os

# 添加vlfm项目根目录到Python路径，确保能够导入vlfm模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ===== 导入VLFM策略模块 =====
from vlfm.policy.jetracer_ros_policy import JetRacerROSITMPolicyV2  # VLFM的JetRacer ROS策略V2版本
from vlfm.mapping.obstacle_map import ObstacleMap  # VLFM障碍物地图构建模块
from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix  # VLFM几何工具：坐标变换矩阵计算
from vlfm.utils.jetracer_visualization import (
    VLFMVisualizer,
    VLFMVideoRecorder,
)  # VLFM可视化工具
from omegaconf import OmegaConf  # 配置管理库：OmegaConf配置工厂


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
        rospy.init_node("vlfm_jetracer_controller", anonymous=True)

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

        # ===== VLFM算法组件 =====
        # VLFM策略对象：Optional[JetRacerROSITMPolicyV2] 表示可以是JetRacerROSITMPolicyV2对象或None
        # 初始化时设为None，后续在_initialize_vlfm()中创建
        self.policy: Optional[JetRacerROSITMPolicyV2] = None

        # 障碍物地图对象：用于构建环境的占用栅格地图
        self.obstacle_map: Optional[ObstacleMap] = None

        # ===== 可视化组件 =====
        self.visualizer = VLFMVisualizer()
        self.video_recorder = None
        self.enable_visualization = True
        self.save_video = True  # 默认启用视频保存
        self.frame_count = 0
        self.video_segment = 0  # 视频段计数
        self.frames_per_video = 100  # 每个视频段的帧数

        # 目标物体名称：机器人要寻找的目标，默认为"chair"（椅子）
        self.target_object = "chair"

        # ===== Intel RealSense D435i相机参数字典 =====
        self.camera_params = {
            # fx, fy: 焦距参数（像素单位），需要通过相机标定获得准确值
            "fx": 616.0,
            "fy": 616.0,
            # cx, cy: 主点坐标（像素单位），通常在图像中心附近
            "cx": 320.0,
            "cy": 240.0,
            # width, height: 图像分辨率（像素）
            "width": 640,
            "height": 480,
            # depth_scale: 深度数据单位转换因子，RealSense输出毫米，转换为米
            "depth_scale": 0.001,
        }

        # 调用方法设置ROS通信接口（订阅者和发布者）
        self._setup_ros_interface()

        # 调用方法初始化VLFM策略和相关组件
        self._initialize_vlfm()

        # 使用ROS日志系统输出信息级别的消息
        rospy.loginfo("VLFM JetRacer控制器启动完成")

    def _setup_ros_interface(self):
        """
        私有方法：设置ROS通信接口
        功能：创建订阅者（接收数据）和发布者（发送数据）
        """

        # ===== 订阅RGB相机数据（灵活话题检测）=====
        # 常见的相机话题名称列表
        possible_rgb_topics = [
            "/camera/color/image_raw",  # RealSense标准RGB话题
            "/csi_cam_0/image_raw",  # JetRacer CSI相机
            "/camera/image_raw",  # 通用相机格式1
            "/usb_cam/image_raw",  # USB相机格式
            "/csi_cam/image_raw",  # CSI相机格式
        ]

        # 尝试找到可用的RGB话题
        rgb_topic_found = False
        for topic in possible_rgb_topics:
            try:
                # 检查话题是否存在
                topics = rospy.get_published_topics()
                if any(topic in t[0] for t in topics):
                    rospy.loginfo(f"找到RGB话题: {topic}")
                    rospy.Subscriber(
                        topic,  # 话题名称
                        Image,  # 消息类型：ROS的Image消息类型
                        self._rgb_callback,  # 回调函数：收到消息时调用的方法
                        queue_size=1,  # 队列大小：只保留最新1条消息
                        buff_size=2**24,  # 缓冲区大小：16MB，防止图像数据丢失
                    )
                    rgb_topic_found = True
                    break
            except Exception:
                continue

        if not rgb_topic_found:
            # 如果没找到，使用默认话题
            default_topic = "/camera/color/image_raw"
            rospy.logwarn(f"未找到RGB话题，使用默认: {default_topic}")
            rospy.Subscriber(
                default_topic,  # 默认话题
                Image,  # 消息类型
                self._rgb_callback,  # 回调函数
                queue_size=1,  # 队列大小
                buff_size=2**24,  # 缓冲区大小
            )

        # 订阅矫正深度图像话题 - 使用实际存在的话题
        rospy.Subscriber(
            "/camera/depth/image_rect_raw",  # 矫正后的深度图像话题
            Image,  # 消息类型：Image
            self._depth_callback,  # 深度图像回调函数
            queue_size=1,  # 队列大小：1
            buff_size=2**24,  # 缓冲区大小：16MB
        )

        # 订阅IMU惯性测量单元数据 - 使用JetRacer的IMU话题
        rospy.Subscriber(
            "/imu",  # JetRacer IMU话题
            Imu,  # 消息类型：IMU消息，包含姿态和加速度
            self._imu_callback,  # IMU数据回调函数
            queue_size=1,  # 队列大小：1（IMU数据较小，不需要大缓冲区）
        )

        # ===== 订阅JetRacer底盘数据 =====
        # 订阅里程计数据（机器人位置和速度信息）- 使用实际存在的odom话题
        rospy.Subscriber(
            "/odom",  # 直接使用/odom话题（不需要命名空间）
            Odometry,  # 消息类型：里程计消息
            self._odom_callback,  # 里程计回调函数
            queue_size=1,  # 队列大小：1
        )

        # ===== 发布控制指令到JetRacer =====
        # 创建发布者，用于发送速度控制指令 - 使用实际存在的cmd_vel话题
        self._cmd_pub = rospy.Publisher(
            "/cmd_vel",  # 直接使用/cmd_vel话题（不需要命名空间）
            Twist,  # 消息类型：Twist包含线速度和角速度
            queue_size=1,  # 队列大小：1条消息
        )

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
            config = OmegaConf.create(
                {
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
                        # 是否启用可视化（主机端设为False节省资源）
                        "visualize": False,
                        # 明确设置连续动作
                        "discrete_actions": False,
                    }
                }
            )

            # ===== 创建VLFM策略实例 =====
            # JetRacerROSITMPolicyV2 需要text_prompt参数
            # 使用正确的提示格式，包含target_object占位符
            self.policy = JetRacerROSITMPolicyV2(
                text_prompt="Seems like there is a target_object ahead.", **config.policy
            )

            # ===== 创建障碍物地图实例 =====
            self.obstacle_map = ObstacleMap(
                # 障碍物最小高度：0.15米（过滤桌子以下的物体）
                min_height=0.15,
                # 障碍物最大高度：0.88米（过滤人的高度以上的物体）
                max_height=0.88,
                # 机器人半径：0.18米（JetRacer的大致半径，用于路径规划）
                agent_radius=0.18,
                # 最小障碍物区域阈值：1.5平方米（过滤小的噪声区域）
                area_thresh=1.5,
            )

            # 输出初始化成功的信息
            rospy.loginfo("VLFM策略初始化成功")

        # 异常处理：捕获初始化过程中的任何错误
        except Exception as e:
            rospy.logerr(f"❌ VLFM初始化失败: {e}")
            import traceback

            rospy.logerr(f"详细错误: {traceback.format_exc()}")
            # 继续运行，但使用安全模式
            rospy.logwarn("🔧 继续运行基本功能（安全模式）")
            self.policy = None
            rospy.logwarn("🔄 VLFM初始化失败，将尝试基础导航模式")

    def _rgb_callback(self, msg: Image):
        """
        RGB图像数据回调函数
        参数：msg - ROS Image消息对象，包含图像数据和元信息
        功能：将ROS图像消息转换为OpenCV格式并存储，支持多种图像编码格式
        """
        try:
            if msg.encoding == "bgr8":
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
                self._current_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            elif msg.encoding == "rgb8":
                self._current_rgb = self.cv_bridge.imgmsg_to_cv2(msg, "rgb8")
            else:
                # 尝试自动转换
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
                if len(cv_image.shape) == 3:
                    self._current_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                else:
                    # 灰度图转RGB
                    self._current_rgb = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)

            # 调试：检查图像内容
            if self._current_rgb is not None:
                # 记录RGB数据接收时间戳
                self._last_rgb_timestamp = rospy.Time.now().to_sec()

                # 记录图像接收状态
                if not hasattr(self, "_rgb_status_logged"):
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
            # 根据消息的编码格式进行不同处理
            # encoding属性表示图像数据的编码方式

            if msg.encoding == "16UC1":
                # ===== RealSense D435i标准格式处理 =====
                # "16UC1": 16位无符号整数，单通道，单位为毫米
                depth_image = self.cv_bridge.imgmsg_to_cv2(msg, "16UC1")

                # 数据类型转换和单位转换
                # .astype(np.float32): 转换为32位浮点数（节省内存，提高计算效率）
                # * self.camera_params["depth_scale"]: 毫米转米 (depth_scale = 0.001)
                self._current_depth = depth_image.astype(np.float32) * self.camera_params["depth_scale"]

            elif msg.encoding == "32FC1":
                # ===== 已经是浮点格式的深度数据 =====
                # "32FC1": 32位浮点数，单通道，通常已经是米单位
                self._current_depth = self.cv_bridge.imgmsg_to_cv2(msg, "32FC1")

            else:
                # ===== 处理其他未知格式 =====
                # "passthrough": 保持原始数据格式不变
                depth_raw = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")

                # 根据数据类型判断处理方式
                if depth_raw.dtype == np.uint16:
                    # 如果是16位无符号整数，假设单位为毫米
                    self._current_depth = depth_raw.astype(np.float32) * self.camera_params["depth_scale"]
                else:
                    # 其他情况直接转为浮点数
                    self._current_depth = depth_raw.astype(np.float32)

            # ===== 深度数据验证和调试 =====
            if self._current_depth is not None:
                # 记录深度数据接收时间戳
                self._last_depth_timestamp = rospy.Time.now().to_sec()

                # 记录深度传感器状态
                if not hasattr(self, "_depth_status_logged"):
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
            yaw = math.atan2(2 * (ori.w * ori.z + ori.x * ori.y), 1 - 2 * (ori.y * ori.y + ori.z * ori.z))

            self._current_odom = {"x": pos.x, "y": pos.y, "yaw": yaw}

        except Exception as e:
            rospy.logerr(f"里程计处理错误: {e}")

    def _imu_callback(self, msg: Imu):
        """IMU回调"""
        try:
            # 简单提取yaw角
            ori = msg.orientation
            yaw = math.atan2(2 * (ori.w * ori.z + ori.x * ori.y), 1 - 2 * (ori.y * ori.y + ori.z * ori.z))
            self._current_imu = {"yaw": yaw}

        except Exception as e:
            rospy.logerr(f"IMU处理错误: {e}")

    def _create_vlfm_observations(self) -> Optional[Dict[str, Any]]:
        """创建VLFM观测数据"""
        if self._current_rgb is None or self._current_depth is None:
            rospy.logwarn_throttle(5, "⚠️ 传感器数据缺失: RGB或深度图像为None")
            return None

        # 检查传感器数据可用性
        current_time = rospy.Time.now().to_sec()
        rgb_age = current_time - self._last_rgb_timestamp
        depth_age = current_time - self._last_depth_timestamp

        if rgb_age > 1.0 or depth_age > 1.0:
            rospy.logwarn("⚠️ 传感器数据过期")

        # 获取机器人位置（移到try块外面，确保可视化能访问）
        robot_xy = np.array([self._current_odom["x"], self._current_odom["y"]])
        robot_heading = self._current_odom["yaw"]

        try:

            # 更新障碍物地图
            if self.obstacle_map is not None:
                tf_matrix = xyz_yaw_to_tf_matrix(np.array([robot_xy[0], robot_xy[1], 0.0]), robot_heading)

                depth_for_map = self._current_depth

                self.obstacle_map.update_map(
                    depth_for_map,
                    tf_matrix,
                    min_depth=0.1,
                    max_depth=5.0,  # 增加最大深度范围
                    fx=self.camera_params["fx"],
                    fy=self.camera_params["fy"],
                    topdown_fov=90.0,  # 添加视野角度参数
                    explore=True,
                )

                self.obstacle_map.update_agent_traj(robot_xy, robot_heading)
                frontiers = self.obstacle_map.frontiers

                rospy.loginfo_throttle(
                    10, f"状态: 探索中, 位置: ({robot_xy[0]:.2f}, {robot_xy[1]:.2f}), 前沿: {len(frontiers)}"
                )
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
                "object_map_rgbd": [
                    (
                        self._current_rgb,
                        self._current_depth,
                        tf_camera_to_episodic,
                        0.1,  # min_depth
                        5.0,  # max_depth
                        90.0,  # fov
                    )
                ],
                "value_map_rgbd": [
                    (
                        self._current_rgb,
                        self._current_depth,
                        tf_camera_to_episodic,
                        0.1,  # min_depth
                        5.0,  # max_depth
                        90.0,  # fov
                    )
                ],
            }

            # 显示导航状态
            rospy.loginfo_throttle(10, f"导航中: 目标={observations['objectgoal']}, 前沿={len(frontiers)}")

            return observations

        except Exception as e:
            import traceback

            rospy.logerr(f"观测创建错误: {e}")
            rospy.logerr(f"详细错误: {traceback.format_exc()}")
            return None

    def _run_vlfm_step(self) -> Optional[Twist]:
        """运行VLFM一步"""
        try:
            if self.policy is None:
                # 安全模式：返回简单的探索动作
                cmd = Twist()
                cmd.linear.x = 0.1  # 缓慢前进
                cmd.angular.z = 0.0
                rospy.loginfo_throttle(5, "🔧 安全模式：VLFM未初始化，使用简单动作")
                return cmd

            observations = self._create_vlfm_observations()
            if observations is None:
                return None

            # 第一次调用使用False mask来触发重置和目标设置
            if not hasattr(self, "_first_call_done"):
                masks = torch.zeros(1, 1, dtype=torch.bool)  # False触发重置
                self._first_call_done = True
                rospy.loginfo(f"开始导航，目标: {self.target_object}")
            else:
                masks = torch.ones(1, 1, dtype=torch.bool)  # 后续调用正常

            action_result = self.policy.act(
                observations=observations, rnn_hidden_states=None, prev_actions=None, masks=masks, deterministic=True
            )

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

            # 获取cosine相似度值（如果有的话）
            cos_sim = action_result.get("cosine_similarity", "N/A")
            if cos_sim != "N/A" and isinstance(cos_sim, (int, float)):
                rospy.loginfo(f"动作输出: 线速度={cmd.linear.x:.3f}, 角速度={cmd.angular.z:.3f}, cos值={cos_sim:.3f}")
            else:
                rospy.loginfo(f"动作输出: 线速度={cmd.linear.x:.3f}, 角速度={cmd.angular.z:.3f}")

            return cmd

        except Exception as e:
            import traceback

            rospy.logerr(f"VLFM运行错误: {e}")
            rospy.logerr(f"详细错误: {traceback.format_exc()}")
            # 返回停止指令
            cmd = Twist()

        # ===== 可视化处理（在try-except外面，确保变量可访问） =====
        try:
            # 检查可视化条件
            vis_enabled = self.enable_visualization
            rgb_available = self._current_rgb is not None
            action_available = "action_result" in locals() and action_result is not None

            if vis_enabled and rgb_available and action_available:
                robot_xy = np.array([self._current_odom["x"], self._current_odom["y"]], dtype=np.float32)
                robot_heading = float(self._current_odom["yaw"])
                self._update_visualization(action_result, robot_xy, robot_heading)

        except Exception as vis_e:
            rospy.logwarn_throttle(10, f"可视化处理失败: {vis_e}")

        return cmd if "cmd" in locals() else Twist()

    def set_target(self, target: str):
        """设置目标对象"""
        self.target_object = target
        rospy.loginfo(f"目标设置为: {target}")

    def run(self):
        """运行主控制循环"""
        rate = rospy.Rate(5)  # 5Hz, 与测试版本一致，更稳定

        rospy.loginfo(f"🚀 开始VLFM导航，目标: {self.target_object}")
        rospy.loginfo("📷 等待传感器数据...")

        # 等待传感器数据就绪
        sensor_ready = False
        wait_count = 0
        while not rospy.is_shutdown() and not sensor_ready and wait_count < 50:  # 最多等待10秒
            if self._current_rgb is not None and self._current_depth is not None:
                sensor_ready = True
                rospy.loginfo("✅ 传感器数据就绪！")
            else:
                wait_count += 1
                rospy.sleep(0.2)

        if not sensor_ready:
            rospy.logwarn("⚠️ 传感器数据等待超时，继续运行...")

        while not rospy.is_shutdown():
            try:
                if self._current_rgb is None:
                    rospy.logwarn_throttle(5, "等待RGB相机数据...")
                    rate.sleep()
                    continue

                # 运行VLFM
                cmd = self._run_vlfm_step()

                if cmd is not None:
                    # 发送控制指令
                    self._cmd_pub.publish(cmd)

                rate.sleep()

            except Exception as e:
                rospy.logerr(f"主循环错误: {e}")
                import traceback

                rospy.logerr(f"详细错误: {traceback.format_exc()}")
                # 发送停止指令并尝试恢复
                self._cmd_pub.publish(Twist())
                rospy.logwarn("🔄 尝试从错误中恢复...")
                rate.sleep()

        # 停止机器人
        self._cmd_pub.publish(Twist())

        # 清理可视化
        cv2.destroyAllWindows()

        # 确保最后的视频段被保存
        if self.save_video and self.video_recorder is not None:
            self.video_recorder.stop_recording()

    def _update_visualization(self, action_result: Dict[str, Any], robot_xy: np.ndarray, robot_heading: float):
        """更新可视化显示 - 参考Habitat的video_option="[disk]"方式"""
        try:
            if self._current_rgb is not None:
                # 获取地图信息
                frontiers = self.obstacle_map.frontiers if self.obstacle_map else []
                obstacle_map = self.obstacle_map._navigable_map if self.obstacle_map else None

                # 创建完整可视化帧（类似Habitat的top_down_map + rgb观测）
                vis_frame = self.visualizer.create_navigation_frame(
                    rgb_image=self._current_rgb,
                    depth_image=self._current_depth,
                    obstacle_map=obstacle_map,
                    frontiers=frontiers,
                    robot_xy=robot_xy,
                    robot_heading=robot_heading,
                    target_object=self.target_object,
                    action_info=action_result,
                    policy_info=action_result.get("info", {}),
                )

                # 实时显示
                cv2.imshow("VLFM Navigation", vis_frame)
                cv2.waitKey(1)

                # 视频保存（仿照Habitat的episode视频保存方式）
                self._save_video_frame(vis_frame)

        except Exception as e:
            rospy.logwarn_throttle(10, f"可视化更新失败: {e}")

    def _save_video_frame(self, vis_frame: np.ndarray):
        """保存视频帧 - 每100帧保存一个视频段（类似Habitat的episode录制）"""
        # 计算当前在第几段视频的第几帧
        frames_in_current_segment = self.frame_count % self.frames_per_video

        # 如果是新段的开始，创建新的视频文件
        if frames_in_current_segment == 0:
            # 先结束上一个视频（如果有）
            if self.video_recorder is not None:
                self.video_recorder.stop_recording()

            # 开始新的视频段
            self.video_segment += 1
            timestamp = rospy.get_time()
            video_filename = f"/tmp/vlfm_episode_{self.video_segment:03d}_{int(timestamp)}.mp4"

            self.video_recorder = VLFMVideoRecorder(video_filename, fps=10)  # 10fps类似Habitat
            h, w = vis_frame.shape[:2]
            self.video_recorder.start_recording((h, w))

            pass  # 静默开始录制

        # 添加当前帧
        if self.save_video and self.video_recorder is not None:
            self.video_recorder.add_frame(vis_frame)

        self.frame_count += 1

        # 静默完成录制

    def stop_video_recording(self):
        """停止录制视频"""
        if self.video_recorder is not None:
            self.video_recorder.stop_recording()
            self.video_recorder = None
            self.save_video = False

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


if __name__ == "__main__":
    main()
