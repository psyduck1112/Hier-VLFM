# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
# Modifications Copyright (c) 2026 Yikang.

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

# ROS imports
try:
    import rospy
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import Image as RosImage, Imu
    from nav_msgs.msg import Odometry
    from cv_bridge import CvBridge
    import cv2
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("Warning: ROS not available. JetRacer functionality will be limited.")

# VLFM项目内部模块
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.policy.base_objectnav_policy import VLFMConfig
from vlfm.policy.itm_policy import ITMPolicyV2
from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix


class JetRacerROSMixin:
    """
    JetRacer ROS混入类
    提供通过ROS与JetRacer + RealSense D435i通信的接口
    使用非Habitat PointNav实现以支持连续动作控制
    """

    # 策略配置
    _stop_action: Tensor = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    _load_yolo: bool = True
    _non_coco_caption: str = (
        "chair . table . tv . laptop . microwave . toaster . sink . refrigerator . book"
        " . clock . vase . scissors . teddy bear . hair drier . toothbrush ."
    )
    
    @staticmethod
    def _disable_habitat_for_pointnav():
        """临时禁用Habitat检测，强制使用非Habitat PointNav实现"""
        import sys
        # 临时移除habitat_baselines模块，使WrappedPointNavResNetPolicy使用非Habitat分支
        habitat_modules = [k for k in sys.modules.keys() if k.startswith('habitat')]
        removed_modules = {}
        for mod in habitat_modules:
            removed_modules[mod] = sys.modules.pop(mod, None)
        return removed_modules
    
    @staticmethod  
    def _restore_habitat_modules(removed_modules):
        """恢复Habitat模块"""
        import sys
        for mod, obj in removed_modules.items():
            if obj is not None:
                sys.modules[mod] = obj
    
    # 观测数据缓存
    _observations_cache: Dict[str, Any] = {}
    _policy_info: Dict[str, Any] = {}
    _done_initializing: bool = False  # 初始化完成标志
    _initial_yaws: List[float] = []   # 初始化偏航角序列
    
    # ROS状态
    _ros_initialized: bool = False
    _data_lock = threading.Lock()
    
    # 传感器数据
    _current_rgb: np.ndarray = None
    _current_depth: np.ndarray = None
    _current_odom: Dict[str, float] = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    _imu_data: Dict[str, float] = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
    _last_data_time: float = 0.0
    
    # RealSense D435i 默认相机参数
    _camera_params = {
        "fx": 616.0,
        "fy": 616.0,
        "cx": 320.0,
        "cy": 240.0,
        "width": 640,
        "height": 480,
        "depth_scale": 0.001,
    }

    def __init__(self: Union["JetRacerROSMixin", ITMPolicyV2], *args: Any, **kwargs: Any) -> None:
        """初始化JetRacer ROS混入"""
        # 检查是否跳过ROS初始化
        skip_ros_init = kwargs.pop("_skip_ros_init", False)
        
        kwargs['sync_explored_areas'] = True

        super().__init__(*args, **kwargs)
        
        if not ROS_AVAILABLE:
            raise ImportError("ROS is required for JetRacer integration")
            
        if not skip_ros_init:
            self._init_ros()
        else:
            print("🔧 跳过JetRacer ROS策略内部的ROS初始化（由外部控制器管理）")
            self._ros_initialized = True
        
        # 不需要深度估计模型，使用RealSense真实深度
        # 配置对象地图使用真实深度数据
        self._object_map.use_dbscan = True

    def _init_ros(self) -> None:
        """初始化ROS节点和通信"""
        try:
            # 初始化ROS节点
            if not rospy.get_node_uri():
                rospy.init_node('vlfm_jetracer_policy', anonymous=True)
            
            # 创建CV Bridge
            self._cv_bridge = CvBridge()
            
            # 发布者 - 控制指令
            # 发布者 - JetRacer速度控制指令  
            self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
            
            # 订阅者 - 传感器数据
            self._setup_subscribers()
            
            # 等待传感器数据
            self._wait_for_sensors()
            
            self._ros_initialized = True
            rospy.loginfo("JetRacer ROS interface initialized successfully")
            
        except Exception as e:
            rospy.logerr(f"Failed to initialize ROS: {e}")
            raise

    def _setup_subscribers(self) -> None:
        """设置ROS订阅者"""
        # RGB图像
        self._rgb_sub = rospy.Subscriber(
            '/camera/color/image_raw',
            RosImage,
            self._rgb_callback,
            queue_size=1,
            buff_size=2**24
        )
        
        # 深度图像 - 使用矫正后的深度图像
        self._depth_sub = rospy.Subscriber(
            '/camera/depth/image_rect_raw',
            RosImage,
            self._depth_callback,
            queue_size=1,
            buff_size=2**24
        )
        
        # IMU数据 - 使用JetRacer的IMU话题
        self._imu_sub = rospy.Subscriber(
            '/imu',
            Imu,
            self._imu_callback,
            queue_size=1
        )
        
        # 里程计数据 - 使用JetRacer的里程计话题
        self._odom_sub = rospy.Subscriber(
            '/odom',
            Odometry,
            self._odom_callback,
            queue_size=1
        )

    def _wait_for_sensors(self, timeout: float = 10.0) -> None:
        """等待传感器数据就绪"""
        rospy.loginfo("Waiting for sensor data...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._data_lock:
                if (self._current_rgb is not None and 
                    self._current_depth is not None):
                    rospy.loginfo("Sensor data ready!")
                    return
            rospy.sleep(0.1)
        
        rospy.logwarn("Timeout waiting for sensor data")

    def _rgb_callback(self, msg: RosImage) -> None:
        """处理RGB图像数据"""
        try:
            cv_image = self._cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            with self._data_lock:
                self._current_rgb = rgb_image
                self._last_data_time = time.time()
                
        except Exception as e:
            rospy.logerr(f"Error processing RGB image: {e}")

    def _depth_callback(self, msg: RosImage) -> None:
        """处理深度图像数据"""
        try:
            # RealSense深度图像是16位
            depth_image = self._cv_bridge.imgmsg_to_cv2(msg, "16UC1")
            # 转换为米
            depth_meters = depth_image.astype(np.float32) * self._camera_params["depth_scale"]
            
            with self._data_lock:
                self._current_depth = depth_meters
                
        except Exception as e:
            rospy.logerr(f"Error processing depth image: {e}")

    def _imu_callback(self, msg: Imu) -> None:
        """处理IMU数据"""
        try:
            # 四元数转欧拉角
            q = msg.orientation
            
            # Roll (x轴旋转)
            sin_r = 2 * (q.w * q.x + q.y * q.z)
            cos_r = 1 - 2 * (q.x * q.x + q.y * q.y)
            roll = math.atan2(sin_r, cos_r)
            
            # Pitch (y轴旋转)
            sin_p = 2 * (q.w * q.y - q.z * q.x)
            pitch = math.asin(np.clip(sin_p, -1.0, 1.0))
            
            # Yaw (z轴旋转)
            sin_y = 2 * (q.w * q.z + q.x * q.y)
            cos_y = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(sin_y, cos_y)
            
            with self._data_lock:
                self._imu_data = {"roll": roll, "pitch": pitch, "yaw": yaw}
                
        except Exception as e:
            rospy.logerr(f"Error processing IMU data: {e}")

    def _odom_callback(self, msg: Odometry) -> None:
        """处理里程计数据"""
        try:
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            
            # 计算yaw角
            sin_yaw = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
            cos_yaw = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
            yaw = math.atan2(sin_yaw, cos_yaw)
            
            with self._data_lock:
                self._current_odom = {
                    "x": position.x,
                    "y": position.y,
                    "yaw": yaw
                }
                
        except Exception as e:
            rospy.logerr(f"Error processing odometry: {e}")

    @classmethod
    def from_config(cls, config: DictConfig, *args_unused: Any, **kwargs_unused: Any) -> Any:
        """从配置文件创建策略实例，强制使用非Habitat PointNav实现"""
        policy_config: VLFMConfig = config.policy
        kwargs = {k: policy_config[k] for k in VLFMConfig.kwaarg_names}
        
        # 临时禁用Habitat检测，确保使用连续动作的非Habitat PointNav实现
        removed_modules = cls._disable_habitat_for_pointnav()
        
        try:
            # 创建策略实例（此时WrappedPointNavResNetPolicy会使用非Habitat分支）
            instance = cls(**kwargs)
            print("✅ JetRacer策略使用非Habitat PointNav实现，支持连续动作")
            return instance
        finally:
            # 恢复Habitat模块，不影响其他组件
            cls._restore_habitat_modules(removed_modules)

    def act(
        self: Union["JetRacerROSMixin", ITMPolicyV2],
        observations: Dict[str, Any],
        rnn_hidden_states: Union[Tensor, Any],
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """执行动作并发送到JetRacer"""
        # 更新目标对象描述
        if observations["objectgoal"] not in self._non_coco_caption:
            self._non_coco_caption = observations["objectgoal"] + " . " + self._non_coco_caption
        
        # 调用父类策略获取动作
        parent_cls: ITMPolicyV2 = super()
        action: Tensor = parent_cls.act(
            observations, rnn_hidden_states, prev_actions, masks, deterministic
        )[0]
        
        # 发送ROS控制指令
        if self._ros_initialized:
            self._publish_control_command(action)
        
        # 返回动作信息
        action_dict = {
            "linear": action[0][1].item(),
            "angular": action[0][0].item(), 
            "info": self._policy_info,
        }
        
        if "rho_theta" in self._policy_info:
            action_dict["rho_theta"] = self._policy_info["rho_theta"]
        
        return action_dict

    def _publish_control_command(self, action: Tensor) -> None:
        """发布控制指令到JetRacer"""
        try:
            twist_msg = Twist()
            
            # VLFM输出: action[0][0]=angular, action[0][1]=linear
            linear_vel = float(action[0][1].item())
            angular_vel = float(action[0][0].item())
            
            # 安全限制
            linear_vel = np.clip(linear_vel, -0.3, 0.3)   # 最大0.3 m/s
            angular_vel = np.clip(angular_vel, -0.8, 0.8) # 最大0.8 rad/s
            
            twist_msg.linear.x = linear_vel
            twist_msg.angular.z = angular_vel
            
            self._cmd_vel_pub.publish(twist_msg)
            
            rospy.logdebug(f"Control command: linear={linear_vel:.3f}, angular={angular_vel:.3f}")
            
        except Exception as e:
            rospy.logerr(f"Error publishing control command: {e}")

    def get_action(self, observations: Dict[str, Any], masks: Tensor, deterministic: bool = True) -> Dict[str, Any]:
        """获取动作的简化接口"""
        return self.act(observations, None, None, masks, deterministic=deterministic)

    def _reset(self: Union["JetRacerROSMixin", ITMPolicyV2]) -> None:
        """重置策略状态"""
        parent_cls: ITMPolicyV2 = super()
        parent_cls._reset()
        
        # 停止机器人
        self.stop_robot()

    def _initialize(self) -> torch.Tensor:
        """JetRacer初始化动作 - 原地转一圈观察环境"""
        if not hasattr(self, '_init_steps'):
            self._init_steps = 0
        
        self._init_steps += 1
        
        # 原地转圈：分16步完成一圈（每步约22.5度）
        if self._init_steps <= 6:
            print(f"🚗 JetRacer初始化步骤 {self._init_steps}/6: 原地左转 ({self._init_steps * 22.5:.1f}°)")
            # 纯角速度转向，线速度为0（原地转圈）
            return torch.tensor([[0.04, 0.04]], dtype=torch.float32)  # [angular, linear]
        else:
            # 初始化完成，标记进入策略控制模式
            if not self._done_initializing:
                print("✅ JetRacer初始化完成（已转一圈），切换到VLFM策略控制")
                self._done_initializing = True
            
            # 返回停止动作，让策略的_explore方法接管控制
            return self._stop_action
    
    def _cache_observations(self: Union["JetRacerROSMixin", ITMPolicyV2], observations: Dict[str, Any]) -> None:
        """缓存当前观测数据"""
        # 每次都清空并重新填充缓存，确保数据实时更新
        self._observations_cache.clear()

        with self._data_lock:
            # 获取最新传感器数据
            rgb_image = self._current_rgb.copy() if self._current_rgb is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            depth_image = self._current_depth.copy() if self._current_depth is not None else np.zeros((480, 640), dtype=np.float32)
            robot_xy = np.array([self._current_odom["x"], self._current_odom["y"]])
            robot_heading = self._current_odom["yaw"]
            
            # DEBUG: 确认数据更新
            if rgb_image is not None:
                rospy.loginfo_throttle(5, f"🔄 缓存更新: RGB均值={rgb_image.mean():.1f}, 位置=({robot_xy[0]:.2f}, {robot_xy[1]:.2f})")

        # 更新障碍物地图
        self._obstacle_map: ObstacleMap
        
        # 创建相机到世界坐标变换
        tf_matrix = xyz_yaw_to_tf_matrix(
            np.array([robot_xy[0], robot_xy[1], 0.0]), 
            robot_heading
        )
        
        # 使用真实深度数据更新地图
        self._obstacle_map.update_map(
            depth_image,
            tf_matrix,
            min_depth=0.1,
            max_depth=3.0,
            fx=self._camera_params["fx"],
            fy=self._camera_params["fy"],
            topdown_fov=90,
            explore=True,
        )

        self._obstacle_map.update_agent_traj(robot_xy, robot_heading)
        frontiers = self._obstacle_map.frontiers

        # 准备深度数据给PointNav - 使用非Habitat分支，无需resize
        # 直接使用原始深度数据，参考实机策略的处理方式
        height, width = depth_image.shape
        nav_depth = torch.from_numpy(depth_image)
        nav_depth = nav_depth.reshape(1, height, width, 1).to("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔧 JetRacer 非Habitat PointNav调试:")
        print(f"   - 原始depth_image: {depth_image.shape}")
        print(f"   - nav_depth: {nav_depth.shape}, dtype={nav_depth.dtype}")
        print(f"   - 深度值范围: [{np.min(depth_image):.3f}, {np.max(depth_image):.3f}]")
        print(f"   - 使用非Habitat实现，支持连续动作")

        # 参考实机策略的观测缓存格式
        # 相机内参
        fx, fy = 600.0, 600.0
        min_depth, max_depth = 0.1, 5.0
        topdown_fov = 90.0
        
        # 构建观测缓存 - 参考reality_policies.py格式
        self._observations_cache = {
            "frontier_sensor": frontiers,
            "nav_depth": nav_depth, 
            "robot_xy": robot_xy,
            "robot_heading": robot_heading,
            # 实机策略格式：object_map_rgbd包含7元组列表
            "object_map_rgbd": [(
                rgb_image, depth_image, tf_matrix, 
                min_depth, max_depth, fx, fy
            )],
            # 实机策略格式：value_map_rgbd包含6元组列表  
            "value_map_rgbd": [(
                rgb_image, depth_image, tf_matrix,
                min_depth, max_depth, topdown_fov
            )],
        }

    def stop_robot(self) -> None:
        """停止机器人运动"""
        if self._ros_initialized:
            stop_msg = Twist()
            self._cmd_vel_pub.publish(stop_msg)
            rospy.loginfo("Robot stopped")

    def get_sensor_status(self) -> Dict[str, Any]:
        """获取传感器状态"""
        with self._data_lock:
            return {
                "rgb_available": self._current_rgb is not None,
                "depth_available": self._current_depth is not None,
                "imu_available": len(self._imu_data) > 0,
                "odom_available": any(v != 0.0 for v in self._current_odom.values()),
                "ros_initialized": self._ros_initialized,
                "data_age_seconds": time.time() - self._last_data_time if self._last_data_time > 0 else -1,
                "current_odom": self._current_odom.copy(),
                "current_imu": self._imu_data.copy(),
            }


@dataclass
class JetRacerROSConfig(DictConfig):
    """JetRacer ROS配置类"""
    policy: VLFMConfig = VLFMConfig()
    
    # 运动限制
    max_linear_velocity: float = 0.3
    max_angular_velocity: float = 0.8
    
    # 相机参数
    camera_fx: float = 616.0
    camera_fy: float = 616.0
    camera_cx: float = 320.0
    camera_cy: float = 240.0
    camera_width: int = 640
    camera_height: int = 480
    depth_scale: float = 0.001


class JetRacerROSITMPolicyV2(JetRacerROSMixin, ITMPolicyV2):
    """
    JetRacer ROS ITM策略V2
    结合JetRacer ROS通信和VLFM ITM策略的完整实现
    """
    pass
