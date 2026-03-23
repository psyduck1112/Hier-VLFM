#!/usr/bin/env python3
# 指定Python解释器路径，使脚本可直接执行
"""
VLFM JetRacer ROS运行脚本
在JetRacer小车上通过ROS运行VLFM导航策略
程序作用：作为JetRacer端的主控程序，协调传感器数据接收、VLFM策略执行、运动控制
"""

import argparse     # 命令行参数解析
import signal      # 系统信号处理（如Ctrl+C）
import sys         # 系统相关功能
import time        # 时间相关功能
from typing import Dict, Any  # 类型注解

import torch       # PyTorch深度学习框架
import numpy as np # 数值计算库
from omegaconf import DictConfig, OmegaConf  # 配置文件管理

# ROS imports - 机器人操作系统
try:
    import rospy   # ROS Python API
    from std_msgs.msg import String  # ROS标准消息类型
    ROS_AVAILABLE = True  # 标记ROS可用
except ImportError:
    # 如果ROS不可用，程序无法运行
    ROS_AVAILABLE = False
    print("Error: ROS not available")
    sys.exit(1)  # 退出程序

# VLFM imports - 导入我们创建的JetRacer策略
from vlfm.policy.jetracer_ros_policy import JetRacerROSITMPolicyV2, JetRacerROSConfig


class VLFMJetRacerRunner:
    """
    VLFM JetRacer运行器主类
    作用：管理整个导航系统的生命周期，包括初始化、运行、关闭
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化运行器
        参数：config_path - 配置文件路径（可选）
        """
        self.config = self._load_config(config_path)  # 加载配置
        self.policy = None    # VLFM策略实例（初始为None）
        self.running = False  # 运行状态标志
        
        # 注册信号处理函数，确保程序能优雅退出
        signal.signal(signal.SIGINT, self._signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, self._signal_handler)  # 终止信号

    def _load_config(self, config_path: str = None) -> DictConfig:
        """
        加载配置文件
        作用：如果提供了配置文件就加载，否则使用默认配置
        """
        if config_path:
            # 从YAML文件加载配置
            config = OmegaConf.load(config_path)
        else:
            # 使用内置默认配置
            config = OmegaConf.create({
                "policy": {
                    "obs_transform": "center_crop_resize",    # 图像预处理方式
                    "rgb_image_size": 224,                    # RGB图像尺寸
                    "depth_image_size": 256,                  # 深度图像尺寸
                    "normalize_depth": True,                  # 是否归一化深度
                    "coco_threshold": 0.8,                    # COCO对象检测阈值
                    "non_coco_threshold": 0.4,                # 非COCO对象检测阈值
                    "spatial_sample_rate": 1,                 # 空间采样率
                    "object_map_erosion_size": 3,             # 对象地图腐蚀核大小
                },
                "max_linear_velocity": 0.3,     # 最大线性速度 (m/s)
                "max_angular_velocity": 0.8,    # 最大角速度 (rad/s)
                "target_object": "chair",       # 默认搜索目标对象
            })
        
        # 打印加载的配置，便于调试
        rospy.loginfo(f"Loaded config: {OmegaConf.to_yaml(config)}")
        return config

    def _signal_handler(self, signum, frame):
        """
        信号处理函数
        作用：当收到终止信号时，优雅地关闭系统
        """
        rospy.loginfo(f"Received signal {signum}, shutting down...")
        self.shutdown()  # 调用关闭函数

    def initialize_policy(self) -> None:
        """
        初始化VLFM策略
        作用：创建策略实例，建立与传感器和VLM服务器的连接
        """
        try:
            rospy.loginfo("Initializing VLFM policy...")
            
            # 创建JetRacer专用的VLFM策略实例
            # 这个实例会：1.初始化ROS订阅者 2.连接VLM服务器 3.设置运动控制
            self.policy = JetRacerROSITMPolicyV2.from_config(self.config)
            
            # 等待VLM服务器就绪（在主机端运行的VLM推理服务器）
            self._wait_for_vlm_servers()
            
            rospy.loginfo("VLFM policy initialized successfully")
            
        except Exception as e:
            rospy.logerr(f"Failed to initialize policy: {e}")
            raise  # 重新抛出异常

    def _wait_for_vlm_servers(self, timeout: float = 60.0) -> None:
        """
        等待VLM服务器就绪
        作用：确保主机端的VLM推理服务器已启动并可用
        参数：timeout - 超时时间（秒）
        """
        rospy.loginfo("Waiting for VLM servers...")
        # TODO: 这里应该实际检查VLM服务器状态
        # 可以通过HTTP请求测试服务器连通性
        time.sleep(2)  # 简单等待（实际应用中需要改进）
        rospy.loginfo("VLM servers ready")

    def run_navigation_loop(self) -> None:
        """
        运行导航主循环
        作用：持续执行"感知→决策→行动"的导航循环
        """
        if not self.policy:
            raise RuntimeError("Policy not initialized")
        
        rospy.loginfo("Starting navigation loop...")
        self.running = True  # 设置运行标志
        
        # 创建模拟观测数据结构（VLFM策略需要的掩码）
        masks = torch.ones(1, 1, dtype=torch.bool)  # 全1掩码
        
        rate = rospy.Rate(10)  # 设置循环频率为10Hz（每秒10次）
        
        # 主导航循环
        while self.running and not rospy.is_shutdown():
            try:
                # 步骤1：构造观测数据（包含目标对象信息）
                observations = self._create_observations()
                
                # 步骤2：调用VLFM策略获取动作
                # 策略会：1.处理传感器数据 2.调用VLM推理 3.计算导航动作
                action_dict = self.policy.get_action(
                    observations=observations,
                    masks=masks,
                    deterministic=True  # 确定性策略（不使用随机性）
                )
                
                # 步骤3：记录运行状态，便于监控和调试
                self._log_status(action_dict)
                
                # 步骤4：等待下一个循环周期
                rate.sleep()  # 保持10Hz频率
                
            except Exception as e:
                rospy.logerr(f"Error in navigation loop: {e}")
                break  # 出现错误时退出循环
        
        rospy.loginfo("Navigation loop ended")

    def _create_observations(self) -> Dict[str, Any]:
        """
        创建观测数据字典
        作用：为VLFM策略准备输入数据，主要是目标对象信息
        """
        # 检查传感器状态
        sensor_status = self.policy.get_sensor_status()
        
        # 如果传感器数据不可用，发出警告
        if not sensor_status["rgb_available"] or not sensor_status["depth_available"]:
            rospy.logwarn("Sensor data not available")
        
        # 构造观测数据字典
        observations = {
            "objectgoal": self.config.get("target_object", "chair"),  # 目标对象
            # 注意：RGB、深度等传感器数据由policy内部自动处理
        }
        
        return observations

    def _log_status(self, action_dict: Dict[str, Any]) -> None:
        """
        记录运行状态
        作用：输出当前动作和传感器状态，便于监控系统运行
        """
        sensor_status = self.policy.get_sensor_status()
        
        # 输出详细状态信息
        rospy.logdebug(
            f"Action: linear={action_dict['linear']:.3f}, "        # 线性速度
            f"angular={action_dict['angular']:.3f} | "             # 角速度
            f"Sensors: RGB={sensor_status['rgb_available']}, "     # RGB可用性
            f"Depth={sensor_status['depth_available']}, "          # 深度可用性
            f"Data age={sensor_status['data_age_seconds']:.1f}s"   # 数据新鲜度
        )

    def run(self) -> None:
        """
        运行完整的导航系统
        作用：系统的主入口函数，管理整个运行流程
        """
        try:
            rospy.loginfo("Starting VLFM JetRacer navigation system...")
            
            # 步骤1：初始化策略（建立所有连接）
            self.initialize_policy()
            
            # 步骤2：运行导航循环（主要工作逻辑）
            self.run_navigation_loop()
            
        except KeyboardInterrupt:
            # 用户按Ctrl+C中断
            rospy.loginfo("Interrupted by user")
        except Exception as e:
            # 其他致命错误
            rospy.logerr(f"Fatal error: {e}")
        finally:
            # 无论如何都要执行清理工作
            self.shutdown()

    def shutdown(self) -> None:
        """
        关闭系统
        作用：优雅地停止所有组件，清理资源
        """
        rospy.loginfo("Shutting down VLFM JetRacer system...")
        
        self.running = False  # 停止主循环
        
        if self.policy:
            # 停止机器人运动，确保安全
            self.policy.stop_robot()
        
        rospy.loginfo("Shutdown complete")


def main():
    """
    主函数
    作用：程序入口，处理命令行参数，创建并运行系统
    """
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="VLFM JetRacer ROS Navigation")
    parser.add_argument("--config", type=str, help="Path to config file")          # 配置文件路径
    parser.add_argument("--target", type=str, default="chair", help="Target object to search for")  # 目标对象
    parser.add_argument("--log-level", type=str, default="info",                   # 日志级别
                       choices=["debug", "info", "warn", "error"],
                       help="ROS log level")
    args = parser.parse_args()  # 解析命令行参数
    
    # 设置ROS日志级别
    if args.log_level == "debug":
        rospy.set_param('/rospy/logger/level', 'DEBUG')
    
    try:
        # 创建运行器实例
        runner = VLFMJetRacerRunner(config_path=args.config)
        
        # 设置目标对象（命令行参数覆盖配置文件）
        if args.target:
            runner.config.target_object = args.target
        
        # 运行系统
        runner.run()
        
    except Exception as e:
        rospy.logerr(f"Failed to start system: {e}")
        sys.exit(1)  # 错误退出


if __name__ == "__main__":
    # 当脚本直接执行时，调用main函数
    main()