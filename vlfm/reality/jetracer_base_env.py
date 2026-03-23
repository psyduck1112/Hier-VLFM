# JetRacer Base Environment - Independent of Spot Dependencies
# 创建JetRacer专用的基础环境，不依赖Spot组件

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rospy

from vlfm.utils.geometry_utils import (
    convert_to_global_frame,
    pt_from_rho_theta, 
    rho_theta,
)


class JetRacerBaseEnv:
    """
    JetRacer基础环境 - 不依赖Spot组件
    
    提供基础的导航环境功能，适用于地面移动机器人
    """

    goal: Any = (None,)  # must be set by reset()
    info: Dict = {}

    def __init__(
        self,
        robot,  # JetRacerRobot实例，不强制类型
        max_body_cam_depth: float = 3.0,
        max_lin_dist: float = 0.3,
        max_ang_dist: float = np.deg2rad(30),
        time_step: float = 1.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        初始化JetRacer基础环境
        
        Args:
            robot: JetRacer机器人实例
            max_body_cam_depth: 最大相机深度
            max_lin_dist: 最大线速度
            max_ang_dist: 最大角速度
            time_step: 时间步长
        """
        self.robot = robot
        
        self._max_body_cam_depth = max_body_cam_depth
        self._max_lin_dist = max_lin_dist
        self._max_ang_dist = max_ang_dist
        self._time_step = time_step
        self._cmd_id: Optional[Any] = None
        self._num_steps = 0

    def reset(self, goal: Any = None, relative: bool = True, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        重置环境
        
        Args:
            goal: 目标位置或目标对象
            relative: 是否是相对目标
            
        Returns:
            初始观测字典
        """
        if isinstance(goal, np.ndarray):
            # PointNav模式：数值目标
            if relative:
                # 转换相对目标到全局坐标
                pos, yaw = self.robot.xy_yaw
                pos_w_z = np.array([pos[0], pos[1], 0.0])
                goal_w_z = np.array([goal[0], goal[1], 0.0])
                global_goal = convert_to_global_frame(goal_w_z, pos_w_z, yaw)
                self.goal = global_goal[:2]  # 只取x,y
            else:
                self.goal = goal
        else:
            # ObjectNav模式：字符串目标
            self.goal = goal
        
        self._num_steps = 0
        self.info = {}
        
        return self.get_observations()

    def get_observations(self) -> Dict[str, Any]:
        """
        获取观测数据
        
        Returns:
            观测数据字典
        """
        observations = {}
        
        try:
            # 获取机器人位置
            robot_xy, robot_yaw = self.robot.xy_yaw
            observations.update({
                "robot_xy": robot_xy,
                "robot_heading": robot_yaw,
                "robot_position": np.concatenate([robot_xy, [robot_yaw]]),
            })
            
            # 如果是数值目标（PointNav），计算相对位置
            if isinstance(self.goal, np.ndarray):
                # 计算目标相对位置
                goal_rho_theta = rho_theta(self.goal, robot_xy, robot_yaw)
                observations.update({
                    "pointgoal": goal_rho_theta,
                    "goal_position": self.goal,
                })
            
            # 添加基础信息
            observations.update({
                "step_count": self._num_steps,
                "goal": self.goal,
            })
            
        except Exception as e:
            rospy.logwarn(f"获取观测数据失败: {e}")
            # 提供默认观测
            observations = {
                "robot_xy": np.array([0.0, 0.0]),
                "robot_heading": 0.0,
                "step_count": self._num_steps,
                "goal": self.goal,
            }
        
        return observations

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        执行一步动作
        
        Args:
            action: 动作字典
            
        Returns:
            (observations, reward, done, info)
        """
        self._num_steps += 1
        
        try:
            # 解析动作
            if "rho_theta" in action:
                # PointNav动作格式
                rho_theta_action = action["rho_theta"]
                
                if isinstance(rho_theta_action, np.ndarray):
                    if len(rho_theta_action) >= 2:
                        rho = float(rho_theta_action[0])
                        theta = float(rho_theta_action[1])
                    else:
                        rho = float(rho_theta_action[0])
                        theta = 0.0
                else:
                    rho = float(rho_theta_action)
                    theta = 0.0
                
                # 转换为速度指令
                linear_vel = np.clip(rho * 0.5, -self._max_lin_dist, self._max_lin_dist)
                angular_vel = np.clip(theta * 0.5, -self._max_ang_dist, self._max_ang_dist)
                
            elif "linear" in action and "angular" in action:
                # 直接速度控制
                linear_vel = np.clip(action["linear"], -self._max_lin_dist, self._max_lin_dist)
                angular_vel = np.clip(action["angular"], -self._max_ang_dist, self._max_ang_dist)
                
            else:
                # 默认停止
                linear_vel = 0.0
                angular_vel = 0.0
            
            # 发送控制指令
            self.robot.move(linear_vel, angular_vel)
            
            # 等待动作执行
            rospy.sleep(self._time_step)
            
            # 获取新观测
            observations = self.get_observations()
            
            # 计算奖励和完成条件
            reward = self._compute_reward(observations)
            done = self._is_episode_done(observations)
            
            # 更新信息
            self.info.update({
                "step_count": self._num_steps,
                "action": action,
                "linear_vel": linear_vel,
                "angular_vel": angular_vel,
            })
            
            return observations, reward, done, self.info
        
        except Exception as e:
            rospy.logerr(f"执行步骤失败: {e}")
            # 停止机器人
            self.robot.stop()
            
            observations = self.get_observations()
            return observations, -1.0, True, {"error": str(e)}

    def _compute_reward(self, observations: Dict[str, Any]) -> float:
        """
        计算奖励
        
        Args:
            observations: 当前观测
            
        Returns:
            奖励值
        """
        reward = 0.0
        
        try:
            if isinstance(self.goal, np.ndarray):
                # PointNav奖励：基于距离目标的距离
                robot_xy = observations["robot_xy"]
                distance_to_goal = np.linalg.norm(self.goal - robot_xy)
                
                # 距离奖励（越近越好）
                reward = -distance_to_goal * 0.1
                
                # 到达奖励
                if distance_to_goal < 0.5:
                    reward += 10.0
                    
            else:
                # ObjectNav奖励：时间惩罚
                reward = -0.01
        
        except Exception as e:
            rospy.logwarn(f"奖励计算失败: {e}")
            reward = -0.01
        
        return reward

    def _is_episode_done(self, observations: Dict[str, Any]) -> bool:
        """
        判断episode是否结束
        
        Args:
            observations: 当前观测
            
        Returns:
            是否结束
        """
        try:
            # 步数限制
            if self._num_steps > 200:
                rospy.loginfo("达到最大步数限制")
                return True
            
            if isinstance(self.goal, np.ndarray):
                # PointNav完成条件：到达目标
                robot_xy = observations["robot_xy"]
                distance_to_goal = np.linalg.norm(self.goal - robot_xy)
                
                if distance_to_goal < 0.5:
                    rospy.loginfo("到达目标位置")
                    return True
            
            # 其他情况继续
            return False
            
        except Exception as e:
            rospy.logwarn(f"完成条件检查失败: {e}")
            return False

    def close(self):
        """关闭环境"""
        rospy.loginfo("关闭JetRacer基础环境")
        
        if hasattr(self.robot, 'stop'):
            self.robot.stop()
        
        if hasattr(self.robot, 'disconnect'):
            self.robot.disconnect()