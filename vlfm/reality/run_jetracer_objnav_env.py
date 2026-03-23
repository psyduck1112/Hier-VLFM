# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
# Modifications Copyright (c) 2026 Yikang.
# JetRacer ObjectNav Runner - Complete Reality Architecture Implementation

# 添加VLFM包搜索路径
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import time

import hydra
import numpy as np
import torch
import rospy
from omegaconf import OmegaConf

# 导入JetRacer专用Reality策略
try:
    from vlfm.reality.jetracer_reality_policy import JetRacerRealityConfig, JetRacerRealityITMPolicyV2
    # 设置别名以保持代码兼容性
    RealityConfig = JetRacerRealityConfig
    RealityITMPolicyV2 = JetRacerRealityITMPolicyV2
except ImportError:
    # 如果JetRacer Reality策略不可用，尝试使用通用Reality策略
    rospy.logwarn("JetRacer Reality策略不可用，尝试使用通用Reality策略")
    try:
        from vlfm.policy.reality_policies import RealityConfig, RealityITMPolicyV2
    except ImportError:
        # 如果都不可用，使用基础ITM策略
        rospy.logwarn("所有Reality策略都不可用，将使用基础ITM策略")
        from vlfm.policy.itm_policy import ITMPolicyV2 as RealityITMPolicyV2
        
        # 创建简化的配置类
        class RealityConfig:
            def __init__(self):
                self.policy = None
                self.env = None

from vlfm.reality.jetracer_objectnav_env import JetRacerObjectNavEnv
from vlfm.reality.robots.jetracer_robot import JetRacerRobot


@hydra.main(version_base=None, config_path="../../config/", config_name="experiments/jetracer_reality")
def main(cfg) -> None:
    """
    JetRacer ObjectNav主函数 - 完全遵循Reality架构模式
    
    使用Hydra配置管理，RealityITMPolicyV2策略，标准run_env流程
    """
    print("🚀 启动JetRacer ObjectNav - Reality架构")
    print("=" * 50)
    print(OmegaConf.to_yaml(cfg))
    
    # 创建Reality策略 - 使用标准Reality组件
    rospy.loginfo("🧠 初始化Reality ITM策略...")
    policy = RealityITMPolicyV2.from_config(cfg)
    rospy.loginfo("✅ Reality策略创建成功")
    
    # 初始化ROS节点
    if not rospy.get_node_uri():
        rospy.init_node('jetracer_reality_main', anonymous=True)
    
    try:
        # 创建JetRacer机器人连接
        rospy.loginfo("🤖 连接JetRacer机器人...")
        robot = JetRacerRobot()
        
        # 等待机器人就绪
        rospy.loginfo("⏳ 等待JetRacer传感器就绪...")
        wait_count = 0
        while not robot.is_ready() and wait_count < 30:  # 30秒超时
            rospy.loginfo(f"等待传感器数据... ({wait_count}/30)")
            rospy.sleep(1.0)
            wait_count += 1
        
        if not robot.is_ready():
            rospy.logwarn("⚠️ 传感器数据等待超时，继续运行...")
        else:
            rospy.loginfo("✅ JetRacer传感器就绪")
        
        # JetRacer初始化（原地转圈，类似Spot的arm positioning）
        rospy.loginfo("🔄 JetRacer初始化 - 原地转圈扫描环境...")
        for i in range(8):  # 8步完成一圈
            robot.move(0.0, 0.4)  # 纯转动
            rospy.sleep(0.5)
            rospy.loginfo(f"初始化转圈 {i+1}/8")
        
        robot.stop()
        rospy.loginfo("✅ JetRacer初始化完成")
        rospy.sleep(1.0)  # 稳定一下
        
        # 创建ObjectNav环境
        rospy.loginfo("🌍 创建JetRacer ObjectNav环境...")
        env = JetRacerObjectNavEnv(
            robot=robot,
            max_body_cam_depth=cfg.env.max_body_cam_depth,
            max_gripper_cam_depth=cfg.env.max_gripper_cam_depth,
            max_lin_dist=cfg.env.max_lin_dist,
            max_ang_dist=cfg.env.max_ang_dist,
            time_step=cfg.env.time_step,
        )
        rospy.loginfo("✅ 环境创建成功")
        
        # 获取目标物体
        goal = cfg.env.goal
        rospy.loginfo(f"🎯 开始导航，目标: {goal}")
        
        # 运行导航任务
        run_env(env, policy, goal)
        
    except KeyboardInterrupt:
        rospy.loginfo("用户中断程序")
    except Exception as e:
        rospy.logerr(f"❌ 程序错误: {e}")
        import traceback
        rospy.logerr(f"详细错误: {traceback.format_exc()}")
    finally:
        # 清理资源
        rospy.loginfo("🔄 清理资源...")
        if 'robot' in locals():
            robot.stop()
            robot.disconnect()
        if 'env' in locals():
            env.close()
        rospy.loginfo("✅ 程序结束")


def run_env(env: JetRacerObjectNavEnv, policy, goal: str) -> None:
    """
    运行环境主循环 - 完全遵循Reality架构的run_env模式
    
    Args:
        env: JetRacer ObjectNav环境
        policy: Reality ITM策略
        goal: 目标物体名称
    """
    rospy.loginfo(f"🏃 开始导航循环，目标: {goal}")
    
    # 重置环境
    observations = env.reset(goal)
    done = False
    
    # 创建mask张量 - Reality架构标准
    mask = torch.zeros(1, 1, device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bool)
    
    # 获取初始动作
    st = time.time()
    action = policy.get_action(observations, mask)
    elapsed = time.time() - st
    rospy.loginfo(f"⚡ 初始动作计算耗时: {elapsed:.2f}秒")
    
    step_count = 0
    start_time = time.time()
    
    # 主导航循环 - 完全遵循Reality架构模式
    while not done:
        step_start = time.time()
        
        try:
            # 执行动作
            observations, reward, done, info = env.step(action)
            step_count += 1
            
            # 记录步骤信息
            robot_pos = env.robot.pose
            rospy.loginfo(f"📍 步骤 {step_count}: 位置({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), "
                         f"朝向{np.rad2deg(robot_pos[2]):.1f}°, 奖励{reward:.3f}")
            
            if done:
                total_time = time.time() - start_time
                rospy.loginfo(f"🎉 Episode完成！")
                rospy.loginfo(f"📊 总计: {step_count}步, {total_time:.1f}秒, 目标: {goal}")
                break
            
            # 获取下一个动作
            action_start = time.time()
            action = policy.get_action(observations, mask, deterministic=True)
            action_time = time.time() - action_start
            
            # 更新mask - Reality架构要求
            mask = torch.ones_like(mask)
            
            # 性能监控
            step_time = time.time() - step_start
            rospy.loginfo(f"⚡ 动作计算: {action_time:.2f}s, 总步骤: {step_time:.2f}s")
            
            # 安全检查 - 避免死循环
            if step_count > 500:
                rospy.logwarn("⚠️ 达到最大步数限制，结束episode")
                break
            
            # 时间限制
            if time.time() - start_time > 600:  # 10分钟限制
                rospy.logwarn("⚠️ 达到时间限制，结束episode")
                break
        
        except KeyboardInterrupt:
            rospy.loginfo("用户中断导航")
            break
        except Exception as e:
            rospy.logerr(f"❌ 导航步骤错误: {e}")
            break
    
    # 导航结束，停止机器人
    env.robot.stop()
    total_time = time.time() - start_time
    rospy.loginfo(f"🏁 导航结束: {step_count}步, {total_time:.1f}秒")


if __name__ == "__main__":
    main()
