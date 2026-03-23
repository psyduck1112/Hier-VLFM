# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
# Modifications Copyright (c) 2026 Yikang.

# 导入必要的库和模块
import numpy as np  # 数值计算库，用于处理数组和数学运算
import torch  # PyTorch深度学习框架，用于张量操作和神经网络
from spot_wrapper.spot import Spot  # Boston Dynamics Spot机器人的Python封装库

# 导入VLFM项目内部模块
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy  # 点导航策略类
from vlfm.reality.pointnav_env import PointNavEnv  # 真实环境中的点导航环境类
from vlfm.reality.robots.bdsw_robot import BDSWRobot  # BDSW(Boston Dynamics Spot Wrapper)机器人适配器类


def run_env(env: PointNavEnv, policy: WrappedPointNavResNetPolicy, goal: np.ndarray) -> None:
    """
    运行点导航环境的主循环函数
    
    参数:
        env: 点导航环境实例，负责与Spot机器人交互
        policy: 点导航策略实例，负责根据观测生成导航动作
        goal: 目标位置，numpy数组格式 [x, y]，表示机器人要到达的坐标
    """
    # 重置环境到初始状态，设置目标位置，返回初始观测数据
    observations = env.reset(goal)
    
    # 初始化完成标志为False，表示任务尚未完成
    done = False
    
    # 创建掩码张量，用于指示episode是否开始
    # shape: [1, 1], 在策略的device上(通常是GPU)，数据类型为布尔型
    # 初始为False，表示这是episode的第一步
    mask = torch.zeros(1, 1, device=policy.device, dtype=torch.bool)
    
    # 策略根据初始观测生成第一个动作
    # observations包含深度图像和目标相对位置等信息
    action = policy.act(observations, mask)
    
    # 将策略输出的动作包装成环境期望的格式
    # "rho_theta"表示极坐标形式：rho是距离，theta是角度
    action_dict = {"rho_theta": action}
    
    # 主循环：持续执行动作直到任务完成
    while not done:
        # 执行动作并获取环境反馈
        # observations: 新的观测数据（深度图、GPS等）
        # _: 奖励值（在这里未使用）
        # done: 任务是否完成的布尔标志
        # info: 额外信息字典（在这里未使用）
        observations, _, done, info = env.step(action_dict)
        
        # 根据新观测生成下一个动作
        # deterministic=True表示使用确定性策略（不使用随机性）
        action = policy.act(observations, mask, deterministic=True)
        
        # 更新掩码为True，表示不再是episode的第一步
        # 这对RNN类型的策略很重要，用于区分序列的开始
        mask = torch.ones_like(mask)


if __name__ == "__main__":
    # 当文件作为主程序运行时执行以下代码
    import argparse  # 导入命令行参数解析库

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    
    # 添加必需的位置参数：点导航模型检查点文件路径
    parser.add_argument(
        "pointnav_ckpt_path",  # 参数名称
        type=str,  # 参数类型为字符串
        default="pointnav_resnet_18.pth",  # 默认值
        help="Path to the pointnav model checkpoint",  # 帮助文本
    )
    
    # 添加可选参数：目标位置坐标
    parser.add_argument(
        "-g",  # 短选项名
        "--goal",  # 长选项名
        type=str,  # 参数类型为字符串
        default="3.5,0.0",  # 默认目标位置：x=3.5米，y=0.0米
        help="Goal location in the form x,y",  # 帮助文本
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 从解析结果中提取检查点文件路径
    pointnav_ckpt_path = args.pointnav_ckpt_path
    
    # 使用检查点文件创建点导航策略实例
    # 这会加载预训练的ResNet-18网络权重
    policy = WrappedPointNavResNetPolicy(pointnav_ckpt_path)
    
    # 解析目标位置字符串，转换为numpy数组
    # 例如：将"3.5,0.0"分割并转换为[3.5, 0.0]
    goal = np.array([float(x) for x in args.goal.split(",")])

    # 创建Spot机器人连接实例
    # "BDSW_env"是连接名称，可以是任意字符串标识符
    spot = Spot("BDSW_env")  # just a name, can be anything
    
    # 使用with语句确保机器人资源的正确管理
    # get_lease()获取机器人控制权，退出时自动释放
    with spot.get_lease():  # turns the robot on, and off upon any errors or completion
        # 开启机器人电源
        spot.power_on()
        
        # 让机器人站立，这是一个阻塞操作，会等待直到站立完成
        spot.blocking_stand()
        
        # 创建BDSW机器人适配器，封装Spot SDK的底层操作
        robot = BDSWRobot(spot)
        
        # 创建点导航环境实例，将机器人适配器传入
        env = PointNavEnv(robot)
        
        # 运行主环境循环，开始点导航任务
        # 机器人将尝试导航到指定的目标位置
        run_env(env, policy, goal)
