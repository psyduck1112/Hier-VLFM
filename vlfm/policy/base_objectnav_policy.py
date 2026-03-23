# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
# Modifications Copyright (c) 2026 Yikang.

import os
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from torch import Tensor

from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap  # 物体点云地图
from vlfm.mapping.obstacle_map import ObstacleMap  # 障碍物地图
from vlfm.obs_transformers.utils import image_resize
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.utils.geometry_utils import get_fov, rho_theta
from vlfm.vlm.blip2 import BLIP2Client  # VQA（视觉问答）
from vlfm.vlm.coco_classes import COCO_CLASSES  # COCO 数据集类别列表
from vlfm.vlm.grounding_dino import GroundingDINOClient, ObjectDetections  # 任意目标检测
from vlfm.vlm.sam import MobileSAMClient  # 图像分割
from vlfm.vlm.yolov7 import YOLOv7Client  # YOLOv7目标检测客户端

# 临时回退到YOLOv7直到YOLO-World索引问题解决


try:  # 尝试从habitat_baselines 和 vlfm 导入基类
    from habitat_baselines.common.tensor_dict import TensorDict

    from vlfm.policy.base_policy import BasePolicy

except Exception:  # 如果失败（比如在测试环境里），就定义一个空的 BasePolicy

    class BasePolicy:  # type: ignore
        pass


class BaseObjectNavPolicy(BasePolicy):
    _target_object: str = ""  # 目标物体名称
    _policy_info: Dict[str, Any] = {}  # 保存调试/可视化信息
    _object_masks: Union[np.ndarray, Any] = None  # 物体掩码 set by ._update_object_map()
    _stop_action: Union[Tensor, Any] = None  # 导航停止时的动作 MUST BE SET BY SUBCLASS
    _observations_cache: Dict[str, Any] = {}  # 缓存当前时刻的传感器数据
    _non_coco_caption = ""  # 非COCO类别的文本描述
    # COCO 数据集（Common Objects in Context）里定义的物体类别
    _load_yolo: bool = True

    def __init__(  # 初始化所有子模块
        self,
        pointnav_policy_path: str,
        depth_image_shape: Tuple[int, int],
        pointnav_stop_radius: float,
        object_map_erosion_size: float,
        visualize: bool = True,
        compute_frontiers: bool = True,
        min_obstacle_height: float = 0.15,
        max_obstacle_height: float = 0.88,
        agent_radius: float = 0.18,
        obstacle_map_area_threshold: float = 1.5,
        hole_area_thresh: int = 100000,
        use_vqa: bool = False,  # 是否启用VQA
        vqa_prompt: str = "Is this ",  # 提问的前缀
        coco_threshold: float = 0.8,  # 置信度阈值，小于这个值丢弃结果
        non_coco_threshold: float = 0.4,  # 非coco类目标检测阈值
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # 创建客户端
        print("🔗 正在连接GroundingDINO客户端...")
        self._object_detector = GroundingDINOClient(port=int(os.environ.get("GROUNDING_DINO_PORT", "12181")))  #
        print("✅ GroundingDINO客户端创建成功")

        print("🔗 正在连接YOLOv7客户端...")
        self._coco_object_detector = YOLOv7Client(port=int(os.environ.get("YOLOV7_PORT", "12184")))
        print("✅ YOLOv7客户端创建成功")

        print("🔗 正在连接SAM客户端...")
        self._mobile_sam = MobileSAMClient(port=int(os.environ.get("SAM_PORT", "12183")))
        print("✅ SAM客户端创建成功")
        self._use_vqa = use_vqa
        if use_vqa:
            self._vqa = BLIP2Client(port=int(os.environ.get("BLIP2_PORT", "12185")))
        # 初始化
        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)  # 导航策略
        self._object_map: ObjectPointCloudMap = ObjectPointCloudMap(
            erosion_size=object_map_erosion_size
        )  # 存储目标点云
        self._depth_image_shape = tuple(depth_image_shape)  # 保存深度图尺寸
        self._pointnav_stop_radius = pointnav_stop_radius  # 保存停止半径
        self._visualize = visualize
        self._vqa_prompt = vqa_prompt
        self._coco_threshold = coco_threshold
        self._non_coco_threshold = non_coco_threshold

        self._num_steps = 0  # 记录步数
        self._did_reset = False  # 是否重置
        self._last_goal = np.zeros(2)  # 上一次目标坐标
        self._done_initializing = False  # 是否完成初始化
        self._called_stop = False  # 是否已经调用停止
        self._compute_frontiers = compute_frontiers  # 是否计算边界
        if compute_frontiers:
            self._obstacle_map = ObstacleMap(  # 初始化障碍物地图类
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                area_thresh=obstacle_map_area_threshold,
                agent_radius=agent_radius,
                hole_area_thresh=hole_area_thresh,
                size=1500,  # 与ValueMap(size=1500)保持一致，避免size不匹配断言错误
                pixels_per_meter=20,
            )

        # RGB截取计数器控制
        # self._capture_counter = 0
        # self._capture_interval = 1  # 每帧截取一次，可调整

    def _reset(self) -> None:
        self._target_object = ""
        self._pointnav_policy.reset()
        self._object_map.reset()
        self._last_goal = np.zeros(2)
        self._num_steps = 0
        self._done_initializing = False
        self._called_stop = False
        # 重置RGB截取计数器
        self._capture_counter = 0
        if self._compute_frontiers:
            self._obstacle_map.reset()
        self._did_reset = True

        # 清除停止原因
        self._policy_info.pop("stop_reason", None)

    '''
    截取图片
    
    def _save_habitat_rgb_interval(self, observations):
        """按间隔截取RGB图像"""
        self._capture_counter += 1
        if self._capture_counter >= self._capture_interval:
            self._save_habitat_rgb(observations, f"interval_{self._capture_counter}")
            self._capture_counter = 0  # 重置计数器
    
    def _save_habitat_rgb(self, observations, event_type="manual"):
        """保存Habitat RGB观测数据"""
        import os
        from datetime import datetime
        
        if "rgb" not in observations:
            return
        
        rgb_obs = observations["rgb"]  
        
        if isinstance(rgb_obs, torch.Tensor):
            rgb_obs = rgb_obs[0].cpu().numpy()  # 选择batch中的第一个图像
        else:
            # 如果已经是numpy数组，也要去掉batch维度
            rgb_obs = rgb_obs[0] if len(rgb_obs.shape) == 4 else rgb_obs
    
    # 额外的安全检查：确保数据类型是 uint8 (OpenCV 某些版本对 float 有严格要求)
        if rgb_obs.dtype != np.uint8:
        # 如果是 0-1 的浮点数，乘以 255 并转换
         if rgb_obs.max() <= 1.0:
            rgb_obs = (rgb_obs * 255).astype(np.uint8)
        else:
            rgb_obs = rgb_obs.astype(np.uint8)
    # ---------------------------
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"habitat_rgb_{event_type}_{timestamp}.jpg"
        
        # 保存路径
        save_dir = "captured_images"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        
        # 保存图像
        rgb_bgr = cv2.cvtColor(rgb_obs, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(save_path, rgb_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if success:
            print(f"📸 RGB已保存: {save_path}")
        else:
            print(f"❌ RGB保存失败: {save_path}")
    '''

    def act(
        self,
        observations: Dict,  # 传感器的观测数据（相机图像、深度图、位姿信息等），字典形式
        rnn_hidden_states: Any,  # RNN 的隐藏状态
        prev_actions: Any,  # 上一步执行的动作
        masks: Tensor,  # 掩码，用来表示 episode 是否结束
        deterministic: bool = False,  # 是否用确定性动作
    ) -> Any:
        """每一步调用：根据观测更新内部状态，决定机器人动作（探索/导航/停止）
        启动任务时，首先进行“初始化”，让机器人确定自身方位（例如：通过原地旋转来全面观察周围环境）。
        随后，在场景中进行探索，直至找到目标物体。一旦发现目标物体，便导航至该物体所在位置。
        Starts the episode by 'initializing' and allowing robot to get its bearings
        (e.g., spinning in place to get a good view of the scene).
        Then, explores the scene until it finds the target object.
        Once the target object is found, it navigates to the object.
        """
        # 按间隔截取RGB图像
        # self._save_habitat_rgb_interval(observations)

        self._pre_step(observations, masks)  # 接收数据和掩码，存入缓存

        object_map_rgbd = self._observations_cache["object_map_rgbd"]  # 从缓存里取出 RGB-D 数据（RGB + Depth 图像）
        detections = [  # 循环更新物体地图，返回每帧检测结果
            self._update_object_map(
                rgb, depth, tf, min_depth, max_depth, fx, fy
            )  # 对 object_map_rgbd 的每一帧图像进行循环处理
            for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
        ]
        #  打印检测结果
        for detection in detections:
            print("detections:", detection.phrases)

        robot_xy = self._observations_cache["robot_xy"]  # 取出机器人当前位置
        goal = self._get_target_object_location(robot_xy)  # 查询目标物体的地图位置

        if not self._done_initializing:  # Initialize
            mode = "initialize"
            pointnav_action = self._initialize()  # 执行初始化
        elif goal is None:  # Haven't found target object yet
            mode = "explore"
            pointnav_action = self._explore(observations)  # 调用 _explore()，让机器人去探索环境
        else:
            mode = "navigate"
            # pointnav_action = self._pointnav(goal[:2], stop=True) # goal[:2] 是 (x, y) 坐标，stop=True 表示到达目标时执行"停止"动作
            pointnav_action = self._stop_action
        # 防护检查：确保pointnav_action不为None
        if pointnav_action is None:
            print(f"⚠️ {mode} 模式返回了None动作，使用停止动作")
            pointnav_action = self._stop_action

        action_numpy = (
            pointnav_action.detach().cpu().numpy()[0]
        )  # 先分离计算图，不再计算梯度，再移到 CPU，再转成 NumPy 数组
        if len(action_numpy) == 1:
            action_numpy = action_numpy[0]  # [0] 是因为动作一般在 batch 维度上有一层包装
        self._policy_info.update(self._get_policy_info(detections[0]))  # 从检测结果提取有用信息，更新策略内部信息
        self._policy_info["mode"] = mode  # 添加当前导航模式到策略信息中
        self._num_steps += 1

        self._observations_cache = {}  # 清除本次缓存观测数据
        self._did_reset = False

        # 如果已停止，保持停止原因直到下次重置
        if not (hasattr(self, "_called_stop") and self._called_stop):
            self._policy_info.pop("stop_reason", None)  # 移除停止原因（如果未停止）

        return pointnav_action, rnn_hidden_states  # 返回这一步要执行的动作，以及更新后的 RNN 隐藏状态

    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:
        """这是一个预处理步骤方法，在每个环境步骤之前执行。
        接收观测数据（TensorDict格式）和掩码（Tensor格式），不返回任何值
        """
        assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        if not self._did_reset and masks[0] == 0:  # 如果没有执行过重置
            self._reset()
            self._target_object = observations["objectgoal"]  # 从观测数据中提取目标对象信息并存储到_target_object属性中
        try:
            self._cache_observations(observations)  # 尝试将当前的观测数据缓存起来，通常用于后续的分析或历史记录
        except (
            IndexError
        ):  # 如果缓存过程中发生索引错误（通常是到达地图边界），打印错误信息并抛出StopIteration异常来停止当前 episode
            raise StopIteration
        self._policy_info = {}  # 策略信息重置

    def _initialize(self) -> Tensor:  # 父类占位符，规定接口
        raise NotImplementedError

    def _explore(self, observations: "TensorDict") -> Tensor:
        raise NotImplementedError

    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]:
        """如果当前地图有目标物体，在点云数据中为智能体选择最佳的目标点"""
        if self._object_map.has_object(self._target_object):
            return self._object_map.get_best_object(self._target_object, position)
        else:
            return None

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        # 检查物体地图中是否包含目标物体
        if self._object_map.has_object(self._target_object):
            # 如果有目标物体，获取目标物体的3D点云数据
            target_point_cloud = self._object_map.get_target_cloud(self._target_object)
        else:
            # 如果没有找到目标物体，设置为空数组
            target_point_cloud = np.array([])

        # 构建基础策略信息字典，包含导航和调试所需的核心信息
        policy_info = {
            # 目标物体名称（只取第一个，如果有多个用|分隔的话）
            "target_object": self._target_object.split("|")[0],
            # GPS坐标转换：y坐标取负值（因为Habitat坐标系y轴与实际相反）
            "gps": str(self._observations_cache["robot_xy"] * np.array([1, -1])),
            # 机器人朝向角度，从弧度转换为度数
            "yaw": np.rad2deg(self._observations_cache["robot_heading"]),
            # 布尔值：是否已检测到目标物体
            "target_detected": self._object_map.has_object(self._target_object),
            # 目标物体的3D点云数据（用于导航决策）
            "target_point_cloud": target_point_cloud,
            # 当前导航目标点坐标
            "nav_goal": self._last_goal,
            # 布尔值：是否已调用停止动作
            "stop_called": self._called_stop,
            # 指定哪些信息要在视频底部显示（不覆盖在自我中心图像上）
            "render_below_images": [
                "target_object",  # 目标物体名称将显示在图像下方
            ],
        }

        # 如果不需要可视化，直接返回基础信息
        if not self._visualize:
            return policy_info

        # ================== 可视化部分 ==================
        # 从缓存中获取深度图像并转换为可视化格式
        annotated_depth = self._observations_cache["object_map_rgbd"][0][1] * 255  # 深度值从[0,1]缩放到[0,255]
        annotated_depth = cv2.cvtColor(annotated_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)  # 灰度转RGB三通道

        # 检查是否有检测到的物体掩码
        if self._object_masks.sum() > 0:
            # 如果物体掩码不全为零，说明检测到了物体
            # 找到物体掩码的轮廓
            contours, _ = cv2.findContours(self._object_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # 在RGB图像上绘制物体轮廓（蓝色，线宽2像素）
            annotated_rgb = cv2.drawContours(detections.annotated_frame, contours, -1, (255, 0, 0), 2)
            # 在深度图像上也绘制相同的轮廓
            annotated_depth = cv2.drawContours(annotated_depth, contours, -1, (255, 0, 0), 2)
        else:
            # 如果没有检测到物体，使用原始RGB图像
            annotated_rgb = self._observations_cache["object_map_rgbd"][0][0]

        # 将带注释的图像添加到策略信息中（用于视频生成）
        policy_info["annotated_rgb"] = annotated_rgb  # 带物体轮廓的RGB图像
        policy_info["annotated_depth"] = annotated_depth  # 带物体轮廓的深度图像

        # 如果启用了前沿计算，添加障碍物地图可视化
        if self._compute_frontiers:
            # 获取障碍物地图的可视化图像并转换颜色格式（BGR->RGB）
            policy_info["obstacle_map"] = cv2.cvtColor(self._obstacle_map.visualize(), cv2.COLOR_BGR2RGB)

        # 调试信息处理：如果环境变量中有DEBUG_INFO
        if "DEBUG_INFO" in os.environ:
            # 将debug信息添加到底部显示列表
            policy_info["render_below_images"].append("debug")
            # 获取调试信息内容
            policy_info["debug"] = "debug: " + os.environ["DEBUG_INFO"]

        return policy_info  # 返回完整的策略信息字典

    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
        """
        私有方法，接收numpy数组格式的图像，检测器检测后返回ObjectDetections对象(边界框，类别标签，置信度分数)
        """
        target_classes = self._target_object.split("|")  # 将配置的目标对象字符串按|分割成列表(一个目标物体的多个名称）
        has_coco = (
            any(c in COCO_CLASSES for c in target_classes) and self._load_yolo
        )  # 判断是否有目标类别在coco内，同时yolo是否已加载
        has_non_coco = any(c not in COCO_CLASSES for c in target_classes)  # 没有coco类型目标

        detections = (  # 根据条件选择使用YOLO-World或通用检测器 三元条件运算符
            self._coco_object_detector.predict(img)  # YOLO-World检测器 (现在替换了YOLOv7)
            if has_coco
            else self._object_detector.predict(img, caption=self._non_coco_caption)  # 不含coco，通用
        )
        # 检测器返回的是一个ObjectDetections类的实例
        detections.filter_by_class(target_classes)  # 过滤检测结果，只保留目标类别
        det_conf_threshold = (
            self._coco_threshold if has_coco else self._non_coco_threshold
        )  # 根据检测器类型不同设置不同阈值
        detections.filter_by_conf(det_conf_threshold)  # 过滤置信度低于阈值的结果

        if has_coco and has_non_coco and detections.num_detections == 0:  # 当混合类型检测失效时
            # Retry with non-coco object detector
            detections = self._object_detector.predict(img, caption=self._non_coco_caption)  # 使用通用检测器
            detections.filter_by_class(target_classes)
            detections.filter_by_conf(self._non_coco_threshold)

        return detections

    def _pointnav(self, goal: np.ndarray, stop: bool = False) -> Tensor:
        """
        在观察和给定目标中使用gps和航向传感器计算从机器人当前位置到目标的rho和theta，
        然后使用pointnav policy生成下一个行动
        Calculates rho and theta from the robot's current position to the goal using the
        gps and heading sensors within the observations and the given goal, then uses
        it to determine the next action to take using the pre-trained pointnav policy.

        Args:
            goal (np.ndarray): The goal to navigate to as (x, y), where x and y are in
                meters.
            stop (bool): Whether to stop if we are close enough to the goal.

        """
        masks = torch.tensor(
            [self._num_steps != 0], dtype=torch.bool, device="cuda"
        )  # 如果不是第一步，就设成 True，否则是 False
        if not np.array_equal(goal, self._last_goal):  # 如果目标不是上次坐标
            if np.linalg.norm(goal - self._last_goal) > 0.1:  # 如果变化大于0.1m，说明换了一个目标点
                self._pointnav_policy.reset()  # 重置内部状态
                masks = torch.zeros_like(masks)  # mask设置全0
            self._last_goal = goal  # 最后更新
        robot_xy = self._observations_cache["robot_xy"]  # 从缓存中取出机器人当前位置
        heading = self._observations_cache["robot_heading"]  # 朝向角度 heading
        rho, theta = rho_theta(
            robot_xy, heading, goal
        )  # 得到机器人和目标点之间的距离，目标点相对于机器人正前方的角度偏差
        rho_theta_tensor = torch.tensor(
            [[rho, theta]], device="cuda", dtype=torch.float32
        )  # 把极坐标(rho, theta) 转成张量形式，放到 GPU 上
        obs_pointnav = {  # 构建 policy 的输入 obs_pointnav 字典，两个键值
            "depth": image_resize(
                self._observations_cache["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        self._policy_info["rho_theta"] = np.array([rho, theta])  # 把 (rho, theta) 存起来，方便调试或日志记录
        if rho < self._pointnav_stop_radius and stop:  # 如果机器人离目标小于 stop 半径，并且 stop=True
            self._called_stop = True
            self._policy_info["stop_reason"] = "target_reached"  # 标记停止原因：到达目标
            return self._stop_action  # 返回一个停止动作
        action = self._pointnav_policy.act(
            obs_pointnav, masks, deterministic=True
        )  # 调用pointnav策略（使用模型生成动作）
        return action  # 返回的action就是机器人下一步该做的动作

    def _update_object_map(
        self,
        rgb: np.ndarray,  # rgb图像，用于物体检测和分割
        depth: np.ndarray,  # 深度图像，归一化到[0,1]
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> ObjectDetections:
        """
        读取RGB数组，检测物体，使用sam分割获得物体掩码，BLIP进行vqa验证，生成物体点云

        Updates the object map with the given rgb and depth images, and the given
        transformation matrix from the camera to the episodic coordinate frame.

        参数说明:
        rgb (np.ndarray): 用于更新物体地图的 RGB 图像。用于目标检测和 Mobile SAM 分割，以提取更好的物体点云。
        depth (np.ndarray): 用于更新物体地图的深度图像。该图像已归一化到 [0, 1] 范围，形状为 (高度, 宽度)。
        tf_camera_to_episodic (np.ndarray): 从相机坐标系到 episodic 坐标系的变换矩阵。
        min_depth (float): 深度图像的最小深度值（单位：米）。
        max_depth (float): 深度图像的最大深度值（单位：米）。
        fx (float): 相机在 x 方向上的焦距。
        fy (float): 相机在 y 方向上的焦距。

        返回:
        ObjectDetections: 来自目标检测器的检测结果。

        Args:
            rgb (np.ndarray): The rgb image to use for updating the object map. Used for
                object detection and Mobile SAM segmentation to extract better object
                point clouds.
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).
            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.

        Returns:
            ObjectDetections: The object detections from the object detector.
        """
        detections = self._get_object_detections(rgb)  # rgb图像检测物体类型
        """
        1. 边界框 (boxes)
        - 类型: torch.Tensor
        - 形状: (N, 4) N个检测框
        - 格式: [x1, y1, x2, y2] 左上角和右下角坐标
        - 范围: 归一化到[0,1]之间
        
        2. 置信度分数 (logits)
        - 类型: torch.Tensor
        - 形状: (N,) N个置信度分数
        - 范围: [0,1] 表示检测的可信度

        3. 类别标签 (phrases)
        - 类型: List[str]
        - 内容: 检测到的物体类别名称
        - 例子: ["chair", "table", "bed"]

        4. 原始图像 (image_source)        
        - 类型: np.ndarray
        - 形状: (H, W, 3) RGB图像
        - 用途: 用于可视化和后续处理
        """
        height, width = rgb.shape[:2]
        self._object_masks = np.zeros((height, width), dtype=np.uint8)  # 初始化物体掩码
        if (
            np.array_equal(depth, np.ones_like(depth)) and detections.num_detections > 0
        ):  # 如果深度全为1（即无效深度值），且检测到物体
            depth = self._infer_depth(rgb, min_depth, max_depth)  # 使用深度估计模型推断深度

            # 更新观察缓存中的深度信息
            obs = list(self._observations_cache["object_map_rgbd"][0])
            obs[1] = depth
            self._observations_cache["object_map_rgbd"][0] = tuple(obs)

        """ 根据detection，生成物掩码，进而生成所有物体的点云 """
        for idx in range(len(detections.logits)):  # 遍历所有检测到的类别的置信度
            # 将归一化的边界框坐标转换为像素坐标
            bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
            # 使用Mobile SAM模型分割物体，获取精确的物体掩码
            object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist())

            # If we are using vqa, then use the BLIP2 model to visually confirm whether
            # the contours are actually correct.
            # 如果启用VQA（视觉问答），使用BLIP2模型验证检测是否正确
            if self._use_vqa:
                # 在图像上绘制检测到的轮廓
                contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255, 0, 0), 2)
                # 构建验证问题
                question = f"Question: {self._vqa_prompt}"
                if not detections.phrases[idx].endswith("ing"):
                    question += "a "
                question += detections.phrases[idx] + "? Answer:"
                # 使用VQA模型验证
                answer = self._vqa.ask(annotated_rgb, question)
                if not answer.lower().startswith("yes"):
                    continue  # 如果失败，跳过该物体

            self._object_masks[object_mask > 0] = 1  # 更新物体掩码图
            self._object_map.update_map(  # 更新物体地图，生成物体点云字典，过滤、添加标识符
                self._target_object,
                depth,
                object_mask,
                tf_camera_to_episodic,
                min_depth,
                max_depth,
                fx,
                fy,
            )

        cone_fov = get_fov(fx, depth.shape[1])  # 获取fov
        self._object_map.update_explored(tf_camera_to_episodic, max_depth, cone_fov)  # 保留视野范围内的点云

        return detections

    def _cache_observations(self, observations: "TensorDict") -> None:
        """Extracts the rgb, depth, and camera transform from the observations.

        Args:
            observations ("TensorDict"): The observations from the current timestep.
        """
        raise NotImplementedError

    def _infer_depth(self, rgb: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        """Infers the depth image from the rgb image.
        从RGB图像推断深度图像
        定义接口规范而非具体实现
        Args:
            rgb (np.ndarray): The rgb image to infer the depth from.

        Returns:
            np.ndarray: The inferred depth image.
        """
        raise NotImplementedError


@dataclass  # 装饰器，自动生成特殊方法
class VLFMConfig:
    name: str = "HabitatITMPolicy"
    text_prompt: str = "Seems like there is a target_object ahead."
    pointnav_policy_path: str = "data/pointnav_weights.pth"
    depth_image_shape: Tuple[int, int] = (224, 224)
    pointnav_stop_radius: float = 0.9
    use_max_confidence: bool = False
    object_map_erosion_size: int = 5
    exploration_thresh: float = 0.0
    obstacle_map_area_threshold: float = 1.5  # in square meters
    min_obstacle_height: float = 0.61
    max_obstacle_height: float = 0.88
    hole_area_thresh: int = 100000
    use_vqa: bool = False
    vqa_prompt: str = "Is this "
    coco_threshold: float = 0.8
    non_coco_threshold: float = 0.4
    agent_radius: float = 0.18

    # Additional parameters from BaseObjectNavPolicy.__init__
    visualize: bool = True
    compute_frontiers: bool = True
    discrete_actions: bool = False
    sync_explored_areas: bool = False

    # Detection-based policy specific parameters
    similarity_threshold: float = 0.3
    confidence_weight: float = 0.4
    similarity_weight: float = 0.6
    use_embeddings: bool = False

    # ITM policy specific parameters
    use_yoloworld: bool = False  # Use original YOLO-World for similarity calculation
    use_ultralytics_yoloworld: bool = True  # Use Ultralytics YOLO-World for similarity calculation

    @classmethod  # type: ignore
    @property
    def kwaarg_names(cls) -> List[str]:
        # This returns all the fields listed above, except the name field
        return [f.name for f in fields(VLFMConfig) if f.name != "name"]


cs = ConfigStore.instance()
cs.store(group="policy", name="vlfm_config_base", node=VLFMConfig())
