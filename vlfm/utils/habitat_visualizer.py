# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

# ============================================================================
# 导入模块说明 (Import Modules Documentation)
# ============================================================================
from typing import Any, Dict, List, Optional, Tuple  # 类型提示，用于函数参数和返回值的类型标注

import cv2  # OpenCV库，用于图像处理操作（如颜色空间转换）
import numpy as np  # NumPy库，用于数组和矩阵操作
from frontier_exploration.utils.general_utils import xyz_to_habitat  # 坐标系转换：将XYZ坐标转换为Habitat坐标系
from habitat.utils.common import flatten_dict  # Habitat工具函数：将嵌套字典扁平化
from habitat.utils.visualizations import maps  # Habitat地图可视化模块
from habitat.utils.visualizations.maps import MAP_TARGET_POINT_INDICATOR  # 地图上目标点的指示符常量
from habitat.utils.visualizations.utils import overlay_text_to_image  # Habitat文本叠加工具
from habitat_baselines.common.tensor_dict import TensorDict  # Habitat基线的张量字典类

from vlfm.utils.geometry_utils import transform_points  # VLFM几何工具：点变换函数
from vlfm.utils.img_utils import (  # VLFM图像工具模块：
    reorient_rescale_map,    # 地图重新定向和缩放
    resize_image,           # 单图像尺寸调整
    resize_images,          # 多图像尺寸调整
    rotate_image,           # 图像旋转
)
from vlfm.utils.visualization import add_text_to_image, pad_images  # VLFM可视化工具：文本叠加和图像填充


class HabitatVis:
    """
    Habitat环境可视化器类 (Habitat Environment Visualizer Class)
    
    用于收集、处理和生成Habitat导航环境的可视化帧序列。
    支持RGB图像、深度图像、地图和自定义可视化地图的组合显示。
    """
    def __init__(self) -> None:
        """初始化可视化器，创建空的数据存储列表"""
        # ======================== 图像数据存储 ========================
        self.rgb: List[np.ndarray] = []          # RGB图像序列：存储每一帧的RGB观察图像
        self.depth: List[np.ndarray] = []        # 深度图像序列：存储每一帧的深度图像（转换为RGB格式）
        self.maps: List[np.ndarray] = []         # 地图图像序列：存储Habitat生成的俯视图地图
        self.vis_maps: List[List[np.ndarray]] = [] # 可视化地图序列：存储自定义地图（如obstacle_map, value_map）
        self.texts: List[List[str]] = []         # 文本信息序列：存储每帧要显示的文本信息
        
        # ======================== 状态标志 ========================
        self.using_vis_maps = False              # 标志：是否使用自定义可视化地图
        self.using_annotated_rgb = False         # 标志：是否使用带注释的RGB图像
        self.using_annotated_depth = False       # 标志：是否使用带注释的深度图像

    def reset(self) -> None:
        """重置可视化器状态，清空所有累积的数据"""
        self.rgb = []                            # 清空RGB图像列表
        self.depth = []                          # 清空深度图像列表
        self.maps = []                           # 清空地图图像列表
        self.vis_maps = []                       # 清空自定义可视化地图列表
        self.texts = []                          # 清空文本信息列表
        self.using_annotated_rgb = False         # 重置RGB注释标志
        self.using_annotated_depth = False       # 重置深度注释标志

    def collect_data(
        self,
        observations: TensorDict,          # 环境观察数据：包含RGB、depth等传感器数据
        infos: List[Dict[str, Any]],       # 环境信息：包含地图、指标等元数据
        policy_info: List[Dict[str, Any]], # 策略信息：包含可视化数据和自定义地图
    ) -> None:
        """
        收集单步数据用于可视化
        
        处理流程：
        1. 处理深度图像（原始或带注释）
        2. 处理RGB图像（原始或带注释）
        3. 在地图上可视化目标点云
        4. 生成Habitat俯视图
        5. 处理自定义可视化地图
        6. 收集文本信息
        """
        assert len(infos) == 1, "Only support one environment for now"  # 断言：当前只支持单环境

        # ======================== 处理深度图像 ========================
        if "annotated_depth" in policy_info[0]:
            # 如果策略提供了带注释的深度图，直接使用
            depth = policy_info[0]["annotated_depth"]
            self.using_annotated_depth = True  # 标记使用带注释深度图
        else:
            # 否则从观察数据中获取原始深度图
            depth = (observations["depth"][0].cpu().numpy() * 255.0).astype(np.uint8)  # 深度值缩放到0-255范围
            depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)  # 将灰度图转换为RGB格式（3通道）
        depth = overlay_frame(depth, infos[0])  # 在深度图上叠加文本信息（如指标数据）
        self.depth.append(depth)  # 将处理后的深度图添加到序列中

        # ======================== 处理RGB图像 ========================
        if "annotated_rgb" in policy_info[0]:
            # 如果策略提供了带注释的RGB图，直接使用
            rgb = policy_info[0]["annotated_rgb"]
            self.using_annotated_rgb = True  # 标记使用带注释RGB图
        else:
            # 否则从观察数据中获取原始RGB图
            rgb = observations["rgb"][0].cpu().numpy()  # 将张量转换为numpy数组
        self.rgb.append(rgb)  # 将RGB图添加到序列中

        # ======================== 可视化目标点云 ========================
        # 在地图上标记目标点云的位置（修改infos中的地图数据）
        color_point_cloud_on_map(infos, policy_info)

        # ======================== 生成Habitat俯视图 ========================
        # 创建带有智能体位置和朝向的彩色俯视图，高度匹配深度图
        map = maps.colorize_draw_agent_and_fit_to_height(infos[0]["top_down_map"], self.depth[0].shape[0])
        self.maps.append(map)  # 将地图图像添加到序列中
        
        # ======================== 处理自定义可视化地图 ========================
        vis_map_imgs = [
            self._reorient_rescale_habitat_map(infos, policy_info[0][vkey])  # 重新定向和缩放地图
            for vkey in ["obstacle_map", "value_map"]  # 处理障碍物地图和价值地图
            if vkey in policy_info[0]  # 仅处理存在的地图
        ]
        if vis_map_imgs:
            # 如果有自定义可视化地图，标记并添加到序列
            self.using_vis_maps = True
            self.vis_maps.append(vis_map_imgs)
            
        # ======================== 收集文本信息 ========================
        text = [
            policy_info[0][text_key]  # 从策略信息中提取文本内容
            for text_key in policy_info[0].get("render_below_images", [])  # 获取要渲染的文本键列表
            if text_key in policy_info[0]  # 仅处理存在的文本键
        ]
        self.texts.append(text)  # 将文本信息添加到序列中

    def flush_frames(self, failure_cause: str) -> List[np.ndarray]:
        """
        输出所有累积的帧并返回完整的可视化视频序列
        
        Args:
            failure_cause: episode结束的原因（如"success", "timeout", "collision"等）
            
        Returns:
            List[np.ndarray]: 处理后的帧序列，每帧都是组合的可视化图像
        """
        # ======================== 处理时序偏移问题 ========================
        # 由于带注释的帧实际上有一步延迟，需要调整时序
        # 将第一帧移到最后，添加占位符帧到末尾（之后会被移除）
        if self.using_annotated_rgb is not None:
            self.rgb.append(self.rgb.pop(0))      # RGB：将第一帧移动到最后
        if self.using_annotated_depth is not None:
            self.depth.append(self.depth.pop(0))  # 深度：将第一帧移动到最后
        if self.using_vis_maps:  # 成本地图也有一步延迟
            self.vis_maps.append(self.vis_maps.pop(0))  # 可视化地图：将第一组移动到最后

        # ======================== 生成帧序列 ========================
        frames = []
        num_frames = len(self.depth) - 1  # 最后一帧来自下一个episode，需要移除
        for i in range(num_frames):
            # 为每一步创建组合帧
            frame = self._create_frame(
                self.depth[i],      # 当前帧的深度图
                self.rgb[i],        # 当前帧的RGB图
                self.maps[i],       # 当前帧的地图
                self.vis_maps[i],   # 当前帧的可视化地图组
                self.texts[i],      # 当前帧的文本信息
            )
            # 在帧顶部添加失败原因文本
            failure_cause_text = "Failure cause: " + failure_cause
            frame = add_text_to_image(frame, failure_cause_text, top=True)
            frames.append(frame)

        # ======================== 后处理帧序列 ========================
        if len(frames) > 0:
            frames = pad_images(frames, pad_from_top=True)  # 统一帧尺寸，从顶部填充

        # 调整帧尺寸为标准大小
        frames = [resize_image(f, 480 * 2) for f in frames]  # 每帧调整为960像素宽度

        # ======================== 清理状态 ========================
        self.reset()  # 重置可视化器状态，准备下一个episode

        return frames  # 返回完整的视频帧序列

    @staticmethod
    def _reorient_rescale_habitat_map(infos: List[Dict[str, Any]], vis_map: np.ndarray) -> np.ndarray:
        """
        重新定向和缩放可视化地图以匹配Habitat地图的显示格式
        
        Args:
            infos: 环境信息列表，包含地图元数据
            vis_map: 要处理的可视化地图（如obstacle_map, value_map）
            
        Returns:
            np.ndarray: 处理后的地图图像
        """
        # ======================== 旋转匹配智能体起始朝向 ========================
        # 根据智能体在episode开始时的朝向旋转成本地图
        start_yaw = infos[0]["start_yaw"]  # 获取智能体起始偏航角
        if start_yaw != 0.0:
            # 如果起始朝向不是0度，则旋转地图以匹配
            vis_map = rotate_image(vis_map, start_yaw, border_value=(255, 255, 255))

        # ======================== 处理地图纵横比 ========================
        # 如果对应的Habitat地图高度大于宽度，则将图像顺时针旋转90度
        habitat_map = infos[0]["top_down_map"]["map"]  # 获取Habitat原始地图
        if habitat_map.shape[0] > habitat_map.shape[1]:
            vis_map = np.rot90(vis_map, 1)  # 旋转90度（顺时针）

        # ======================== 标准化定向和缩放 ========================
        # 应用标准的地图重新定向和缩放处理
        vis_map = reorient_rescale_map(vis_map)

        return vis_map  # 返回处理后的地图

    @staticmethod
    def _create_frame(
        depth: np.ndarray,            # 深度图像 (H, W, 3)
        rgb: np.ndarray,              # RGB图像 (H, W, 3)  
        map: np.ndarray,              # 地图图像，3通道RGB，可能与depth/rgb尺寸不同
        vis_map_imgs: List[np.ndarray], # 其他地图图像列表，每个都是3通道RGB，可能尺寸不同
        text: List[str],              # 要渲染在图像上方的字符串列表
    ) -> np.ndarray:
        """
        使用所有给定图像创建单个组合帧
        
        布局策略：
        1. 深度和RGB图像垂直堆叠（左侧）
        2. 所有地图合并为单独图像（右侧）
        3. 两个图像水平拼接（深度-RGB在左，地图在右）
        
        地图组合布局（至少2行1列）：
        - 'map'参数位于左上角
        - 'vis_map_imgs'的第一个元素位于左下角
        - 如果'vis_map_imgs'有多个元素：第二个在右上角，第三个在右下角，以此类推
        
        Args:
            depth: 深度图像 (H, W, 3)
            rgb: RGB图像 (H, W, 3)
            map: 地图图像，3通道RGB图像，但可能与depth和rgb的形状不同
            vis_map_imgs: 其他地图图像列表，每个都是3通道RGB图像，但可能尺寸不同
            text: 要渲染在图像上方的字符串列表
            
        Returns:
            np.ndarray: 组合的帧图像
        """
        # ======================== 垂直堆叠深度和RGB图像 ========================
        depth_rgb = np.vstack((depth, rgb))  # 将深度图放在上方，RGB图放在下方

        # ======================== 准备地图图像组合 ========================
        map_imgs = [map] + vis_map_imgs  # 将主地图和可视化地图合并为一个列表
        if len(map_imgs) % 2 == 1:
            # 如果地图数量为奇数，添加白色占位符图像以保持2列布局
            map_imgs.append(np.ones_like(map_imgs[-1]) * 255)

        # ======================== 创建地图网格布局 ========================
        even_index_imgs = map_imgs[::2]    # 偶数索引图像：0, 2, 4, ... (左列)
        odd_index_imgs = map_imgs[1::2]    # 奇数索引图像：1, 3, 5, ... (右列)
        
        # 创建顶行：水平拼接偶数索引图像，高度匹配
        top_row = np.hstack(resize_images(even_index_imgs, match_dimension="height"))
        # 创建底行：水平拼接奇数索引图像，高度匹配
        bottom_row = np.hstack(resize_images(odd_index_imgs, match_dimension="height"))

        # ======================== 组合最终帧 ========================
        # 垂直堆叠顶行和底行，宽度匹配
        frame = np.vstack(resize_images([top_row, bottom_row], match_dimension="width"))
        # 调整深度-RGB组合和地图组合的高度匹配，然后水平拼接
        depth_rgb, frame = resize_images([depth_rgb, frame], match_dimension="height")
        frame = np.hstack((depth_rgb, frame))  # 深度-RGB在左，地图在右

        # ======================== 添加文本标签 ========================
        # 逆序遍历文本列表，在帧顶部添加文本（最后添加的在最上方）
        for t in text[::-1]:
            frame = add_text_to_image(frame, t, top=True)

        return frame  # 返回完整的组合帧


def sim_xy_to_grid_xy(
    upper_bound: Tuple[int, int],      # 网格上边界 (x_max, y_max)
    lower_bound: Tuple[int, int],      # 网格下边界 (x_min, y_min)  
    grid_resolution: Tuple[int, int],  # 网格分辨率 (rows, cols)
    sim_xy: np.ndarray,               # 仿真坐标数组 (N, 2)，每行为[x, y]
    remove_duplicates: bool = True,    # 是否移除重复的网格坐标
) -> np.ndarray:
    """
    将仿真世界坐标转换为网格坐标
    
    坐标系转换说明：
    - 仿真坐标：连续的世界坐标系，单位通常为米
    - 网格坐标：离散的像素/单元坐标系，用于地图索引
    
    Args:
        upper_bound: 网格的上边界 (x_max, y_max)
        lower_bound: 网格的下边界 (x_min, y_min)  
        grid_resolution: 网格的分辨率 (行数, 列数)
        sim_xy: 仿真坐标的numpy数组，形状为(N, 2)，每行代表一个2D坐标点
        remove_duplicates: 是否移除重复的网格坐标
        
    Returns:
        np.ndarray: 转换后的2D网格坐标数组，形状为(M, 2)，每行为[row, col]
    """
    # ======================== 计算网格单元大小 ========================
    grid_size = np.array([
        abs(upper_bound[1] - lower_bound[1]) / grid_resolution[0],  # Y方向单元大小
        abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],  # X方向单元大小
    ])
    
    # ======================== 坐标变换 ========================
    # 1. 将仿真坐标平移到原点 (减去下边界)
    # 2. 除以单元大小得到网格索引
    # 3. 转换为整数坐标
    # 注意：lower_bound[::-1] 是为了匹配xy到yx的坐标轴映射
    grid_xy = ((sim_xy - lower_bound[::-1]) / grid_size).astype(int)

    # ======================== 去重处理 ========================
    if remove_duplicates:
        grid_xy = np.unique(grid_xy, axis=0)  # 移除重复的坐标点

    return grid_xy  # 返回网格坐标数组


def color_point_cloud_on_map(infos: List[Dict[str, Any]], policy_info: List[Dict[str, Any]]) -> None:
    """
    在Habitat俯视图地图上标记目标点云的位置
    
    功能说明：
    将策略发现的目标点云从episode坐标系转换到地图坐标系，
    并在俯视图上用特殊颜色标记这些点的位置，用于可视化目标检测结果。
    
    Args:
        infos: 环境信息列表，包含地图元数据和变换矩阵
        policy_info: 策略信息列表，包含检测到的目标点云数据
    """
    # ======================== 检查点云数据 ========================
    if len(policy_info[0]["target_point_cloud"]) == 0:
        return  # 如果没有检测到目标点云，直接返回

    # ======================== 获取地图参数 ========================
    upper_bound = infos[0]["top_down_map"]["upper_bound"]          # 地图上边界
    lower_bound = infos[0]["top_down_map"]["lower_bound"]          # 地图下边界  
    grid_resolution = infos[0]["top_down_map"]["grid_resolution"]  # 地图网格分辨率
    tf_episodic_to_global = infos[0]["top_down_map"]["tf_episodic_to_global"]  # episode到全局坐标的变换矩阵

    # ======================== 坐标系变换序列 ========================
    # 1. 提取点云的XYZ坐标（忽略其他特征，如颜色、置信度等）
    cloud_episodic_frame = policy_info[0]["target_point_cloud"][:, :3]  # 形状: (N, 3)
    
    # 2. 从episode坐标系变换到全局坐标系
    cloud_global_frame_xyz = transform_points(tf_episodic_to_global, cloud_episodic_frame)
    
    # 3. 从XYZ坐标转换到Habitat坐标系（不同的轴定义）
    cloud_global_frame_habitat = xyz_to_habitat(cloud_global_frame_xyz)
    
    # 4. 提取2D坐标用于地图投影（选择Z和X轴，对应Habitat的俯视图）
    cloud_global_frame_habitat_xy = cloud_global_frame_habitat[:, [2, 0]]  # 形状: (N, 2)

    # ======================== 转换为网格坐标 ========================
    grid_xy = sim_xy_to_grid_xy(
        upper_bound,                    # 地图边界
        lower_bound, 
        grid_resolution,               # 网格分辨率
        cloud_global_frame_habitat_xy, # 2D坐标点
        remove_duplicates=True,        # 移除重复点以避免重复标记
    )

    # ======================== 在地图上标记目标点 ========================
    new_map = infos[0]["top_down_map"]["map"].copy()  # 复制原始地图以避免修改原数据
    # 将目标点的位置标记为特殊颜色（MAP_TARGET_POINT_INDICATOR是预定义的颜色值）
    new_map[grid_xy[:, 0], grid_xy[:, 1]] = MAP_TARGET_POINT_INDICATOR

    # ======================== 更新地图数据 ========================
    infos[0]["top_down_map"]["map"] = new_map  # 将修改后的地图写回信息字典


def overlay_frame(frame: np.ndarray, info: Dict[str, Any], additional: Optional[List[str]] = None) -> np.ndarray:
    """
    在图像帧上叠加信息字典中的文本内容
    
    功能说明：
    从环境信息字典中提取各种指标和状态信息，格式化为文本行，
    然后叠加到图像上，用于实时监控和调试可视化。
    
    Args:
        frame: 要叠加文本的图像帧
        info: 环境信息字典，包含各种指标和状态数据
        additional: 可选的额外文本行列表
        
    Returns:
        np.ndarray: 叠加了文本的图像帧
    """
    
    # ======================== 处理信息字典 ========================
    lines = []  # 存储要显示的文本行
    flattened_info = flatten_dict(info)  # 将嵌套字典扁平化为单层字典
    
    # ======================== 格式化信息为文本 ========================
    for k, v in flattened_info.items():
        if isinstance(v, str):
            # 字符串值：直接显示
            lines.append(f"{k}: {v}")
        else:
            try:
                # 数值：格式化为2位小数
                lines.append(f"{k}: {v:.2f}")
            except TypeError:
                # 无法格式化的类型：跳过（如复杂对象、None等）
                pass
                
    # ======================== 添加额外信息 ========================
    if additional is not None:
        lines.extend(additional)  # 将额外的文本行添加到列表末尾

    # ======================== 叠加文本到图像 ========================
    frame = overlay_text_to_image(frame, lines, font_size=0.25)  # 使用小字体叠加所有文本行

    return frame  # 返回带有叠加文本的图像
