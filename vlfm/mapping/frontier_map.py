# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
# Modifications Copyright (c) 2026 Yikang.

from typing import List, Tuple

import numpy as np

from vlfm.vlm.blip2itm import BLIP2ITMClient


class Frontier:
    def __init__(self, xyz: np.ndarray, cosine: float):
        self.xyz = xyz
        self.cosine = cosine  # 语义相似度


class FrontierMap:
    frontiers: List[Frontier] = []

    def __init__(self, encoding_type: str = "cosine"):
        self.encoder: BLIP2ITMClient = BLIP2ITMClient()  # 使用BLIP2ITM模型进行匹配

    def reset(self) -> None:
        self.frontiers = []

    def update(self, frontier_locations: List[np.ndarray], curr_image: np.ndarray, text: str) -> None:
        """
        输入：边界数组，当前图像，对比边界，删除旧的添加新的，并为新的边界计算相似度
        Takes in a list of frontier coordinates and the current image observation from
        the robot. Any stored frontiers that are not present in the given list are
        removed. Any frontiers in the given list that are not already stored are added.
        When these frontiers are added, their cosine field is set to the encoding
        of the given image. The image will only be encoded if a new frontier is added.

        Args:
            frontier_locations (List[np.ndarray]): A list of frontier coordinates.
            curr_image (np.ndarray): The current image observation from the robot.
            text (str): The text to compare the image to.
        """
        # Remove any frontiers that are not in the given list. Use np.array_equal.
        # 过滤不在输入列表的边界点
        self.frontiers = [  # [表达式 for 变量 in 可迭代对象 if 条件]
            frontier
            for frontier in self.frontiers
            if any(np.array_equal(frontier.xyz, location) for location in frontier_locations)
        ]  # 遍历self.frontiers列表中的每一个前沿点对象
        # 检查它是否与frontier_locations中的任何一个位置相等
        # 只要有一个位置相等，就返回True，将这个frontier加入列表

        # Add any frontiers that are not already stored. Set their image field to the
        # given image.
        # 添加还没有存储的边界
        cosine = None
        for location in frontier_locations:
            if not any(np.array_equal(frontier.xyz, location) for frontier in self.frontiers):  # 新发现的边界点
                if cosine is None:  # 未计算过相似度
                    cosine = self._encode(curr_image, text)  # 计算相似度
                self.frontiers.append(Frontier(location, cosine))  # 将新创建Frontier对象添加到frontiers数组中

    def _encode(self, image: np.ndarray, text: str) -> float:
        """
        Encodes the given image using the encoding type specified in the constructor.

        Args:
            image (np.ndarray): The image to encode.

        Returns:

        """
        return self.encoder.cosine(image, text)  # 调用BLIP2模型计算图像和文本的余弦相似度

    def sort_waypoints(self) -> Tuple[np.ndarray, List[float]]:
        """
        Returns the frontier with the highest cosine and the value of that cosine.
        """
        # Use np.argsort to get the indices of the sorted cosines
        cosines = [f.cosine for f in self.frontiers]  # 提取frontiers中的所有相似度
        waypoints = [f.xyz for f in self.frontiers]  # 提取frontiers中的边界点坐标
        sorted_inds = np.argsort([-c for c in cosines])  # 相似度降序，返回索引
        sorted_values = [cosines[i] for i in sorted_inds]  # 降序相似度值
        sorted_frontiers = np.array([waypoints[i] for i in sorted_inds])  # 排序后的对应坐标

        return sorted_frontiers, sorted_values  # 返回排序后的边界点坐标和对应相似度
