# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
# Modifications Copyright (c) 2026 Yikang.

from typing import Dict, Union

import cv2
import numpy as np
import open3d as o3d

from vlfm.utils.geometry_utils import (
    extract_yaw,
    get_point_cloud,
    transform_points,
    within_fov_cone, # 判断是否在视锥中
)


class ObjectPointCloudMap: 
    clouds: Dict[str, np.ndarray] = {} # 存储不同类别对象的点云字典
    use_dbscan: bool = True # 是否使用DBSCAN聚类过滤点云

    def __init__(self, erosion_size: float) -> None:
        self._erosion_size = erosion_size  # 腐蚀核大小
        self.last_target_coord: Union[np.ndarray, None] = None

    def reset(self) -> None:
        self.clouds = {}
        self.last_target_coord = None

    def has_object(self, target_class: str) -> bool:
        return target_class in self.clouds and len(self.clouds[target_class]) > 0

    def update_map(
        self,
        object_name: str, # 对象类别名称
        depth_img: np.ndarray,  # 图像深度 已归一化
        object_mask: np.ndarray, # 物体掩码
        tf_camera_to_episodic: np.ndarray, # 相机到全局坐标系的变换矩阵
        min_depth: float, # 最小有效深度 
        max_depth: float, # 最大有效深度
        fx: float,  # 焦距
        fy: float,
    ) -> None:
        """Updates the object map with the latest information from the agent.
            为偏移的图片全图加上标识符
            对于非偏移图片，深度范围外的点加入标识符，方便后续处理
            删除最近点距离智能体过近的图片
        """
        local_cloud = self._extract_object_cloud(depth_img, object_mask, min_depth, max_depth, fx, fy) # 提取点云
        
        if len(local_cloud) == 0:
            return

        # For second-class, bad detections that are too offset or out of range, we
        # assign a random number to the last column of its point cloud that can later
        # be used to identify which points came from the same detection.

        if too_offset(object_mask): # 对于偏移严重的检测，给一个随机标识符
            within_range = np.ones_like(local_cloud[:, 0]) * np.random.rand() 
        else:
            # Mark all points of local_cloud whose distance from the camera is too far
            # as being out of range
            within_range = (local_cloud[:, 0] <= max_depth * 0.95) * 1.0  # 获取点云中所有x坐标(深度方向)，比较是否小于最大深度的95%，将bool转化为浮点数，5% margin
            # 1.0 表示在有效范围内 All values of 1 in within_range will be considered within range, and all 
            # 0.0 表示超出有效范围 values of 0 will be considered out of range; these 0s need to be
            # assigned with a random number so that they can be identified later.
            
            within_range = within_range.astype(np.float32) # 转换32位浮点数
            within_range[within_range == 0] = np.random.rand() # 每个超出范围的点生成一个随机数
 
        global_cloud = transform_points(tf_camera_to_episodic, local_cloud) # 转化到全局系
        global_cloud = np.concatenate((global_cloud, within_range[:, None]), axis=1) # 添加范围识别列
 
        curr_position = tf_camera_to_episodic[:3, 3] # 当前位置
        closest_point = self._get_closest_point(global_cloud, curr_position)
        dist = np.linalg.norm(closest_point[:3] - curr_position)
        if dist < 0.1: # 距离小于0.1米，认为是不可信检测，丢弃当前object的点云
            # Object is too close to trust as a valid object
            return

        # 更新或添加点云数据（根据物体名字）
        if object_name in self.clouds: 
            self.clouds[object_name] = np.concatenate((self.clouds[object_name], global_cloud), axis=0)
        else:
            self.clouds[object_name] = global_cloud

    def get_best_object(self, target_class: str, curr_position: np.ndarray) -> np.ndarray:
        '''
            在点云数据中为智能体选择最佳的目标点 获取点云-找最近点-判断目标稳定性（变化显著时更新）
        '''
        target_cloud = self.get_target_cloud(target_class) # 获取目标点云

        closest_point_2d = self._get_closest_point(target_cloud, curr_position)[:2] # 寻找最近点，提取x,y维度数据

        if self.last_target_coord is None: # 如果是第一次选择目标
            self.last_target_coord = closest_point_2d # 直接使用找到的最近点
        else:
            # Do NOT update self.last_target_coord if:
            # 1. the closest point is only slightly different 最近点只有很小的变化
            # 2. the closest point is a little different, but the agent is too far for 
            #    the difference to matter much 智能体距离太远
            delta_dist = np.linalg.norm(closest_point_2d - self.last_target_coord) 
            if delta_dist < 0.1:
                # closest point is only slightly different
                return self.last_target_coord
            elif delta_dist < 0.5 and np.linalg.norm(curr_position - closest_point_2d) > 2.0: # 显著变化时更新
                # closest point is a little different, but the agent is too far for
                # the difference to matter much
                return self.last_target_coord
            else:
                self.last_target_coord = closest_point_2d

        return self.last_target_coord

    def update_explored(self, tf_camera_to_episodic: np.ndarray, max_depth: float, cone_fov: float) -> None:
        """
        获取当前相机的位置和朝向
        对于每个对象的点云，检查哪些点现在位于相机的视野范围内
        移除那些原本被标记为超出范围但现在进入范围内的点
        保留原本就在范围内的点（这些被认为是有效的检测）
        This method will remove all point clouds in self.clouds that were originally
        detected to be out-of-range, but are now within range. This is just a heuristic
        that suppresses ephemeral false positives that we now confirm are not actually
        target objects.

        Args:
            tf_camera_to_episodic: The transform from the camera to the episode frame.
            max_depth: The maximum distance from the camera that we consider to be
                within range.
            cone_fov: The field of view of the camera.
        """
        camera_coordinates = tf_camera_to_episodic[:3, 3]
        camera_yaw = extract_yaw(tf_camera_to_episodic)

        for obj in self.clouds:
            within_range = within_fov_cone(
                camera_coordinates,
                camera_yaw,
                cone_fov,
                max_depth * 0.5,
                self.clouds[obj],
            )
            range_ids = set(within_range[..., -1].tolist())
            for range_id in range_ids:
                if range_id == 1:
                    # Detection was originally within range
                    continue
                # Remove all points from self.clouds[obj] that have the same range_id
                self.clouds[obj] = self.clouds[obj][self.clouds[obj][..., -1] != range_id]

    def get_target_cloud(self, target_class: str) -> np.ndarray:
        target_cloud = self.clouds[target_class].copy()
        # Determine whether any points are within range
        # 选择标识符为1的点
        within_range_exists = np.any(target_cloud[:, -1] == 1)
        if within_range_exists:
            # Filter out all points that are not within range
            target_cloud = target_cloud[target_cloud[:, -1] == 1]
        return target_cloud # 返回目标点云

    def _extract_object_cloud( # 提取物体点云
        self,
        depth: np.ndarray,
        object_mask: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> np.ndarray:
        final_mask = object_mask * 255 # 转化为8位无符号整型
        final_mask = cv2.erode(final_mask, None, iterations=int(self._erosion_size))  # type: ignore 进行腐蚀

        valid_depth = depth.copy() # 复制深度
        valid_depth[valid_depth == 0] = 1  # 将空洞设置为1，避免计算问题 set all holes (0) to just be far (1)
        valid_depth = valid_depth * (max_depth - min_depth) + min_depth # 将归一化后的深度值反归一化
        cloud = get_point_cloud(valid_depth, final_mask, fx, fy) # 生成点云
        cloud = get_random_subarray(cloud, 5000) # 随机采样5000个点
        if self.use_dbscan:
            cloud = open3d_dbscan_filtering(cloud) # 使用DBSCAN聚类算法过滤噪声点，保留主要物体点云簇

        return cloud

    def _get_closest_point(self, cloud: np.ndarray, curr_position: np.ndarray) -> np.ndarray:
        ndim = curr_position.shape[0]
        '''
        寻找最近的点
        '''
        if self.use_dbscan:
            # Return the point that is closest to curr_position, which is 2D
            closest_point = cloud[np.argmin(np.linalg.norm(cloud[:, :ndim] - curr_position, axis=1))]
        else:
            # Calculate the Euclidean distance from each point to the reference point
            if ndim == 2:
                ref_point = np.concatenate((curr_position, np.array([0.5])))
            else:
                ref_point = curr_position
            distances = np.linalg.norm(cloud[:, :3] - ref_point, axis=1)

            # Use argsort to get the indices that would sort the distances
            sorted_indices = np.argsort(distances)

            # Get the top 20% of points
            percent = 0.25
            top_percent = sorted_indices[: int(percent * len(cloud))]
            try:
                median_index = top_percent[int(len(top_percent) / 2)]
            except IndexError:
                median_index = 0
            closest_point = cloud[median_index]
        return closest_point


def open3d_dbscan_filtering(points: np.ndarray, eps: float = 0.2, min_points: int = 100) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Perform DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(eps, min_points))

    # Count the points in each cluster
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    # Exclude noise points, which are given the label -1
    non_noise_labels_mask = unique_labels != -1
    non_noise_labels = unique_labels[non_noise_labels_mask]
    non_noise_label_counts = label_counts[non_noise_labels_mask]

    if len(non_noise_labels) == 0:  # only noise was detected
        return np.array([])

    # Find the label of the largest non-noise cluster
    largest_cluster_label = non_noise_labels[np.argmax(non_noise_label_counts)]

    # Get the indices of points in the largest non-noise cluster
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]

    # Get the points in the largest non-noise cluster
    largest_cluster_points = points[largest_cluster_indices]

    return largest_cluster_points


def visualize_and_save_point_cloud(point_cloud: np.ndarray, save_path: str) -> None:
    """Visualizes an array of 3D points and saves the visualization as a PNG image.

    Args:
        point_cloud (np.ndarray): Array of 3D points with shape (N, 3).
        save_path (str): Path to save the PNG image.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    ax.scatter(x, y, z, c="b", marker="o")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.savefig(save_path)
    plt.close()


def get_random_subarray(points: np.ndarray, size: int) -> np.ndarray:
    """
    This function returns a subarray of a given 3D points array. The size of the
    subarray is specified by the user. The elements of the subarray are randomly
    selected from the original array. If the size of the original array is smaller than
    the specified size, the function will simply return the original array.

    Args:
        points (numpy array): A numpy array of 3D points. Each element of the array is a
            3D point represented as a numpy array of size 3.
        size (int): The desired size of the subarray.

    Returns:
        numpy array: A subarray of the original points array.
    """
    if len(points) <= size:
        return points
    indices = np.random.choice(len(points), size, replace=False)
    return points[indices]


def too_offset(mask: np.ndarray) -> bool:
    """ 判断对象检测是否过于偏离图像中心
    This will return true if the entire bounding rectangle of the mask is either on the
    left or right third of the mask. This is used to determine if the object is too far
    to the side of the image to be a reliable detection.

    Args:
        mask (numpy array): A 2D numpy array of 0s and 1s representing the mask of the
            object.
    Returns:
        bool: True if the object is too offset, False otherwise.
    """
    # Find the bounding rectangle of the mask
    x, y, w, h = cv2.boundingRect(mask) # 获取边界框

    # Calculate the thirds of the mask
    third = mask.shape[1] // 3 # 计算三分之一宽度 

    #检测是否在左右5%区域
    # Check if the entire bounding rectangle is in the left or right third of the mask
    if x + w <= third:
        # Check if the leftmost point is at the edge of the image
        # return x == 0
        return x <= int(0.05 * mask.shape[1])
    elif x >= 2 * third:
        # Check if the rightmost point is at the edge of the image
        # return x + w == mask.shape[1]
        return x + w >= int(0.95 * mask.shape[1])
    else:
        return False
