import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List

def wrap_heading(angle: float) -> float:
    """确保角度在[0, 2π]范围内"""
    return angle % (2 * np.pi)

def get_two_farthest_points(source: np.ndarray, cnt: np.ndarray, agent_yaw: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回轮廓线中与源点形成最小和最大角度的两个点
    
    参数:
        source: 源点坐标
        cnt: 轮廓点集
        agent_yaw: 代理的偏航角（弧度）
    
    返回:
        形成最小和最大角度的两个点
    """
    pts = cnt.reshape(-1, 2)
    pts = pts - source
    
    # 创建旋转矩阵（旋转-agent_yaw角度）
    rotation_matrix = np.array([
        [np.cos(-agent_yaw), -np.sin(-agent_yaw)],
        [np.sin(-agent_yaw), np.cos(-agent_yaw)]
    ])
    
    # 旋转所有点
    pts_rotated = np.matmul(pts, rotation_matrix)
    
    # 计算旋转后的角度
    angles = np.arctan2(pts_rotated[:, 1], pts_rotated[:, 0])
    
    # 找到形成最小和最大角度的点
    min_idx = np.argmin(angles)
    max_idx = np.argmax(angles)
    
    return cnt.reshape(-1, 2)[min_idx], cnt.reshape(-1, 2)[max_idx]

def vectorize_get_line_points(center: np.ndarray, points: np.ndarray, max_len: float) -> List[np.ndarray]:
    """
    从中心点到各个点生成线段
    """
    line_points = []
    center = center.astype(float)
    
    for point in points:
        point = point.astype(float)
        # 计算方向向量
        direction = point - center
        # 归一化
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
            # 计算终点
            end_point = center + direction * max_len
            line_points.append(np.array([center.astype(int), end_point.astype(int)], dtype=np.int32))
    return line_points

def create_visualization_base(top_down_map: np.ndarray, player_pos: np.ndarray) -> np.ndarray:
    """创建可视化基础图像"""
    vis = np.zeros((top_down_map.shape[0], top_down_map.shape[1], 3), dtype=np.uint8)
    vis[top_down_map > 0] = (100, 100, 100)  # 障碍物灰色
    vis[top_down_map == 0] = (200, 200, 200)  # 可通行区域浅灰色
    cv2.circle(vis, tuple(player_pos.astype(int)), 5, (0, 0, 255), -1)  # 玩家红色
    return vis

def show_image(image: np.ndarray, title: str, info: str = ""):
    """显示图像并等待按键"""
    display = image.copy()
    if len(display.shape) == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    
    # 调整图像大小以便显示
    height, width = display.shape[:2]
    if max(height, width) > 800:
        scale = 800 / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        display = cv2.resize(display, (new_width, new_height))
    
    cv2.putText(display, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if info:
        cv2.putText(display, info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow(title, display)
    print(f"显示: {title} - {info}")
    print("按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyWindow(title)

def reveal_fog_of_war_with_visualization(
    top_down_map: np.ndarray,
    current_fog_of_war_mask: np.ndarray,
    current_point: np.ndarray,
    current_angle: float,
    fov: float = 90,
    max_line_len: float = 100,
) -> np.ndarray:
    """
    带可视化步骤的Fog of War揭示函数
    """
    print("=== Fog of War 算法步骤可视化 ===")
    
    # 步骤1: 坐标转换
    curr_pt_cv2 = current_point.astype(int)
    angle_cv2 = np.rad2deg(wrap_heading(-current_angle + np.pi / 2))
    print(f"玩家位置: {curr_pt_cv2}, 视角角度: {angle_cv2:.1f}°")
    
    # 创建可视化基础图像
    vis_base = create_visualization_base(top_down_map, curr_pt_cv2)
    show_image(vis_base, "0. 初始地图", "灰色:障碍物, 浅灰:可通行, 红色:玩家")
    
    # 步骤2: 创建扇形视野
    cone_mask = np.zeros_like(top_down_map, dtype=np.uint8)
    cv2.ellipse(
        cone_mask,
        tuple(curr_pt_cv2),
        (int(max_line_len), int(max_line_len)),
        0,
        angle_cv2 - fov / 2,
        angle_cv2 + fov / 2,
        255,
        -1,
    )
    
    # 修复：创建扇形可视化时，先复制基础图像，然后添加扇形覆盖
    vis_cone = vis_base.copy()
    # 创建一个半透明的扇形覆盖
    cone_overlay = np.zeros_like(vis_base, dtype=np.uint8)
    cone_overlay[cone_mask > 0] = (100, 200, 255)  # 扇形区域蓝色
    # 使用加权叠加实现半透明效果
    vis_cone = cv2.addWeighted(vis_cone, 0.7, cone_overlay, 0.3, 0)
    show_image(vis_cone, "1. 创建扇形视野", "蓝色:理论视野范围")
    
    # 步骤3: 检测扇形内障碍物
    obstacles = (1 - top_down_map).astype(np.uint8)  # 修复：确保数据类型为uint8
    obstacles_in_cone = cv2.bitwise_and(cone_mask, obstacles * 255)  # 修复：障碍物掩码值
    
    vis_obstacles = vis_base.copy()
    # 先显示扇形区域
    cone_overlay = np.zeros_like(vis_base, dtype=np.uint8)
    cone_overlay[cone_mask > 0] = (100, 200, 255)
    vis_obstacles = cv2.addWeighted(vis_obstacles, 0.7, cone_overlay, 0.3, 0)
    # 再显示障碍物
    vis_obstacles[obstacles_in_cone > 0] = (255, 0, 0)  # 障碍物红色
    show_image(vis_obstacles, "2. 扇形内障碍物", "红色:扇形内障碍物")
    
    # 步骤4: 查找障碍物轮廓
    obstacle_contours, _ = cv2.findContours(
        obstacles_in_cone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    vis_contours = vis_base.copy()
    cv2.drawContours(vis_contours, obstacle_contours, -1, (0, 255, 0), 2)  # 绿色轮廓
    show_image(vis_contours, "3. 障碍物轮廓检测", f"找到 {len(obstacle_contours)} 个障碍物轮廓")
    
    if len(obstacle_contours) == 0:
        print("没有障碍物在视野内，直接返回扇形视野")
        # 如果没有障碍物，整个扇形都是可见的
        new_fog = current_fog_of_war_mask.copy()
        # 只在可通行区域内的扇形部分更新迷雾
        visible_cone_mask = cv2.bitwise_and(cone_mask, top_down_map.astype(np.uint8) * 255)
        new_fog[visible_cone_mask > 0] = 1
        return new_fog
    
    # 步骤5: 提取关键点
    points = []
    for cnt in obstacle_contours:
        if len(cnt) >= 3:
            # 使用正确的get_two_farthest_points函数
            pt1, pt2 = get_two_farthest_points(curr_pt_cv2, cnt, current_angle)
            points.append(pt1.reshape(-1, 2))
            points.append(pt2.reshape(-1, 2))
        else:
            # 对于点数不足的，使用所有点
            points.append(cnt.reshape(-1, 2))
    
    if len(points) == 0:
        print("没有提取到关键点，返回原迷雾")
        return current_fog_of_war_mask
        
    points = np.concatenate(points, axis=0)
    
    vis_points = vis_base.copy()
    cv2.drawContours(vis_points, obstacle_contours, -1, (0, 255, 0), 1)  # 绿色轮廓
    for point in points:
        cv2.circle(vis_points, tuple(point.astype(int)), 3, (255, 255, 0), -1)  # 黄色关键点
    show_image(vis_points, "4. 关键点提取", f"提取了 {len(points)} 个关键点(黄色)")
    
    # 步骤6: 创建可见区域掩码并画分割线
    visible_cone_mask = cv2.bitwise_and(cone_mask, top_down_map.astype(np.uint8) * 255)
    line_points = vectorize_get_line_points(curr_pt_cv2, points, max_line_len * 1.2)
    
    vis_lines = vis_base.copy()
    cv2.drawContours(vis_lines, obstacle_contours, -1, (0, 255, 0), 1)
    for line in line_points:
        cv2.line(vis_lines, tuple(line[0]), tuple(line[1]), (255, 0, 255), 2)  # 粉色分割线
    show_image(vis_lines, "5. 分割线绘制", f"绘制了 {len(line_points)} 条分割线(粉色)")
    
    # 在掩码上画线（分割区域）
    for line in line_points:
        cv2.line(visible_cone_mask, tuple(line[0]), tuple(line[1]), 0, 3)  # 稍微加粗线条
    
    # 步骤7: 查找最终轮廓
    final_contours, _ = cv2.findContours(
        visible_cone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    vis_final_contours = vis_base.copy()
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100), (255, 100, 255)]
    for i, cnt in enumerate(final_contours):
        color = colors[i % len(colors)]
        cv2.drawContours(vis_final_contours, [cnt], -1, color, 2)
        # 计算中心点显示编号
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(vis_final_contours, f"{i}", (cx, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    show_image(vis_final_contours, "6. 分割后轮廓", f"分割为 {len(final_contours)} 个区域")
    
    # 步骤8: 选择可见区域
    visible_area = None
    min_dist = np.inf
    dist_info = []
    
    player_point_tuple = (int(curr_pt_cv2[0]), int(curr_pt_cv2[1]))
    
    for i, cnt in enumerate(final_contours):
        dist = cv2.pointPolygonTest(cnt, player_point_tuple, True)
        abs_dist = abs(dist)
        dist_info.append((i, dist, abs_dist))
        
        if abs_dist < min_dist:
            min_dist = abs_dist
            visible_area = cnt
    
    print("各轮廓到玩家的距离:")
    for i, dist, abs_dist in dist_info:
        status = "内部" if dist >= 0 else "外部"
        print(f"  轮廓{i}: {dist:.2f} ({status}, 绝对值: {abs_dist:.2f})")
    print(f"选择距离最小的轮廓: {min_dist:.2f}")
    
    if visible_area is None or min_dist > 5:
        print("没有合适的轮廓，使用整个扇形区域")
        new_fog = current_fog_of_war_mask.copy()
        new_fog[visible_cone_mask > 0] = 1
        return new_fog
    
    # 步骤9: 更新迷雾
    new_fog = current_fog_of_war_mask.copy()
    temp_mask = np.zeros_like(new_fog, dtype=np.uint8)
    cv2.drawContours(temp_mask, [visible_area], 0, 1, -1)
    new_fog[temp_mask > 0] = 1
    
    vis_result = vis_base.copy()
    result_overlay = np.zeros_like(vis_base, dtype=np.uint8)
    cv2.drawContours(result_overlay, [visible_area], -1, (0, 255, 0), -1)
    vis_result = cv2.addWeighted(vis_result, 0.7, result_overlay, 0.3, 0)
    show_image(vis_result, "7. 最终可见区域", "绿色区域为本次揭示的视野")
    
    return new_fog

def test_fog_of_war():
    """测试Fog of War算法"""
    # 创建测试地图 (1=可通行, 0=障碍物)
    map_size = 400
    top_down_map = np.ones((map_size, map_size), dtype=np.uint8)
    
    # 添加一些障碍物（设置为0）
    cv2.rectangle(top_down_map, (100, 100), (200, 250), 0, -1)  # 矩形障碍物
    cv2.circle(top_down_map, (300, 150), 50, 0, -1)  # 圆形障碍物
    cv2.rectangle(top_down_map, (50, 300), (150, 350), 0, -1)  # 另一个矩形
    
    # 初始化迷雾（全黑）
    fog_mask = np.zeros_like(top_down_map)
    
    # 测试不同位置和角度
    test_cases = [
        (np.array([80, 80]), np.pi/4),    # 靠近障碍物
        (np.array([250, 100]), np.pi/2),  # 看向圆形障碍物
        (np.array([350, 350]), -np.pi/4), # 边缘位置
    ]
    
    for i, (position, angle) in enumerate(test_cases):
        print(f"\n{'='*50}")
        print(f"=== 测试用例 {i+1}: 位置{position}, 角度{np.rad2deg(angle):.1f}° ===")
        print(f"{'='*50}")
        
        fog_mask = reveal_fog_of_war_with_visualization(
            top_down_map=top_down_map,
            current_fog_of_war_mask=fog_mask,
            current_point=position,
            current_angle=angle,
            fov=90,
            max_line_len=150,
        )
        
        # 显示累计迷雾
        vis_fog = create_visualization_base(top_down_map, position.astype(int))
        # 创建迷雾覆盖层
        fog_overlay = np.zeros_like(vis_fog, dtype=np.uint8)
        fog_overlay[fog_mask > 0] = (0, 255, 0)  # 已揭示区域绿色
        vis_fog = cv2.addWeighted(vis_fog, 0.7, fog_overlay, 0.3, 0)
        show_image(vis_fog, f"累计迷雾状态 - 测试{i+1}", "绿色为已揭示区域")

if __name__ == "__main__":
    # 设置OpenCV窗口属性
    cv2.namedWindow("temp", cv2.WINDOW_NORMAL)
    cv2.destroyWindow("temp")
    
    try:
        test_fog_of_war()
        print("测试完成！")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()  