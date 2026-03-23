import cv2
import numpy as np

from .general_utils import wrap_heading


def get_two_farthest_points(source, cnt, agent_yaw): 
    """Returns the two points in the contour cnt that form the smallest and largest
    angles from the source point.
    返回轮廓线中与源点形成最小和最大角度的两个点"""
    pts = cnt.reshape(-1, 2)
    pts = pts - source
    rotation_matrix = np.array(
        [
            [np.cos(-agent_yaw), -np.sin(-agent_yaw)],
            [np.sin(-agent_yaw), np.cos(-agent_yaw)],
        ]
    )
    pts = np.matmul(pts, rotation_matrix)
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    # Get the two points that form the smallest and largest angles from the source
    min_idx = np.argmin(angles)
    max_idx = np.argmax(angles)
    return cnt[min_idx], cnt[max_idx]


def vectorize_get_line_points(current_point, points, max_line_len):
    angles = np.arctan2(
        points[..., 1] - current_point[1], points[..., 0] - current_point[0]
    )
    endpoints = np.stack(
        (
            points[..., 0] + max_line_len * np.cos(angles),
            points[..., 1] + max_line_len * np.sin(angles),
        ),
        axis=-1,
    )
    endpoints = endpoints.astype(np.int32)

    line_points = np.stack([points.reshape(-1, 2), endpoints.reshape(-1, 2)], axis=1)
    return line_points


def get_line_points(current_point, points, maxlen):
    current_point = np.repeat(current_point[np.newaxis, :], 2 * len(points), axis=0)
    points = np.repeat(points, 2, axis=0)
    diffs = current_point - points
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    end_points = current_point + maxlen * np.column_stack(
        (np.cos(angles), np.sin(angles))
    )
    line_points = np.concatenate((points, end_points), axis=1)
    line_points = np.array(line_points, dtype=np.int32)
    return line_points


def reveal_fog_of_war(
    top_down_map: np.ndarray, #二维数组，表示俯视地图
    current_fog_of_war_mask: np.ndarray, #当前迷雾掩码
    current_point: np.ndarray, #智能体当前坐标
    current_angle: float, #智能体当前朝向rad
    fov: float = 90, #视野角度（默认90度）
    max_line_len: float = 100, #最大视线距离
    enable_debug_visualization: bool = False, #是否启动调试可视化
) -> np.ndarray:
    curr_pt_cv2 = current_point[::-1].astype(int) #(x,y)坐标转为(y,x)以适应OpenCV的坐标
    
    angle_cv2 = np.rad2deg(wrap_heading(-current_angle + np.pi / 2)) #确保弧度在0到2pi，转角度
    
    #创建扇形视野区域掩码
    cone_mask = cv2.ellipse( #绘制椭圆，椭圆弧
        np.zeros_like(top_down_map), #创建与地图大小相同的空白图像
        curr_pt_cv2,                 #智能体位置
        (int(max_line_len), int(max_line_len)), #椭圆半径
        0,                          #旋转角度
        angle_cv2 - fov / 2,        #起始角度
        angle_cv2 + fov / 2,        #结束角度
        1,                          #颜色值
        -1,                         #填充整个扇形
    )

    #创建掩码：扇形区域与障碍物区域的交集合
    # Create a mask of pixels that are both in the cone and NOT in the top_down_map
    obstacles_in_cone = cv2.bitwise_and(cone_mask, 1 - top_down_map) #按位进行与运算

    # Find the contours of the obstacles in the cone
    # 检测障碍物的轮廓
    obstacle_contours, _ = cv2.findContours( 
        obstacles_in_cone, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    #调试可视化部分
    if enable_debug_visualization:
        #创建可视化地图
        vis_top_down_map = top_down_map * 255
        vis_top_down_map = cv2.cvtColor(vis_top_down_map, cv2.COLOR_GRAY2BGR)
        vis_top_down_map[top_down_map > 0] = (60, 60, 60)   #障碍物显示为灰色
        vis_top_down_map[top_down_map == 0] = (255, 255, 255) #可通行区域为白色
        cv2.circle(vis_top_down_map, tuple(curr_pt_cv2), 3, (255, 192, 15), -1) #标记玩家位置
        cv2.imshow("vis_top_down_map", vis_top_down_map)
        cv2.waitKey(0)
        cv2.destroyWindow("vis_top_down_map")

        # 显示扇形区域减去障碍物后的区域
        cone_minus_obstacles = cv2.bitwise_and(cone_mask, top_down_map)
        vis_cone_minus_obstacles = vis_top_down_map.copy()
        vis_cone_minus_obstacles[cone_minus_obstacles == 1] = (127, 127, 127)
        cv2.imshow("vis_cone_minus_obstacles", vis_cone_minus_obstacles)
        cv2.waitKey(0)
        cv2.destroyWindow("vis_cone_minus_obstacles")

        # 显示障碍物轮廓
        vis_obstacles_mask = vis_cone_minus_obstacles.copy()
        cv2.drawContours(vis_obstacles_mask, obstacle_contours, -1, (0, 0, 255), 1)
        cv2.imshow("vis_obstacles_mask", vis_obstacles_mask)
        cv2.waitKey(0)
        cv2.destroyWindow("vis_obstacles_mask")

    #如果没有障碍物在扇形内，直接返回当前迷雾掩码
    if len(obstacle_contours) == 0:
        return current_fog_of_war_mask  # there were no obstacles in the cone

    # 处理每个障碍物轮廓，找到关键点
    # Find the two points in each contour that form the smallest and largest angles
    # from the current position
    # 凸轮廓是指轮廓内部没有凹陷的部分，从轮廓内任意两点连一条直线，这条直线完全位于轮廓内部。
    # 凹轮廓是指轮廓内部至少有一个凹陷的部分，存在至少一对点，它们之间的直线会穿过轮廓外部。
    points = []
    for cnt in obstacle_contours:
        if cv2.isContourConvex(cnt): # 如果是凸轮廓
            # 获取距离
            pt1, pt2 = get_two_farthest_points(curr_pt_cv2, cnt, angle_cv2) #获得轮廓线中与原点形成最大最小角度的两个点
            points.append(pt1.reshape(-1, 2))
            points.append(pt2.reshape(-1, 2))
        else: #如果是凹轮廓
            # Just add every point in the contour 加入所有点
            points.append(cnt.reshape(-1, 2)) #
    points = np.concatenate(points, axis=0) # 按行拼接

    # 创建可见区域掩码（扇形区域内的可通行区域）
    # Fragment the cone using obstacles and two lines per obstacle in the cone
    # 从智能体位置到各个关键点画线，分割可见区域
    visible_cone_mask = cv2.bitwise_and(cone_mask, top_down_map)
    line_points = vectorize_get_line_points(curr_pt_cv2, points, max_line_len * 1.05)
    # Draw all lines simultaneously using cv2.polylines
    # 绘制所有分割线
    cv2.polylines(visible_cone_mask, line_points, isClosed=False, color=0, thickness=2)

    # Identify the contour that is closest to the current position
    # 找出最终的可见区域轮廓
    final_contours, _ = cv2.findContours(
        visible_cone_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # 找到包含智能体的最小距离轮廓（实际可见区域）
    visible_area = None # 初始化最终选择的可见区域
    min_dist = np.inf  # 初始化最小距离为无限大
    for cnt in final_contours:
        pt = tuple([int(i) for i in curr_pt_cv2]) # 智能体位置转换为整数元组格式，适应cv2要求
        dist = abs(cv2.pointPolygonTest(cnt, pt, True)) # 计算点到轮廓的距离
        # 选择距离最小的轮廓
        if dist < min_dist:
            min_dist = dist
            visible_area = cnt
    # 调试可视化部分
    if enable_debug_visualization:
        vis_points_mask = vis_obstacles_mask.copy()
        for point in points.reshape(-1, 2):
            cv2.circle(vis_points_mask, tuple(point), 3, (0, 255, 0), -1)
        cv2.imshow("vis_points_mask", vis_points_mask)
        cv2.waitKey(0)
        cv2.destroyWindow("vis_points_mask")

        vis_lines_mask = vis_points_mask.copy()
        cv2.polylines(
            vis_lines_mask, line_points, isClosed=False, color=(0, 0, 255), thickness=2
        )
        cv2.imshow("vis_lines_mask", vis_lines_mask)
        cv2.waitKey(0)
        cv2.destroyWindow("vis_lines_mask")

        vis_final_contours = vis_top_down_map.copy()
        # Draw each contour in a random color
        for cnt in final_contours:
            color = tuple([int(i) for i in np.random.randint(0, 255, 3)])
            cv2.drawContours(vis_final_contours, [cnt], -1, color, -1)
        cv2.imshow("vis_final_contours", vis_final_contours)
        cv2.waitKey(0)
        cv2.destroyWindow("vis_final_contours")

        vis_final = vis_top_down_map.copy()
        # Draw each contour in a random color
        cv2.drawContours(vis_final, [visible_area], -1, (127, 127, 127), -1)
        cv2.imshow("vis_final", vis_final)
        cv2.waitKey(0)
        cv2.destroyWindow("vis_final")

    if min_dist > 3: #如果最小距离大于3
        return current_fog_of_war_mask  # 返回现有掩码 the closest contour was too far away

    # 在迷雾掩码上绘制可见区域
    new_fog = cv2.drawContours(current_fog_of_war_mask, [visible_area], 0, 1, -1)

    return new_fog


def visualize(
    top_down: np.ndarray,
    fog_mask: np.ndarray,
    agent_pos: np.ndarray,
    agent_yaw: float,
    agent_size: int,
) -> np.ndarray:
    """
    Visualize the top-down map with the fog of war and the current position/heading of
    the agent superimposed on top. Fog of war is shown in gray, the current position is
    shown in blue, and the current heading is shown as a line segment stemming from the
    center of the agent towards the heading direction.

    Args:
        top_down: The top-down map of the environment.
        fog_mask: The fog of war mask.
        agent_pos: The current position of the agent.
        agent_yaw: The current heading of the agent.
        agent_size: The size (radius) of the agent, in pixels.
    Returns:
        The visualization of the top-down map with the fog of war and the current
        position/heading of the agent superimposed on top.
    """
    img_size = (*top_down.shape[:2], 3)
    viz = np.ones(img_size, dtype=np.uint8) * np.array((60, 60, 60), dtype=np.uint8)
    viz[top_down == 0] = (255, 255, 255)
    viz[fog_mask > 0] = (127, 127, 127)
    cv2.circle(viz, agent_pos[::-1], agent_size, (255, 192, 15), -1)

    heading_end_pt = (
        agent_size * 1.4 * np.array([np.sin(agent_yaw), np.cos(agent_yaw)])
    ) + agent_pos[::-1]

    # Draw a line from the current position showing the current_angle
    cv2.line(
        viz,
        agent_pos[::-1],
        (int(heading_end_pt[0]), int(heading_end_pt[1])),
        (0, 0, 0),
        max(1, agent_size // 4),
    )
    return viz


if __name__ == "__main__":
    import time

    SHOW = True  # whether to imshow the results
    window_size = 1000
    N = 100
    L = (20, 50)
    max_line_len = 500
    fov = 90
    agent_radius = 20
    blank = np.ones((window_size, window_size), dtype=np.uint8)
    times = []
    for _ in range(500):
        t_start = time.time()
        top_down_map = blank.copy()
        # Populate the image with N random rectangles, with a (min, max) length of L
        for _ in range(N):
            rect_0 = np.random.randint(0, window_size, 2)
            rect_1 = rect_0 + np.random.randint(*L, 2)
            cv2.rectangle(top_down_map, rect_0, rect_1, 0, -1)
        # Sample random position and heading
        current_point = np.random.randint(window_size * 0.25, window_size * 0.75, 2)
        # Re-sample current_point if it is inside an obstacle
        while top_down_map[current_point[1], current_point[0]] != 1:
            current_point = np.random.randint(window_size * 0.25, window_size * 0.75, 2)
        current_angle = np.random.uniform(-np.pi, np.pi)

        fog = reveal_fog_of_war(
            top_down_map=top_down_map,
            current_fog_of_war_mask=np.zeros_like(top_down_map),
            current_point=current_point,
            current_angle=current_angle,
            fov=fov,
            max_line_len=max_line_len,
        )

        times.append(time.time() - t_start)

        if SHOW:
            viz = visualize(
                top_down_map, fog, current_point, current_angle, agent_radius
            )
            cv2.imshow("viz", viz)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord("q"):
                break

    print(f"Average time: {np.mean(times[1:])}")
