import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

def contour_to_frontiers(contour, unexplored_mask):
    """
    从cv2接受一个已探索轮廓,返回一个numpy数组组成的列表,每个数组包含形成一个独立边界的连续点
    轮廓设想为连续点的集合，但是有一些点并不在边界上,表现为在未探索掩码上数值为0的点
    """
    print(f"输入轮廓包含 {len(contour)} 个点")
    
    # 步骤1：找出"坏点"（不在边界上的点）
    bad_inds = []
    num_contour_points = len(contour)
    
    print("\n步骤1：识别坏点（unexplored_mask值为0的点）")
    for idx in range(num_contour_points):
        x, y = contour[idx][0]
        if unexplored_mask[y, x] == 0:
            bad_inds.append(idx)
            print(f"  坏点索引 {idx}: 坐标({x}, {y})")
    
    print(f"总共找到 {len(bad_inds)} 个坏点: {bad_inds}")
    
    # 步骤2：在坏点位置分割轮廓
    print(f"\n步骤2：使用np.split在坏点位置分割轮廓")
    frontiers = np.split(contour, bad_inds)
    print(f"分割后得到 {len(frontiers)} 个片段，长度分别为: {[len(f) for f in frontiers]}")
    
    # 步骤3：检测是否为环形轮廓
    front_last_split = (
        0 not in bad_inds and 
        len(bad_inds) > 0 and 
        max(bad_inds) < num_contour_points - 2
    )
    print(f"\n步骤3：环形轮廓检测")
    print(f"  0不在坏点中: {0 not in bad_inds}")
    print(f"  存在坏点: {len(bad_inds) > 0}")
    print(f"  最大坏点索引 < 轮廓长度-2: {max(bad_inds) < num_contour_points - 2 if bad_inds else False}")
    print(f"  是否为环形轮廓: {front_last_split}")
    
    # 步骤4：过滤并清理边界片段
    print(f"\n步骤4：过滤边界片段（移除分割点并合并环形边界）")
    filtered_frontiers = []
    
    for idx, f in enumerate(frontiers):
        print(f"  处理片段 {idx}，长度: {len(f)}")
        
        # 边界必须至少有2个点（包含坏点时为3个）
        if len(f) > 2 or (idx == 0 and front_last_split):
            if idx == 0:
                filtered_frontiers.append(f)
                print(f"    -> 第一个片段直接添加，长度: {len(f)}")
            else:
                filtered_frontiers.append(f[1:])  # 移除第一个点（是坏点）
                print(f"    -> 移除首个坏点后添加，长度: {len(f[1:])}")
        else:
            print(f"    -> 片段太短，跳过")
    
    # 步骤5：合并环形轮廓的首尾片段
    if len(filtered_frontiers) > 1 and front_last_split:
        print(f"\n步骤5：合并环形轮廓的首尾片段")
        last_frontier = filtered_frontiers.pop()
        print(f"  弹出最后片段，长度: {len(last_frontier)}")
        filtered_frontiers[0] = np.concatenate((last_frontier, filtered_frontiers[0]))
        print(f"  合并后第一个片段长度: {len(filtered_frontiers[0])}")
    else:
        print(f"\n步骤5：无需合并（非环形或片段不足）")
    
    print(f"\n最终结果：{len(filtered_frontiers)} 个独立边界")
    for i, frontier in enumerate(filtered_frontiers):
        print(f"  边界 {i}: {len(frontier)} 个点")
    
    return filtered_frontiers

def visualize_process(contour, unexplored_mask, frontiers):
    """可视化整个处理过程"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 图1：原始轮廓和掩码
    ax1 = axes[0]
    ax1.imshow(unexplored_mask, cmap='gray', alpha=0.7)
    contour_points = contour.reshape(-1, 2)
    ax1.plot(contour_points[:, 0], contour_points[:, 1], 'b-o', markersize=4, linewidth=2, label='原始轮廓')
    
    # 标记坏点
    bad_points = []
    for idx, point in enumerate(contour_points):
        x, y = int(point[0]), int(point[1])
        if unexplored_mask[y, x] == 0:
            bad_points.append(point)
            ax1.plot(x, y, 'ro', markersize=8, label='坏点' if len(bad_points) == 1 else "")
    
    ax1.set_title('原始轮廓和未探索掩码')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2：分割后的片段
    ax2 = axes[1]
    ax2.imshow(unexplored_mask, cmap='gray', alpha=0.7)
    
    # 显示每个边界片段
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    for i, frontier in enumerate(frontiers):
        if len(frontier) > 0:
            frontier_points = frontier.reshape(-1, 2)
            color = colors[i % len(colors)]
            ax2.plot(frontier_points[:, 0], frontier_points[:, 1], 
                    color=color, marker='o', markersize=4, linewidth=2,
                    label=f'边界 {i} ({len(frontier)}点)')
    
    ax2.set_title('最终边界片段')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3：统计信息
    ax3 = axes[2]
    ax3.axis('off')
    
    info_text = f"""
处理统计：
• 原始轮廓点数: {len(contour)}
• 坏点数量: {len(bad_points)}
• 最终边界数: {len(frontiers)}
• 各边界点数: {[len(f) for f in frontiers]}

边界定义：
• 好点：unexplored_mask[y,x] != 0
• 坏点：unexplored_mask[y,x] == 0
• 边界：由连续好点组成的片段
"""
    
    ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes, 
            fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()

# 创建测试数据
def create_test_data():
    """创建测试用的轮廓和掩码数据"""
    
    # 创建一个简单的L形轮廓
    contour_points = [
        [2, 2], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6],  # 右边
        [5, 6], [4, 6], [3, 6], [2, 6],  # 底边
        [2, 5], [2, 4], [2, 3]   # 左边
    ]
    
    # 转换为OpenCV格式的轮廓
    contour = np.array(contour_points).reshape(-1, 1, 2)
    
    # 创建未探索掩码（模拟机器人探索环境）
    mask = np.ones((10, 10), dtype=np.uint8)
    
    # 设置一些"坏点"（已探索区域边界）
    mask[2, 6] = 0  # 在拐角处设置坏点
    mask[6, 3] = 0  # 在直线段设置坏点
    
    return contour, mask

# 主演示程序
def main():
    print("=" * 60)
    print("边界分割函数演示程序")
    print("=" * 60)
    
    # 创建测试数据
    contour, unexplored_mask = create_test_data()
    
    print("测试场景：L形轮廓，包含2个坏点")
    print("坏点表示该位置旁边是已探索区域，不构成真正的探索边界")
    print("-" * 60)
    
    # 运行边界分割函数
    frontiers = contour_to_frontiers(contour, unexplored_mask)
    
    print("=" * 60)
    print("处理完成！")
    
    # 可视化结果
    visualize_process(contour, unexplored_mask, frontiers)
    
    return frontiers

if __name__ == "__main__":
    main()