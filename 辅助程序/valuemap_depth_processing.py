#!/usr/bin/env python3
"""
可视化深度图像处理过程的详细步骤
"""

import numpy as np
import cv2
from vlfm.mapping.value_map import ValueMap

def add_text_to_image(image, text, position=(10, 30)):
    """
    在图像上添加文字说明
    """
    # 复制图像以避免修改原始图像
    img_with_text = image.copy()
    
    # 添加文字
    cv2.putText(img_with_text, text, position, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img_with_text, text, position, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    return img_with_text

def create_sample_depth_with_explanation():
    """
    创建示例深度图像并解释每个部分
    """
    # 创建一个128x128的深度图像
    depth = np.ones((128, 128), dtype=np.float32) * 0.7  # 默认深度值0.7 (中等深度)
    
    # 添加不同深度的物体
    # 1. 近处物体 - 深度值0.2
    cv2.rectangle(depth, (20, 20), (50, 50), 0.2, -1)  # 左上角的近处物体
    
    # 2. 远处物体 - 深度值0.9
    cv2.circle(depth, (100, 100), 20, 0.9, -1)  # 右下角的远处物体
    
    # 3. 中等深度物体 - 深度值0.5
    cv2.rectangle(depth, (70, 30), (90, 60), 0.5, -1)  # 中间的中等深度物体
    
    print("Creating sample depth image:")
    print("  - Image size: 128x128 pixels")
    print("  - Depth range: 0.0 (closest) to 1.0 (farthest)")
    print("  - Background depth: 0.7 (medium depth)")
    print("  - Top-left rectangle: depth 0.2 (very close)")
    print("  - Center rectangle: depth 0.5 (medium)")
    print("  - Bottom-right circle: depth 0.9 (very far)")
    
    return depth

def visualize_depth_processing_steps(depth):
    """
    可视化深度图像处理的各个步骤
    """
    print("\nVisualizing depth processing steps:")
    
    # 步骤1: 原始深度图像
    depth_vis = (depth * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    depth_with_text = add_text_to_image(depth_colormap, "Step 1: Original depth image")
    cv2.imwrite("step1_original_depth.png", depth_with_text)
    print("  1. Original depth image saved as step1_original_depth.png")
    
    # 步骤2: 每列最大深度值
    depth_row = np.max(depth, axis=0)  # 每列取最大深度值
    # 创建一个更大的图像来显示详细信息
    large_img = np.zeros((150, 512, 3), dtype=np.uint8)
    # 将深度行扩展到图像上部
    depth_row_expanded = np.repeat(depth_row.reshape(1, -1), 50, axis=0)
    depth_row_vis = (depth_row_expanded * 255).astype(np.uint8)
    depth_row_colormap = cv2.applyColorMap(depth_row_vis, cv2.COLORMAP_JET)
    large_img[10:60, 10:510] = cv2.resize(depth_row_colormap, (500, 50))
    
    # 添加说明文字
    cv2.putText(large_img, "Step 2: Max depth per column", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(large_img, "Each column shows max depth value from original image", (10, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imwrite("step2_max_depth_per_column.png", large_img)
    print("  2. Max depth per column saved as step2_max_depth_per_column.png")
    
    # 步骤3: 角度计算和3D坐标计算
    fov = np.deg2rad(79)
    angles = np.linspace(-fov / 2, fov / 2, len(depth_row))
    print(f"  3. Field of view: {np.rad2deg(fov):.1f} degrees")
    print(f"     Angle range: {np.rad2deg(angles[0]):.1f}° to {np.rad2deg(angles[-1]):.1f}°")
    
    # 步骤4: 3D坐标计算可视化
    x_coords = depth_row  # x坐标代表深度（前后距离）
    y_coords = depth_row * np.tan(angles)  # y坐标代表左右偏移
    
    # 创建坐标可视化 - 增大图像尺寸以容纳更多信息
    coord_vis = np.zeros((300, 600, 3), dtype=np.uint8)
    # 绘制坐标轴
    cv2.line(coord_vis, (50, 250), (550, 250), (128, 128, 128), 1)  # X轴
    cv2.line(coord_vis, (300, 250), (300, 50), (128, 128, 128), 1)   # Y轴
    
    # 添加轴标签
    cv2.putText(coord_vis, "X (depth direction)", (450, 270), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(coord_vis, "Y (lateral offset)", (310, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # 绘制点
    points_drawn = 0
    for i in range(0, len(x_coords), 2):  # 每2个点绘制一个
        x = x_coords[i]  # 深度值
        y = y_coords[i]  # 横向偏移
        
        # 转换为图像坐标 (注意坐标系转换)
        img_x = int(300 + y * 300)  # 横向偏移映射到X轴
        img_y = int(250 - x * 200)  # 深度值映射到Y轴 (注意翻转)
        
        # 确保点在图像范围内
        if 0 <= img_x < 600 and 50 <= img_y < 250:
            cv2.circle(coord_vis, (img_x, img_y), 3, (0, 255, 0), -1)
            points_drawn += 1
            
            # 每隔一定数量的点添加一个小标签
            if i % 20 == 0:
                label = f"{x:.2f}"
                cv2.putText(coord_vis, label, (img_x-15, img_y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    
    # 添加说明文字
    cv2.putText(coord_vis, "Step 3: 3D coordinate calculation", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(coord_vis, f"FOV: {np.rad2deg(fov):.1f} degrees, Points plotted: {points_drawn}", (10, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(coord_vis, "X: depth (front-back), Y: lateral offset (left-right)", (10, 320), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    cv2.imwrite("step3_3d_coordinates.png", coord_vis)
    print("  4. 3D coordinate calculation saved as step3_3d_coordinates.png")
    
    # 步骤5: 像素坐标转换
    value_map = ValueMap(value_channels=1)
    pixel_x = (x_coords * value_map.pixels_per_meter + 150).astype(int)
    pixel_y = (y_coords * value_map.pixels_per_meter + 150).astype(int)
    
    # 创建像素坐标可视化
    pixel_vis = np.zeros((300, 600, 3), dtype=np.uint8)
    # 绘制坐标轴
    cv2.line(pixel_vis, (0, 150), (600, 150), (128, 128, 128), 1)  # X轴
    cv2.line(pixel_vis, (150, 0), (150, 300), (128, 128, 128), 1)   # Y轴
    
    # 绘制点
    points_drawn = 0
    for i in range(0, len(pixel_x), 2):
        px = pixel_x[i]
        py = pixel_y[i]
        # 调整坐标到图像范围内
        img_x = px
        img_y = 300 - py  # 翻转Y轴
        if 0 <= img_x < 600 and 0 <= img_y < 300:
            cv2.circle(pixel_vis, (img_x, img_y), 2, (0, 255, 255), -1)
            points_drawn += 1
    
    # 添加说明文字
    cv2.putText(pixel_vis, "Step 4: Pixel coordinate conversion", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(pixel_vis, f"Pixels per meter: {value_map.pixels_per_meter}", (10, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(pixel_vis, f"Points plotted: {points_drawn}", (10, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    cv2.imwrite("step4_pixel_coordinates.png", pixel_vis)
    print("  5. Pixel coordinate conversion saved as step4_pixel_coordinates.png")

def demonstrate_confidence_mask():
    """
    演示置信度掩码的生成过程
    """
    print("\nConfidence mask generation:")
    
    value_map = ValueMap(value_channels=1)
    fov = np.deg2rad(79)
    max_depth = 5.0
    
    # 生成空白锥形掩码
    blank_cone = value_map._get_blank_cone_mask(fov, max_depth)
    blank_vis = (blank_cone * 255).astype(np.uint8)
    blank_with_text = add_text_to_image(blank_vis, "Step 5: Blank cone mask")
    cv2.imwrite("step5_blank_cone_mask.png", blank_with_text)
    print("  1. Blank cone mask saved as step5_blank_cone_mask.png")
    
    # 生成置信度掩码
    confidence_mask = value_map._get_confidence_mask(fov, max_depth)
    confidence_vis = (confidence_mask * 255).astype(np.uint8)
    confidence_colormap = cv2.applyColorMap(confidence_vis, cv2.COLORMAP_HOT)
    confidence_with_text = add_text_to_image(confidence_colormap, "Step 6: Confidence mask (center high, edges low)")
    cv2.imwrite("step6_confidence_mask.png", confidence_with_text)
    print("  2. Confidence mask saved as step6_confidence_mask.png")
    print("     Confidence decreases from center (high) to edges (low)")

def demonstrate_contour_generation(depth):
    """
    演示轮廓生成过程
    """
    print("\nContour generation:")
    
    value_map = ValueMap(value_channels=1)
    fov = np.deg2rad(79)
    min_depth = 0.5
    max_depth = 5.0
    
    # 处理局部数据
    processed_data = value_map._process_local_data(depth, fov, min_depth, max_depth)
    processed_vis = (processed_data * 255).astype(np.uint8)
    processed_with_text = add_text_to_image(processed_vis, "Step 7: Visible area mask")
    cv2.imwrite("step7_visible_area_mask.png", processed_with_text)
    print("  1. Visible area mask saved as step7_visible_area_mask.png")

def main():
    print("Detailed visualization of depth image processing")
    print("=" * 50)
    
    # 创建示例深度图像
    depth = create_sample_depth_with_explanation()
    
    # 可视化处理步骤
    visualize_depth_processing_steps(depth)
    
    # 演示置信度掩码
    demonstrate_confidence_mask()
    
    # 演示轮廓生成
    demonstrate_contour_generation(depth)
    
    print("\nAll processing steps visualization completed!")
    print("Please check the generated image files to understand each step.")

if __name__ == "__main__":
    main()