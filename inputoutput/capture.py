import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
from config import CAPTURE_WIDTH, CAPTURE_HEIGHT, DEPTH_SCALE_FACTOR, DEPTH_TRUNCATION, RESULTS_DIR, SAVE_INTERMEDIATE
from .file_io import save_pointcloud, load_pointcloud
from datetime import datetime
import os
# from utils.visualization import show_pointcloud

def capture_pointcloud(height = 480, width = 640, depth_limit = 3.0):
    """
    Get point cloud data from the scene using RealSense D435.
    Args:
        height: height of the image
        width: width of the image
        depth_limit: depth limit of capturing. Ignores points further than limit
    Returns:
        pcd: point cloud made from rgb-d image.
        color_image: HxW color ndarray.
        depth_image: HxW depth ndarray.
    """
    try:
        # -----------------------------------------------------
        # 1. RealSense 파이프라인 설정 및 시작
        # -----------------------------------------------------
        pipeline = rs.pipeline()
        config = rs.config()

        # 스트림 설정 (해상도, 포맷, 프레임)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)

        # 스트리밍 시작
        profile = pipeline.start(config)

        # 정렬 객체 생성
        align = rs.align(rs.stream.color)

        # -----------------------------------------------------
        # 2. 후처리 필터 객체 생성
        # -----------------------------------------------------
        # 공간 필터 (노이즈 감소 및 구멍 일부 메우기)
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 2)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)

        # 구멍 메우기 필터
        hole_filling = rs.hole_filling_filter()

        # try:
        # 안정적인 프레임을 위해 잠시 대기
        print("Capturing frames...")
        for _ in range(10):
            pipeline.wait_for_frames()
        
        # 프레임 획득 및 정렬
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            raise RuntimeError("Could not acquire depth or color frames.")

        # -----------------------------------------------------
        # 3. 후처리 필터 적용
        # -----------------------------------------------------
        print("Applying post-processing filters...")
        # depth_frame_filtered = spatial.process(depth_frame)
        # depth_frame_filled = hole_filling.process(depth_frame_filtered)

        depth_frame_filled = depth_frame # without filters

        # -----------------------------------------------------
        # 4. Open3D 변환 및 시각화
        # -----------------------------------------------------
        print("Converting to Open3D format...")
        # NumPy 배열로 변환 (필터링된 깊이 프레임 사용)
        depth_image = np.asanyarray(depth_frame_filled.get_data())
        print(depth_image.shape)
        color_image = np.asanyarray(color_frame.get_data())

        # Open3D가 사용하는 RGBDImage 객체로 변환
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image),
            o3d.geometry.Image(depth_image),
            depth_scale=1.0 / profile.get_device().first_depth_sensor().get_depth_scale(),
            depth_trunc=depth_limit, # 3미터 이상 데이터는 무시
            convert_rgb_to_intensity=False)
            
        # 카메라 내부 파라미터(Intrinsics) 가져오기
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
        )
        print(pinhole_camera_intrinsic)
            
        # RGBD 이미지로부터 포인트 클라우드 생성
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            pinhole_camera_intrinsic)
        
        # print("Visualizing point cloud. Press 'L' to toggle lighting, 'Q' to close.")
        
    finally:
        pipeline.stop()

    if SAVE_INTERMEDIATE:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        o3d.io.write_point_cloud(os.path.join(RESULTS_DIR, f'raw_{timestamp}.ply'), pcd)
        cv2.imwrite(os.path.join(RESULTS_DIR, f'color_{timestamp}.png'), cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
        depth_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(RESULTS_DIR, f"depth_map_{timestamp}.png"), depth_colored)

    return pcd, color_image, depth_image, pinhole_camera_intrinsic


def load_and_create_intrinsics(K_matrix: np.ndarray, width: int, height: int) -> o3d.camera.PinholeCameraIntrinsic:
    """
    Creates Open3D camera intrinsics from a 3x3 K matrix and image dimensions.
    The K matrix is structured as: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    """
    if K_matrix.shape != (3, 3):
        raise ValueError(f"K matrix must be a 3x3 array, but got shape {K_matrix.shape}.")
    
    fx = K_matrix[0, 0]
    fy = K_matrix[1, 1]
    cx = K_matrix[0, 2]
    cy = K_matrix[1, 2]

    # Create Open3D Intrinsics object
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )
    return intrinsics


def create_pcd_from_rgbd(color_array: np.ndarray, depth_array: np.ndarray, intrinsics: o3d.camera.PinholeCameraIntrinsic) -> o3d.geometry.PointCloud:
    """
    Creates an Open3D PointCloud object from color and depth NumPy arrays 
    using the provided camera intrinsics.
    """
    # rgb_array = color_array[..., [2, 1, 0]] ## To change
    rgb_array = color_array
    
    # 1. Convert NumPy arrays to Open3D Image objects
    o3d_color = o3d.geometry.Image(rgb_array.astype(np.uint8).copy())
    # Ensure the depth array has the correct dtype (uint16) for raw depth data
    o3d_depth = o3d.geometry.Image(depth_array.astype(np.uint16).copy()) 

    # 2. Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color,
        o3d_depth,
        depth_scale=DEPTH_SCALE_FACTOR,
        depth_trunc=DEPTH_TRUNCATION,
        convert_rgb_to_intensity=False
    )

    # 3. Create Point Cloud using RGBD image and intrinsics
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsics
    )

    print(f"  -> Successfully created Point Cloud with {len(pcd.points)} points.")
    return pcd

def create_pcd_from_rgbd_with_mask(color_array, depth_array, intrinsics, mask):
    """
    Create masked point cloud directly from depth + color arrays (no filtering loss).
    """
    fx = intrinsics.intrinsic_matrix[0, 0]
    fy = intrinsics.intrinsic_matrix[1, 1]
    cx = intrinsics.intrinsic_matrix[0, 2]
    cy = intrinsics.intrinsic_matrix[1, 2]

    # (H, W)
    h, w = depth_array.shape

    # Valid pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Flatten
    u = u.flatten()
    v = v.flatten()
    z = depth_array.flatten().astype(np.float32) / DEPTH_SCALE_FACTOR
    mask_flat = mask.flatten()

    # Combine depth validity + mask
    valid = (z > 0) & mask_flat

    # Filter
    u = u[valid]
    v = v[valid]
    z = z[valid]
    colors = color_array[v, u, :] / 255.0  # BGR 또는 RGB 순서 확인 필요

    # Back-project to 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack points
    points = np.stack((x, y, z), axis=-1)

    # Create point cloud
    pcd_masked = o3d.geometry.PointCloud()
    pcd_masked.points = o3d.utility.Vector3dVector(points)
    pcd_masked.colors = o3d.utility.Vector3dVector(colors)

    print(f"  -> Masked Point Cloud created: {len(points)} valid points.")
    return pcd_masked
 


def main():
    pcd, _, _, _ = capture_pointcloud()
    # save_pointcloud(pcd, 'point_cloud_capture_test')
    # pcd = load_pointcloud(r'C:\Users\danny\OneDrive\Desktop\UROP2\DishPoseEstimator\results\point_cloud_capture_test.ply')
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()

