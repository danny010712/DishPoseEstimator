import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from config import CAPTURE_WIDTH, CAPTURE_HEIGHT, DEPTH_SCALE_FACTOR, DEPTH_TRUNCATION

def capture_pointcloud(height = 480, width = 640, depth_limit = 3.0):
    """
    Get point cloud data from the scene using RealSense D435.
    Args:
        height: height of the image
        width: width of the image
        depth_limit: 
    Returns:
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
            exit(1)

        # -----------------------------------------------------
        # 3. 후처리 필터 적용
        # -----------------------------------------------------
        print("Applying post-processing filters...")
        depth_frame_filtered = spatial.process(depth_frame)
        depth_frame_filled = hole_filling.process(depth_frame_filtered)

        # -----------------------------------------------------
        # 4. Open3D 변환 및 시각화
        # -----------------------------------------------------
        print("Converting to Open3D format...")
        # NumPy 배열로 변환 (필터링된 깊이 프레임 사용)
        depth_image = np.asanyarray(depth_frame_filled.get_data())
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
            intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
            
        # RGBD 이미지로부터 포인트 클라우드 생성
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            pinhole_camera_intrinsic)
        
        print("Visualizing point cloud. Press 'L' to toggle lighting, 'Q' to close.")
        # o3d.visualization.draw_geometries([pcd])
        
    finally:
        pipeline.stop()

    # # save the raw point cloud(to check)
    # o3d.io.write_point_cloud(raw_pcd_file_path, pcd)

    return pcd


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
    rgb_array = color_array[..., [2, 1, 0]] ## To change
    
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
