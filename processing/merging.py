import open3d as o3d
import numpy as np
# from config import VOXEL_SIZE
# from io.file_io import save_pointcloud

def register_point_clouds(camera_world_pose, pcd_list_cam_frame, ee_world_pose_list): # Used for merging point clouds from different views
    """
    Registers multiple point clouds from a fixed camera into a single
    point cloud in the end-effector's coordinate frame.

    Args:
        camera_world_pose (np.ndarray): The 4x4 SE(3) transformation matrix
                                        from the camera frame to the world frame.
        pcd_list_cam_frame (list[o3d.geometry.PointCloud]): A list of point clouds,
                                                            each in the camera frame.
        ee_world_pose_list (list[np.ndarray]): A list of 4x4 SE(3) transformation
                                               matrices for the end-effector's pose
                                               in the world frame for each scene.

    Returns:
        o3d.geometry.PointCloud: A single, merged point cloud in the end-effector's
                                 coordinate frame.
    """
    if not isinstance(pcd_list_cam_frame, list) or not pcd_list_cam_frame:
        print("Error: pcd_list_cam_frame must be a non-empty list.")
        return o3d.geometry.PointCloud()

    # Create an empty point cloud to store the merged result
    merged_pcd_ee_frame = o3d.geometry.PointCloud()

    # The transformation from the world frame to the camera frame is the inverse
    # of the camera's world pose.
    world_to_cam_pose = np.linalg.inv(camera_world_pose)

    for i, pcd_cam in enumerate(pcd_list_cam_frame):
        ee_world_pose = ee_world_pose_list[i]

        pcd_ee = pcd_cam.transform(np.linalg.inv(ee_world_pose) @ camera_world_pose)

        # 3. Merge the transformed point cloud into the final result
        merged_pcd_ee_frame += pcd_ee

    return merged_pcd_ee_frame