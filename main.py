import os
import numpy as np
import open3d as o3d
from inputoutput.capture import capture_pointcloud
from inputoutput.file_io import save_pointcloud, load_pointcloud
from processing.segmentation import crop_around, cluster_point_cloud
from processing.merging import register_point_clouds
from processing.pose_estimation import get_pca_info, find_optimal_obb, refine_pose_with_circles
from utils.visualization import set_axes_equal, show_pointcloud, visualize_step_results, visualize_pca_info
from utils.mathfunc import to_se3, pose_error
from config import BASE_PATH, RESULTS_DIR, SCENE_NUM, CAPTURE_WIDTH, CAPTURE_HEIGHT, DEPTH_SCALE_FACTOR, DEPTH_TRUNCATION, CROP_THRESHOLD, EPS, MIN_POINTS, COLOR_WEIGHT, COLOR_REF, COLOR_THRESHOLD, VOXEL_SIZE, DISTANCE_THRESHOLD, SAVE_INTERMEDIATE


def all_in_one(pcd = [], camera_frame = np.eye(4), gripper_frame = [], return_type = 'world'): # Wrap-up function for whole process
    """
    returns 6d world poses(position, orientation) of the objects.

    Args:
        pcd (list[o3d.geometry.PointCloud]): list of point clouds from different scenes in camera frame. # or rgb-d style data
        camera_frame (np.ndarray): 4x4 SE(3) transformation matrix for the world pose of the camera.
        gripper_frame (list[np.ndarray]): A list of 4x4 SE(3) transformation matrices for the world poses of the gripper in different scenes.
        return_type (string): choose in which frame to return poses repect to.

    Returns:
        estimated_pose (np.ndarray) : 4x4 SE(3) transformation matrix for the pose of the object.
    """
    ################## Hyperparameters #################
    true_pose = None # If we know the ground truth pose
    ####################################################

    scene_num = len(pcd)
    if scene_num < 1:
        print("ERROR: no point cloud data")
        return

    if scene_num != len(gripper_frame):
        print("ERROR: mismatch of data")
        print(f"{len(pcd)} point clouds, {len(gripper_frame)} gripper frames, cannot merge")
        return

    pcd_segmented = []
    for i in range(scene_num):
        ref = (np.linalg.inv(camera_frame) @ gripper_frame[i])[:3, 3] # T_cg = T_co @ T_og
        cropped_pcd, crop_indices = crop_around(pcd[i], ref, threshold=CROP_THRESHOLD)
        print(len(cropped_pcd.points))
        
        visualize_step_results(pcd[i], crop_indices, Title = "After Cropping")

        clustered_pcd, clustered_indices, black_indices = cluster_point_cloud(cropped_pcd, eps=EPS, min_points=MIN_POINTS, color_weight = COLOR_WEIGHT, color_ref = COLOR_REF, mask_threshold=COLOR_THRESHOLD) #0.01, 30, 0.85 / 0.1, 30, 10 / 0.015, 30, 0.5 / 0.15, 25, 10

        visualize_step_results(cropped_pcd, clustered_indices, black_indices, Title = "After Clustering")

        if(SAVE_INTERMEDIATE):
            save_pointcloud(cropped_pcd, RESULTS_DIR + f"/cropped_{i}")
            save_pointcloud(clustered_pcd, RESULTS_DIR + f"/clustered_{i}")

        pcd_segmented.append(clustered_pcd) # list of segmented point clouds, ready to merge.

    pcd_merged = register_point_clouds(camera_frame, pcd_segmented, gripper_frame)

    downsampled_pcd_merged = pcd_merged.voxel_down_sample(VOXEL_SIZE)

    centroid, lowest_point, _, eigenvectors, bounding_box_frame = get_pca_info(downsampled_pcd_merged, DISTANCE_THRESHOLD)
    # visualize_pca_info(downsampled_pcd_merged, centroid, lowest_point, eigenvectors, true_pose, bounding_box_frame, Title = "After PCA")
    center_opt, eigenvectors_opt, lowest_point_opt, box_vertices_world_opt = find_optimal_obb(downsampled_pcd_merged, centroid, eigenvectors)
    visualize_pca_info(downsampled_pcd_merged, center_opt, lowest_point_opt, eigenvectors_opt, true_pose, box_vertices_world_opt, Title = "After Finding OBB")
    refined_centroid, refined_eigenvectors, refined_lowest_point, refined_box_vertices_world = refine_pose_with_circles(downsampled_pcd_merged, center_opt, eigenvectors_opt, VOXEL_SIZE)
    visualize_pca_info(downsampled_pcd_merged, refined_centroid, refined_lowest_point, refined_eigenvectors, true_pose, refined_box_vertices_world, Title = "After Refinement")

    estimated_pose = np.eye(4)
    estimated_pose[:3, :3] = eigenvectors_opt
    estimated_pose[:3, 3] = lowest_point_opt

    if return_type == 'gripper':
        return [estimated_pose]
    elif return_type == 'camera':
        estimated_poses_camera = []
        for i in range(scene_num):
            # list of object poses respect to the camera, scene #1 ~ #N
            estimated_poses_camera.append(np.linalg.inv(camera_frame) @ gripper_frame[i] @ estimated_pose) # T_co * T_og * pose in {g}
        return estimated_poses_camera
    elif return_type == 'world':
        estimated_poses_world = []
        for i in range(scene_num):
            # list of object poses respect to the base(in WORLD FRAME), scene #1 ~ #N
            estimated_poses_world.append(gripper_frame[i] @ estimated_pose) # T_og * pose in {g}
        return estimated_poses_world
    else:
        print("Wrong return type!")
        return
    
def main():
    
    # scene_num = SCENE_NUM
    # input_pcd = []
    # for i in range(scene_num):
    #     pcd = capture_pointcloud(height=CAPTURE_HEIGHT, width=CAPTURE_WIDTH, depth_limit = DEPTH_TRUNCATION)
    #     input_pcd.append(pcd)
    #     save_pointcloud(pcd, f"raw_{i}") # save the raw results

    ##################################################### Algorithm test for real data #####################################################
    # input_pcd = [load_pointcloud(RESULTS_DIR + f"/raw0.ply")]
    # camera_frame_world = np.eye(4)
    # gripper_frame_world = [np.eye(4)]
    # ref = [[0.05, 0.05, 0.35],
    #        [0.05, 0.05, 0.35],
    #        [0.06, 0.06, 0.3],
    #        [0.05, 0.05, 0.35],
    #        [0.1, 0.05, 0.4],
    #        [0.08, 0.06, 0.4],
    #        [0.1, 0.05, 0.4],
    #        [0, -0.05, 0.35],
    #        [0, 0, 0.4],
    #        [0.05, -0.05, 0.4]]
    # gripper_frame_world[0][:3, 3] = ref[0]
    # true_pose = None
    #################################################################################################################################


    ################################################### Algorithm test for Isaac Sim data ####################################################################
    # YCB Bowl
    true_position = np.array([0, -0.02, 0.1]) # for YCBBowl
    true_orientation = np.array([ 0.258819, 0.9659258, 0, 0]) # Euler angles [150, 0, 0]

    # CustomBowl
    # true_position = np.array([0, -0.02, 0.1]) # for CustomBowl
    # true_orientation = np.array([ 0.258819, 0.9659258, 0, 0]) # Euler angles [150, 0, 0]

    # CustomDish
    # true_position = np.array([0, 0.03, 0.12]) # for CustomDish
    # true_orientation = np.array([0.7071068, 0.7071068, 0, 0]) # Euler angles [90, 0, 0]

    # CustomCup
    # true_position = np.array([0, 0.05, 0.16]) # for CustomDish
    # true_orientation = np.array([0, 1, 0, 0]) # Euler angles [180, 0, 0]

    # CustomSquareDish
    # true_position = np.array([0, 0.02, 0.1])
    # true_orientation = np.array([0.7071068, 0.7071068, 0, 0]) # Euler angles [90, 0, 0]

    
    camera_frame_world = np.array([[0, -1, 0, 0.5],
                        [-1, 0, 0, 0.0],
                        [0, 0, -1, 1.0],
                        [0, 0, 0, 1]])

    end_effector_data = [
        {'pos': np.array([ 0.5560329,  -0.21627031,  0.5110392 ]), 'ori': np.array([-0.5555985,   0.7286454,  -0.24826007,  0.31425026])},
        {'pos': np.array([ 0.5647746,  -0.226597,    0.50414854]), 'ori': np.array([-0.6323937,   0.43046516, -0.638184,   -0.08659799])},
        {'pos': np.array([ 0.5610763,  -0.22523722,  0.48907092]), 'ori': np.array([-0.45066842, -0.04291875, -0.7685194,  -0.45214382])}
    ]

    input_pcd = []
    # Load ply files
    scene_num = 3
    for i in range(scene_num):
        input_pcd.append(load_pointcloud(RESULTS_DIR + f"/YCBBowl_{i}.ply")) # Append None to maintain list length if an error occurs

    gripper_frame_world = []

    # Generate list of SE(3) matrices
    for i in range(len(end_effector_data)):
        ee_se3 = to_se3(end_effector_data[i]['pos'], end_effector_data[i]['ori'])
        gripper_frame_world.append(ee_se3)



    true_pose = to_se3(true_position, true_orientation)

    true_pose_1 = (np.linalg.inv(camera_frame_world) @ gripper_frame_world[0]) @ true_pose # T_co @ T_og @ T_g->obj
    true_pose_2 = (np.linalg.inv(camera_frame_world) @ gripper_frame_world[1]) @ true_pose
    true_pose_3 = (np.linalg.inv(camera_frame_world) @ gripper_frame_world[2]) @ true_pose

    true_pose_4 = gripper_frame_world[0] @ true_pose # T_og @ T_g->obj
    true_pose_5 = gripper_frame_world[1] @ true_pose
    true_pose_6 = gripper_frame_world[2] @ true_pose


    ###############################################################################################################

    estimated_pose = all_in_one(input_pcd, camera_frame_world, gripper_frame_world, return_type = 'gripper')

    print(f"Printing object pose...")

    for i in range(len(estimated_pose)):
        print(f"Object pose #{i} : {estimated_pose[i]}")
    
    pose_error(estimated_pose[0], true_pose) # 'gripper'
    # pose_error(estimated_pose[0], true_pose_1) # 'camera'
    # pose_error(estimated_pose[1], true_pose_2)
    # pose_error(estimated_pose[2], true_pose_3)
    # pose_error(estimated_pose[0], true_pose_4) # 'world'
    # pose_error(estimated_pose[1], true_pose_5)
    # pose_error(estimated_pose[2], true_pose_6)

    

if __name__ == "__main__":
    main()
    # pcd = load_pointcloud(RESULTS_DIR + f"/clustered_0.ply")
    # show_pointcloud(pcd)