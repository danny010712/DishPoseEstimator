import os
import numpy as np
import open3d as o3d
from inputoutput.capture import capture_pointcloud, load_and_create_intrinsics, create_pcd_from_rgbd, create_pcd_from_rgbd_with_mask
from inputoutput.file_io import save_pointcloud, load_pointcloud
from processing.segmentation import crop_around, cluster_point_cloud_xyz, FastSAMseg
from processing.merging import register_point_clouds
from processing.pose_estimation import get_pca_info, find_optimal_obb, refine_pose_with_circles, fit_fixed_cylinder, fit_infinite_cylinder, fit_cylinder_fixed_axis, refine_pose_with_mec
from utils.visualization import set_axes_equal, show_pointcloud, visualize_step_results, visualize_pca_info
from utils.mathfunc import to_se3, pose_error
from config import BASE_PATH, DATA_DIR, RESULTS_DIR, SCENE_NUM, CAPTURE_WIDTH, CAPTURE_HEIGHT, DEPTH_SCALE_FACTOR, DEPTH_TRUNCATION, CROP_THRESHOLD, EPS, MIN_POINTS, VOXEL_SIZE, DISTANCE_THRESHOLD, SAVE_INTERMEDIATE
from datetime import datetime

def all_in_one(pcd = [], camera_frame = np.eye(4), gripper_frame = None, return_type = 'world'): # Wrap-up function for whole process
    """
    returns 6d world poses(position, orientation) of the objects.

    Args:
        pcd (list[o3d.geometry.PointCloud]): list of point clouds from different scenes in camera frame. # or rgb-d style data
        camera_frame (np.ndarray): 4x4 SE(3) transformation matrix for the world pose of the camera.
        gripper_frame (np.ndarray): A stack(Nx4x4) of 4x4 SE(3) transformation matrices for the world poses of the gripper in different scenes.
        return_type (string): choose in which frame to return poses repect to.

    Returns:
        estimated_pose (np.ndarray) : 4x4 SE(3) transformation matrix for the pose of the object.
    """
    true_pose = None

    scene_num = len(pcd)
    if scene_num < 1:
        print("ERROR: no point cloud data")
        return
    if scene_num != len(gripper_frame):
        print("ERROR: mismatch of number of data")
        print(f"{scene_num} data, {len(gripper_frame)} gripper frames, cannot merge point clouds")
        return

    pcd_segmented = []
    for i in range(scene_num):
        ########## Cropping around end-effector point ##########
        ref = (np.linalg.inv(camera_frame) @ gripper_frame[i])[:3, 3] # T_cg = T_co @ T_og
        cropped_pcd, crop_indices = crop_around(pcd[i], ref, threshold=CROP_THRESHOLD)
        print(f"{len(cropped_pcd.points)} left after crop.")
        
        visualize_step_results(pcd[i], crop_indices, Title = "After Cropping")

        clustered_pcd, clustered_indices = cluster_point_cloud_xyz(cropped_pcd, eps=EPS, min_points=MIN_POINTS)

        visualize_step_results(cropped_pcd, clustered_indices, Title = "After Clustering")

        if SAVE_INTERMEDIATE:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_pointcloud(cropped_pcd, os.path.join(RESULTS_DIR, f"cropped_{i}_{timestamp}"))
            save_pointcloud(clustered_pcd, os.path.join(RESULTS_DIR, f"clustered_{i}_{timestamp}"))

        pcd_segmented.append(clustered_pcd) # list of segmented point clouds, ready to merge.


    pcd_merged = register_point_clouds(camera_frame, pcd_segmented, gripper_frame)

    # downsampled_pcd_merged = pcd_merged.voxel_down_sample(VOXEL_SIZE)
    downsampled_pcd_merged = pcd_merged # without downsampling

    centroid, lowest_point, _, eigenvectors, bounding_box_frame = get_pca_info(downsampled_pcd_merged, DISTANCE_THRESHOLD)
    # visualize_pca_info(downsampled_pcd_merged, centroid, lowest_point, eigenvectors, true_pose, bounding_box_frame, Title = "After PCA")
    center_opt, eigenvectors_opt, lowest_point_opt, box_vertices_world_opt = find_optimal_obb(downsampled_pcd_merged, centroid, eigenvectors)
    visualize_pca_info(downsampled_pcd_merged, center_opt, lowest_point_opt, eigenvectors_opt, true_pose, box_vertices_world_opt, None, Title = "After Finding OBB")
    refined_centroid, refined_eigenvectors, refined_lowest_point, refined_box_vertices_world, mec_radius, features_matrix = refine_pose_with_mec(downsampled_pcd_merged, center_opt, eigenvectors_opt)
    visualize_pca_info(downsampled_pcd_merged, refined_centroid, refined_lowest_point, refined_eigenvectors, true_pose, refined_box_vertices_world, features_matrix, Title = "After Refinement")
    

    # mean_radius = mec_radius
    # print(f"Mean radius: {mean_radius}")

    estimated_pose = np.eye(4)
    estimated_pose[:3, :3] = refined_eigenvectors
    estimated_pose[:3, 3] = refined_lowest_point

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

    ########################## Example Usage 0(Real Application) ##########################
    
    ### === Check results in /results === ###
    scene_num = 1
    input_pcd = []
    for i in range(scene_num):
        pcd, rgb, depth, intrinsics = capture_pointcloud(height=CAPTURE_HEIGHT, width=CAPTURE_WIDTH, depth_limit = DEPTH_TRUNCATION)
        show_pointcloud(pcd)
        mask = FastSAMseg(rgb)
        masked_pcd = create_pcd_from_rgbd_with_mask(rgb, depth, intrinsics, mask)
        input_pcd.append(masked_pcd)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_pointcloud(masked_pcd, f"fastsamsegmented_{timestamp}") # save the results right after fastsam seg
    camera_frame_world = np.eye(4)
    gripper_frame_world = np.array([np.eye(4)]) # To change
    gripper_frame_world[0][:3,3] = [0,0,0.5] # To change, gripper endpoint works for cropping.
    true_pose = None
    estimated_pose = all_in_one(input_pcd, camera_frame_world, gripper_frame_world, return_type = 'camera')
    
    print(f"Printing object pose...")

    for i in range(len(estimated_pose)):
        print(f"Object pose #{i} : {estimated_pose[i]}")


    #######################################################################################
    
    # # convert txt to ply
    # Object_name = ['YCBBowl', 'CustomBowl', 'CustomCup', 'CustomDish', 'CustomSquareDish', 'SlicedBowl']
    # scene_num = 3
    # for object_name in Object_name:
    #     for i in range(scene_num):
    #         file_path = os.path.join(DATA_DIR, f"dataset2/rsd455_dishonly_{i+1}_{object_name}.txt")
    #         data = np.loadtxt(file_path)
    #         pcd = o3d.geometry.PointCloud()
    #         pcd.points = o3d.utility.Vector3dVector(data[:, :3])
    #         pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255.0)
    #         save_pointcloud(pcd, f"{object_name}_{i}_dishonly")

    ########################## Example Usage 1(Algorithm Test on real data) ##########################

    # ## === Use dataset1 === ###
    # for select_scenenum in range(10):
    #     input_pcd = []
        
    #     K_matrix = np.load(os.path.join(DATA_DIR, "dataset1/K.npy"))
    #     intrinsics = load_and_create_intrinsics(K_matrix, CAPTURE_WIDTH, CAPTURE_HEIGHT)
    #     scene_num = 1 # single-view
    #     input_pcd = []
        
    #     color_filepath = os.path.join(DATA_DIR, f"dataset1/color{select_scenenum+1}.npy")
    #     depth_filepath = os.path.join(DATA_DIR, f"dataset1/depth{select_scenenum+1}.npy")
    #     colors = np.load(color_filepath)
    #     colors = colors[..., ::-1] # check if color order should be changed.
    #     depth = np.load(depth_filepath)
    #     pcd_raw = create_pcd_from_rgbd(colors, depth, intrinsics)
    #     save_pointcloud(pcd_raw, f"raw_{select_scenenum}") # save the raw results
    #     img_filepath = os.path.join(DATA_DIR, f"dataset1/color{select_scenenum+1}.png")
    #     seg_ref = [[400,400],[400,400],[400,400],[400,400],[400,350],[400,350],[500,300],[320,240],[350,300],[350,275]]
    #     mask = FastSAMseg(img_filepath, ref=seg_ref[select_scenenum])
    #     pcd = create_pcd_from_rgbd_with_mask(colors, depth, intrinsics, mask)
    #     input_pcd.append(pcd)
    #     save_pointcloud(pcd, f"fastsamsegmented_{select_scenenum}") # save the fastsam segmented results
        
    #     camera_frame_world = np.eye(4)
    #     gripper_frame_world = np.array([np.eye(4)])
    #     crop_ref = [[0.05, 0.05, 0.35],
    #         [0.05, 0.05, 0.35],
    #         [0.06, 0.06, 0.3],
    #         [0.05, 0.05, 0.35],
    #         [0.1, 0.05, 0.4],
    #         [0.08, 0.06, 0.4],
    #         [0.1, 0.05, 0.4],
    #         [0, -0.05, 0.35],
    #         [0, 0, 0.4],
    #         [0.05, -0.05, 0.4]]
    #     gripper_frame_world[0][:3, 3] = crop_ref[select_scenenum] # for cropping, assummed coordinates of gripper end points
    #     true_pose = None

    #     estimated_pose = all_in_one(input_pcd, camera_frame_world, gripper_frame_world, return_type = 'gripper')
    #     print(f"Printing object pose...")

    #     for i in range(len(estimated_pose)):
    #         print(f"Object pose #{i} : {estimated_pose[i]}")

    #################################################################################################


    ######################## Example Usage 2(Algorithm Test on Isaac Sim data) ########################

    # ### === Use dataset2 === ###
    # # input_pcd = []
    # # Load ply files
    # object_names = ['YCBBowl', 'CustomBowl', 'CustomCup', 'CustomDish', 'CustomSquareDish', 'SlicedBowl']
    # # object_names = ['SlicedBowl']
    # for object_name in object_names:
    #     input_pcd = []
    #     scene_num = 3
    #     for i in range(scene_num):
    #         input_pcd.append(load_pointcloud(DATA_DIR + f"/dataset2/{object_name}_{i}_dishonly.ply"))

    #     camera_frame_world = np.load(os.path.join(DATA_DIR, "dataset2/camera_frame_world.npy"))
    #     gripper_frame_world = np.load(os.path.join(DATA_DIR, "dataset2/gripper_frame_world.npy"))
    #     true_pose_gripper_frame = np.load(os.path.join(DATA_DIR, f"dataset2/{object_name}_GT_pose_gripper_frame.npy"))
    #     true_pose_camera_frame_1 = np.load(os.path.join(DATA_DIR, f"dataset2/{object_name}_GT_pose_camera_frame_1.npy"))
    #     true_pose_camera_frame_2 = np.load(os.path.join(DATA_DIR, f"dataset2/{object_name}_GT_pose_camera_frame_2.npy"))
    #     true_pose_camera_frame_3 = np.load(os.path.join(DATA_DIR, f"dataset2/{object_name}_GT_pose_camera_frame_3.npy"))
    #     true_pose_world_frame_1 = np.load(os.path.join(DATA_DIR, f"dataset2/{object_name}_GT_pose_world_frame_1.npy"))
    #     true_pose_world_frame_2 = np.load(os.path.join(DATA_DIR, f"dataset2/{object_name}_GT_pose_world_frame_2.npy"))
    #     true_pose_world_frame_3 = np.load(os.path.join(DATA_DIR, f"dataset2/{object_name}_GT_pose_world_frame_3.npy"))

    #     estimated_pose = all_in_one(input_pcd, camera_frame_world, gripper_frame_world, return_type = 'gripper') # Available to choose among 'gripper', 'camera', 'world'

    #     print(f"Printing object pose...")

    #     for i in range(len(estimated_pose)):
    #         print(f"Object pose #{i} : {estimated_pose[i]}")
        
    #     pose_error(estimated_pose[0], true_pose_gripper_frame) # 'gripper'
    #     # pose_error(estimated_pose[0], true_pose_camera_frame_1) # 'camera'
    #     # pose_error(estimated_pose[1], true_pose_camera_frame_2)
    #     # pose_error(estimated_pose[2], true_pose_camera_frame_3)
    #     # pose_error(estimated_pose[0], true_pose_world_frame_1) # 'world'
    #     # pose_error(estimated_pose[1], true_pose_world_frame_2)
    #     # pose_error(estimated_pose[2], true_pose_world_frame_3)

    ################################################################################################
    

if __name__ == "__main__":
    main()

    ##### For quick check, comment the above main() and uncomment the below to visualize results #####
    # pcd = load_pointcloud(os.path.join(RESULTS_DIR, "raw_8.ply")) # To edit
    # show_pointcloud(pcd)