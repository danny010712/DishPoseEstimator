import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from inputoutput.file_io import save_pointcloud
from config import CAPTURE_HEIGHT, CAPTURE_WIDTH, CROP_THRESHOLD, EPS, MIN_POINTS, SAVE_INTERMEDIATE, BASE_PATH, RESULTS_DIR, CONF, IOU, MODEL_TYPE
from utils.visualization import visualize_step_results
from FastSAM.fastsam import FastSAM, FastSAMPrompt
import numpy as np
import os
from datetime import datetime

def crop_around(pcd, ref=None, threshold = CROP_THRESHOLD):
    if ref is None:
        return
    points_raw = np.asarray(pcd.points)
    colors_raw = np.asarray(pcd.colors)

    cropped_points_indices = np.where(np.linalg.norm(points_raw - ref, axis=1)<threshold)[0]
    points_cropped = points_raw[cropped_points_indices]
    colors_cropped = colors_raw[cropped_points_indices]

    # Create a new point cloud object from the segmented data
    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(points_cropped)
    cropped_pcd.colors = o3d.utility.Vector3dVector(colors_cropped)


    return cropped_pcd, cropped_points_indices

# def cluster_point_cloud(  # deprecated
#     pcd, eps=EPS, min_points=MIN_POINTS, color_weight=COLOR_WEIGHT, color_ref = COLOR_REF, mask_threshold=COLOR_THRESHOLD
# ):
#     """
#     Performs DBSCAN clustering on a point cloud (XYZ + RGB features),
#     while preserving mapping to the original point cloud indices.
    
#     Steps:
#     1. Color-based filtering (remove dark/black points)
#     2. Feature construction and DBSCAN clustering
#     """

#     points_all = np.asarray(pcd.points)
#     colors_all = np.asarray(pcd.colors) if pcd.has_colors() else None
#     num_points_total = len(points_all)
#     print(f"Loaded point cloud with {num_points_total} points.")
#     original_indices = np.arange(len(points_all))

#     # --- Step 1: Color-based filtering ---
#     print("Step 1: Removing gripper regions...")
#     if colors_all is not None:
        
#         # color_mask = (colors_all[:,0] > mask_threshold) | (colors_all[:,1] > mask_threshold) | (colors_all[:,2] > mask_threshold)
#         # color_mask = np.linalg.norm(colors_all - color_ref, axis=1) > mask_threshold

#         color_std = np.std(colors_all, axis=1)
#         color_mask = color_std > mask_threshold   # 예: mask_threshold = 0.1

#         # map to original indices
#         color_filtered_indices = original_indices[np.where(color_mask)[0]]
#         points_filtered = points_all[np.where(color_mask)[0]]
#         colors_filtered = colors_all[np.where(color_mask)[0]]

#         black_indices = original_indices[np.where(~color_mask)[0]]

#         print(f"Removed {np.sum(~color_mask)} gripper points; {len(points_filtered)} remain after color filter.")
#     else:
#         color_filtered_indices = original_indices
#         points_filtered = points_all
#         colors_filtered = None

#         black_indices = np.array([], dtype=int)

#         print("No color information; skipping color filtering.")

#     # --- Step 1.5: Feature vector construction ---
#     if colors_filtered is not None:
#         features = np.hstack((points_filtered, color_weight * colors_filtered))
#         print(f"Using color information in clustering (weight={color_weight}).")
#     else:
#         features = points_filtered
#         print("Clustering based on XYZ coordinates only.")

#     # --- Step 2: DBSCAN clustering ---
#     print("Step 2: Running DBSCAN clustering...")
#     clustering = DBSCAN(eps=eps, min_samples=min_points).fit(features)
#     labels = clustering.labels_

#     max_label = labels.max()
#     print(f"DBSCAN found {max_label + 1 if max_label >= 0 else 0} clusters.")

#     if max_label < 0:
#         print("No clusters found. Try adjusting parameters.")
#         return None, None

#     # --- Step 3: Select largest cluster ---
#     cluster_sizes = [np.sum(labels == i) for i in range(max_label + 1)]
#     largest_cluster_label = np.argmax(cluster_sizes)
#     largest_cluster_size = cluster_sizes[largest_cluster_label]

#     print(f"Largest cluster = {largest_cluster_label}, size = {largest_cluster_size} points.")

#     cluster_indices_in_filtered = np.where(labels == largest_cluster_label)[0]
#     cluster_indices_in_original = color_filtered_indices[cluster_indices_in_filtered]

#     # --- Step 4: Extract clustered object ---
#     object_pcd = pcd.select_by_index(cluster_indices_in_original)
#     print(f"Selected the largest cluster with {len(object_pcd.points)} points.")

#     return object_pcd, cluster_indices_in_original, black_indices


def cluster_point_cloud_xyz(
    pcd, eps=EPS, min_points=MIN_POINTS
):
    """
    Performs DBSCAN clustering on a point cloud (XYZ features).
    """

    points_all = np.asarray(pcd.points)
    num_points_total = len(points_all)
    print(f"Loaded point cloud with {num_points_total} points.")
    # original_indices = np.arange(len(points_all))

    
    # --- Step 2: DBSCAN clustering ---
    print("Step 2: Running DBSCAN clustering...")
    clustering = DBSCAN(eps=eps, min_samples=min_points).fit(points_all)
    labels = clustering.labels_

    max_label = labels.max()
    print(f"DBSCAN found {max_label + 1 if max_label >= 0 else 0} clusters.")

    if max_label < 0:
        print("No clusters found. Try adjusting parameters.")
        return None, None

    # --- Step 3: Select largest cluster ---
    cluster_sizes = [np.sum(labels == i) for i in range(max_label + 1)]
    largest_cluster_label = np.argmax(cluster_sizes)
    largest_cluster_size = cluster_sizes[largest_cluster_label]

    print(f"Largest cluster = {largest_cluster_label}, size = {largest_cluster_size} points.")

    cluster_indices = np.where(labels == largest_cluster_label)[0]

    # --- Step 4: Extract clustered object ---
    object_pcd = pcd.select_by_index(cluster_indices)
    print(f"Selected the largest cluster with {len(object_pcd.points)} points.")

    return object_pcd, cluster_indices


def FastSAMseg(img_path, ref=np.array([CAPTURE_WIDTH // 2, CAPTURE_HEIGHT // 2]), conf=CONF, iou=IOU):
    """
    img_path: path or np.ndarray type.
    """
    model = FastSAM(os.path.join(BASE_PATH, f'FastSAM/weights/{MODEL_TYPE}.pt'))
    # img = np.load(image_path)
    # IMAGE_PATH = os.path.join(DATA_DIR, f"{image_name}.png")
    DEVICE = 'cpu'
    everything_results = model(img_path, device=DEVICE, retina_masks=True, imgsz=1024, conf=CONF, iou=IOU)
    prompt_process = FastSAMPrompt(img_path, everything_results, device=DEVICE)

    ann = prompt_process.point_prompt(points=[ref], pointlabel=[1])

    if SAVE_INTERMEDIATE:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_process.plot(annotations=ann,output_path=os.path.join(RESULTS_DIR, f'FastSAMresult_{timestamp}.jpg'))

    return ann[0] # returns H*W boolean mask. H: image height, W: image width
    




# def segment(pcd, ref, save_dir = SAVE_INTERMEDIATE): # wrap-up fuction for crop-cluster segmentation
    
#     segmented_pcd = o3d.geometry.PointCloud()

#     cropped_pcd, crop_indices = crop_around(pcd, ref, threshold=0.2)
#     # cropped_pcd_file_path = os.path.join(RESULTS_DIR, f"cropped_{i}.ply")
#     # o3d.io.write_point_cloud(cropped_pcd_file_path, cropped_pcd)
#     # print(f"Cropped point cloud saved to: {cropped_pcd_file_path}")
#     # visualize_step_results(pcd, crop_indices)

#     clustered_pcd, clustered_indices, black_indices = cluster_point_cloud(cropped_pcd, eps=0.015, min_points=25, color_weight = 0.5, mask_threshold=0.14) #0.01, 30, 0.85 / 0.1, 30, 10 / 0.015, 30, 0.5 / 0.15, 25, 10
#     # print(cropped_pcd.colors[clustered_indices[0]])
#     # clustered_pcd_file_path = os.path.join(base_path, f"point_cloud_clustered_{i}.ply")
    
#     segmented_pcd = clustered_pcd
#     # o3d.io.write_point_cloud(segmented_pcd_file_path, clustered_pcd)
#     # print(f"Segmented point cloud saved to: {segmented_pcd_file_path}")
#     # visualize_step_results(cropped_pcd, clustered_indices, black_indices)

#     ########################################
#     return segmented_pcd