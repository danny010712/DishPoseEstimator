import numpy as np
# from io.file_io import save_pointcloud
# from utils.visualization import visualize_pca_info
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from config import VOXEL_SIZE, DISTANCE_THRESHOLD, MAX_ITER_RANSAC, THRESHOLD_RANSAC
import copy

def get_pca_info(pcd, distance_threshold = DISTANCE_THRESHOLD):
    """
    Finds the centroid and principal axes of a point cloud using PCA.
    It identifies the rotational axis by finding the most dissimilar eigenvalue
    and sets this axis as the z-axis of a right-handed coordinate system.

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud.

    Returns:
        tuple: A tuple containing:
               - centroid (np.ndarray): The 3D centroid of the point cloud.
               - eigenvalues (np.ndarray): The eigenvalues representing variance along each axis.
               - eigenvectors (np.ndarray): A 3x3 matrix where the third column is the
                                           rotational axis, forming a right-handed frame.
    """
    points = np.asarray(pcd.points)

    # Calculate the centroid and covariance matrix
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    cov_matrix = np.cov(centered_points, rowvar=False)

    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors by magnitude in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Find the most dissimilar eigenvalue pair
    transformed_points = centered_points @ eigenvectors
    max_coords = np.max(transformed_points, axis=0)
    min_coords = np.min(transformed_points, axis=0)

    # Calculate dimensions
    dimensions = max_coords - min_coords

    # Type 1
    # ratio_12 = eigenvalues[0] / eigenvalues[1]
    # ratio_23 = eigenvalues[1] / eigenvalues[2]
    # Type 2
    ratio_12 = dimensions[0] / dimensions[1]
    ratio_23 = dimensions[1] / dimensions[2] ## check if this works
    # Type 3
    # ratio_12 = eigenvalues[0] - eigenvalues[1]
    # ratio_23 = eigenvalues[1] - eigenvalues[2]


    # Check if the rotational axis is the largest or smallest variance axis
    if ratio_12 > ratio_23:
        # The first two eigenvalues are different from each other.
        # This implies the object is tall (e.g., a tumbler), where two small eigenvalues
        # are similar but the first (largest) is very different.
        # The axis of largest variance (v1) is the rotational axis.
        new_z_axis = eigenvectors[:, 0]
        new_x_axis = eigenvectors[:, 1]
        new_y_axis = eigenvectors[:, 2]
        eigenvalues = eigenvalues[[1, 2, 0]]
    else:
        # The first two eigenvalues are similar, but the third is very different.
        # This implies the object is flat (e.g., a dish), where two big eigenvalues
        # are similar, but the last (smallest) is very different.
        # The axis of smallest variance (v3) is the rotational axis.
        new_z_axis = eigenvectors[:, 2]
        new_x_axis = eigenvectors[:, 0]
        new_y_axis = eigenvectors[:, 1]
        eigenvalues = eigenvalues[[0, 1, 2]]

    # Create the final eigenvector matrix, ensuring it's right-handed
    final_eigenvectors = np.vstack([new_x_axis, new_y_axis, new_z_axis]).T
    if np.linalg.det(final_eigenvectors) < 0:
        final_eigenvectors[:, 1] = -final_eigenvectors[:, 1]

    # points_in_pca_frame = centered_points @ np.linalg.inv(final_eigenvectors)
    points_in_pca_frame = centered_points @ final_eigenvectors

    # distance_threshold = 0.005
    distances_from_z_axis_sq = points_in_pca_frame[:, 0]**2 + points_in_pca_frame[:, 1]**2
    candidate_indices = np.where(distances_from_z_axis_sq < distance_threshold**2)[0]
    # print(f"Found {len(candidate_indices)} candidates for lowest point.")

    if len(candidate_indices) > 0:
        candidate_points = points_in_pca_frame[candidate_indices]

        # Find the point with the minimum z-value among the candidates
        min_z_idx_in_candidates = np.argmin(candidate_points[:, 2])
        # Find the point with the maximum z-value among the candidates
        max_z_idx_in_candidates = np.argmax(candidate_points[:, 2])
        is_flipped = False

        # if the candidate points are much closer to the points where z component is big, consider positive z dir is set to be the bottom, which we want to be flipped
        if np.max(points_in_pca_frame[:, 2]) - candidate_points[max_z_idx_in_candidates, 2] < abs(np.min(points_in_pca_frame[:, 2]) - candidate_points[min_z_idx_in_candidates, 2]):
          min_z_idx_in_candidates = max_z_idx_in_candidates
          is_flipped = True

    else:
        # print("No candidates, cannot know whether frame is flipped..")
        is_flipped = True ## to change

    if is_flipped:
        # print("I think the frame is flipped..??")
        # Flip the rotational axis to point "up"
        final_eigenvectors[:, 2] = -final_eigenvectors[:, 2]
        # Re-calculate the y-axis to maintain a right-handed coordinate system
        final_eigenvectors[:, 1] = -final_eigenvectors[:, 1]

    points_in_pca_frame = centered_points @ final_eigenvectors

    z_min = np.min(points_in_pca_frame[:, 2])
    z_max = np.max(points_in_pca_frame[:, 2])
    x_min = np.min(points_in_pca_frame[:, 0])
    x_max = np.max(points_in_pca_frame[:, 0])
    y_min = np.min(points_in_pca_frame[:, 1])
    y_max = np.max(points_in_pca_frame[:, 1])
    bounding_box_pca_frame = np.array([[x_min, y_min, z_min],
                                       [x_min, y_min, z_max],
                                       [x_min, y_max, z_min],
                                       [x_min, y_max, z_max],
                                       [x_max, y_min, z_min],
                                       [x_max, y_min, z_max],
                                       [x_max, y_max, z_min],
                                       [x_max, y_max, z_max]])
    bounding_box = bounding_box_pca_frame @ final_eigenvectors.T + centroid

    lowest_point_coords = np.array([(x_max+x_min)/2, (y_max+y_min)/2, z_min]) @ final_eigenvectors.T + centroid
    print(eigenvalues)
    print(final_eigenvectors)
    return centroid, lowest_point_coords, eigenvalues, final_eigenvectors, bounding_box


def obb_volume_cost(centroid, eigenvectors, points):
    """
    Objective function to minimize the volume of an oriented bounding box.

    Args:
        centroid(np.ndarray): 1x3 centroid.
        eigenvectors(np.ndarray): 3x3 eigenvectors
        points (np.ndarray): The Nx3 point cloud data.

    Returns:
        float: The volume of the tightest-fitting bounding box for the given pose.
    """
    # initial centroid
    # t = centroid
    t = np.mean(points, axis=0)
    # Create the rotation matrix
    R_mat = eigenvectors

    # Transform points to the bounding box's local frame(pca frame)
    transformed_points = (points - t) @ R_mat

    # Find the extents of the transformed points
    max_coords = np.max(transformed_points, axis=0)
    min_coords = np.min(transformed_points, axis=0)

    # Calculate dimensions
    dimensions = max_coords - min_coords

    # The cost is the volume of this box
    volume = dimensions[0] * dimensions[1] * dimensions[2]

    return volume

def find_optimal_obb(pcd, initial_centroid, initial_eigenvectors):
    """
    Finds the most fitting oriented bounding box using iterative optimization.

    Args:
        pcd (open3d.geometry.PointCloud): The point cloud data.
        initial_centroid (np.ndarray): The initial centroid from PCA.
        initial_eigenvectors (np.ndarray): The initial rotation matrix from PCA.

    Returns:
        np.ndarray: An 8x3 array representing the corners of the optimal OBB.
    """
    points = np.asarray(pcd.points)

    # Create a wrapper function that maps the parameter vector to your cost function's arguments
    def wrapper_cost_function(params):
        # Unpack the single parameter vector into centroid and rotation
        centroid = params[0:3]

        # Convert Euler angles to a rotation matrix
        r_euler = params[3:6]
        eigenvectors = R.from_euler('xyz', r_euler).as_matrix()

        return obb_volume_cost(centroid, eigenvectors, points) # use OBB fitting method

    # Initial guess combines centroid and rotation (as Euler angles)
    r_init = R.from_matrix(initial_eigenvectors).as_euler('xyz')
    initial_params = np.concatenate([initial_centroid, r_init])

    # Use Nelder-Mead for a robust, derivative-free optimization
    result = minimize(wrapper_cost_function, initial_params, method='Nelder-Mead')

    # Unpack the optimized parameters
    optimized_params = result.x
    t_opt = optimized_params[0:3]
    r_opt = optimized_params[3:6]

    # Reconstruct the final rotation matrix
    R_opt = R.from_euler('xyz', r_opt).as_matrix()

    # Find the final dimensions of the box after optimization
    transformed_points_opt = (points - t_opt) @ R_opt
    min_coords_opt = np.min(transformed_points_opt, axis=0)
    max_coords_opt = np.max(transformed_points_opt, axis=0)
    dimensions_opt = max_coords_opt - min_coords_opt

    # Generate the 8 corner points of the final bounding box
    box_vertices_local = np.array([
        [min_coords_opt[0], min_coords_opt[1], min_coords_opt[2]],
        [min_coords_opt[0], min_coords_opt[1], max_coords_opt[2]],
        [min_coords_opt[0], max_coords_opt[1], min_coords_opt[2]],
        [min_coords_opt[0], max_coords_opt[1], max_coords_opt[2]],
        [max_coords_opt[0], min_coords_opt[1], min_coords_opt[2]],
        [max_coords_opt[0], min_coords_opt[1], max_coords_opt[2]],
        [max_coords_opt[0], max_coords_opt[1], min_coords_opt[2]],
        [max_coords_opt[0], max_coords_opt[1], max_coords_opt[2]]
    ])

    # Transform local vertices to the world frame
    box_vertices_world = box_vertices_local @ R_opt.T + t_opt

    lowest_point_opt = np.array([(min_coords_opt[0]+max_coords_opt[0])/2, (min_coords_opt[1]+max_coords_opt[1])/2, min_coords_opt[2]]) @ R_opt.T + t_opt
    center_opt = np.array([(min_coords_opt[0]+max_coords_opt[0])/2, (min_coords_opt[1]+max_coords_opt[1])/2, (min_coords_opt[2]+max_coords_opt[2])/2]) @ R_opt.T + t_opt

    return center_opt, R_opt, lowest_point_opt, box_vertices_world

def fit_circle_ransac(points, max_iterations=MAX_ITER_RANSAC, threshold=THRESHOLD_RANSAC):
    """
    Detects a circle from a point cloud using RANSAC.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        max_iterations (int): Maximum number of RANSAC iterations.
        threshold (float): Distance threshold for inlier determination.

    Returns:
        tuple: A tuple containing:
            - The best-fit circle's center (numpy array).
            - The best-fit circle's radius (float).
            - The best-fit circle's normal vector (numpy array).
            - The indices of the inlier points.
    """
    best_inliers = []
    best_radius = -1
    best_circle_params = None
    radius_weight = 0

    for _ in range(max_iterations):
        # 1. Randomly sample 3 non-collinear points
        sample_indices = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[sample_indices]

        # Check for collinearity to avoid singular matrix
        if np.linalg.norm(np.cross(p2 - p1, p3 - p1)) < 1e-6:
            continue

        # 2. Compute circle parameters from the 3 points
        # The plane normal is found by the cross product of two vectors on the plane
        normal_vector = np.cross(p2 - p1, p3 - p1)
        normal_vector /= np.linalg.norm(normal_vector)
        if normal_vector[2] < 0:
            normal_vector = -normal_vector

        # We find the circle center by solving a system of equations
        # Let's use a geometric approach: center is intersection of perpendicular bisectors
        v1 = p2 - p1
        v2 = p3 - p2

        if np.dot(v1, v2) < 1e-6: # Check if points are too close
            continue

        D = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        if abs(D) < 1e-6:
            continue

        center_x = ((p1[0]**2 + p1[1]**2) * (p2[1] - p3[1]) + (p2[0]**2 + p2[1]**2) * (p3[1] - p1[1]) + (p3[0]**2 + p3[1]**2) * (p1[1] - p2[1])) / D
        center_y = ((p1[0]**2 + p1[1]**2) * (p3[0] - p2[0]) + (p2[0]**2 + p2[1]**2) * (p1[0] - p3[0]) + (p3[0]**2 + p3[1]**2) * (p2[0] - p1[0])) / D

        # The circle center also lies on the plane defined by the 3 points
        d = -np.dot(normal_vector, p1)
        center_z = (-normal_vector[0]*center_x - normal_vector[1]*center_y - d) / normal_vector[2]

        center = np.array([center_x, center_y, center_z])
        radius = np.linalg.norm(center - p1)

        # 3. Count inliers
        inliers = []
        for i, p in enumerate(points):
            # Check if point lies on the same plane
            dist_to_plane = abs(np.dot(p - center, normal_vector))
            if dist_to_plane > threshold:
                continue

            # Check if point is at the same radius from the center on the plane
            dist_to_center = np.linalg.norm(p - center)
            if abs(dist_to_center - radius) < threshold:
                inliers.append(i)

        if len(inliers) + radius*radius_weight > len(best_inliers) + best_radius*radius_weight:
            best_inliers = inliers
            best_radius = radius
            best_circle_params = (center, radius, normal_vector)

    if best_circle_params:
        return best_circle_params, best_inliers
    else:
        return None, None

def refine_pose_with_circles(pcd, initial_centroid, initial_eigenvectors, voxel_size = VOXEL_SIZE):
    """
    Finds the most fitting oriented bounding box using iterative optimization.

    Args:
        pcd (open3d.geometry.PointCloud): The point cloud data.
        initial_centroid (np.ndarray): The initial centroid from PCA+OBB.
        initial_eigenvectors (np.ndarray): The initial rotation matrix from PCA+OBB.

    Returns:
        tuple: A tuple containing the optimized centroid, rotation matrix, lowest point, and the expected bb vertices.
    """
    points = np.asarray(pcd.points)

    transformed_points = (points - initial_centroid) @ initial_eigenvectors

    # 2. Slice the point cloud
    min_z = np.min(transformed_points[:, 2])
    max_z = np.max(transformed_points[:, 2])
    num_slices = int((max_z-min_z) / voxel_size * 2) # YOU CAN EDIT THIS

    z_intervals = np.linspace(min_z, max_z, num_slices + 1)

    all_features = []

    for i in range(num_slices):
        # Get points within the current z-slice
        slice_points_mask = (transformed_points[:, 2] >= z_intervals[i]) & \
                            (transformed_points[:, 2] < z_intervals[i+1])
        slice_points = transformed_points[slice_points_mask]

        if len(slice_points) < 3:
            continue

        best_arc_info, best_arc_inliers = fit_circle_ransac(slice_points)

        if best_arc_info:
            center, radius, normal = best_arc_info
            # Create the feature vector [center_x, center_y, center_z, normal_x, normal_y, normal_z]
            feature_vector = np.concatenate([center, normal, [radius]])
            all_features.append(feature_vector)

    if not all_features:
        return

    features_matrix = np.array(all_features)

    # update
    mean_radius = np.mean(features_matrix[:, 6], axis = 0)
    current_centroid_local = np.mean(features_matrix[:, :3], axis = 0)
    current_centroid_local[2] = (min_z+max_z)/2
    print(features_matrix)
    print(current_centroid_local)

    t_opt = current_centroid_local @ initial_eigenvectors.T + initial_centroid # p_oc + p_cc' = p_oc + R_oc * p_cc'
    R_opt = initial_eigenvectors

    # Find the final dimensions of the box after optimization
    transformed_points_opt_refined = (points - t_opt) @ R_opt

    # Find the lowest point that is on the central axis
    lowest_point_opt = np.array([0, 0, np.min(transformed_points_opt_refined[:, 2])]) @ R_opt.T + t_opt

    dimensions = np.max(np.abs(transformed_points_opt_refined), axis=0)
    max_radius = np.max(np.abs(transformed_points_opt_refined[:, :2]))

    x_min = -dimensions[0]
    x_max = dimensions[0]
    y_min = -dimensions[1]
    y_max = dimensions[1]
    z_min = -dimensions[2]
    z_max = -dimensions[2]


    # 원래 방법
    # min_coords_opt_refined = np.min(transformed_points_opt_refined, axis=0)
    # max_coords_opt_refined = np.max(transformed_points_opt_refined, axis=0)
    # dimensions_op_refined = max_coords_opt_refined - min_coords_opt_refined

    # 새로운 방법들
    # x_min = min_coords_opt_refined[0]
    # x_max = max_coords_opt_refined[0]
    # y_min = min_coords_opt_refined[1]
    # y_max = max_coords_opt_refined[1]
    # z_min = min_coords_opt_refined[2]
    # z_max = max_coords_opt_refined[2]

    x_min = -max_radius
    x_max = max_radius
    y_min = -max_radius
    y_max = max_radius
    z_min = np.min(transformed_points_opt_refined[:, 2])
    z_max = np.max(transformed_points_opt_refined[:, 2])


    bounding_box_pca_frame = np.array([[x_min, y_min, z_min],
                                       [x_min, y_min, z_max],
                                       [x_min, y_max, z_min],
                                       [x_min, y_max, z_max],
                                       [x_max, y_min, z_min],
                                       [x_max, y_min, z_max],
                                       [x_max, y_max, z_min],
                                       [x_max, y_max, z_max]])
    refined_box_vertices = bounding_box_pca_frame @ R_opt.T + t_opt


    
    
    centers = features_matrix[:, 0:2]
    normals = features_matrix[:, 3:6]
    radii = features_matrix[:, 6]

    # --- (1) center 기반 ---
    center_mean = np.mean(centers, axis=0)
    center_dist = np.linalg.norm(centers - center_mean, axis=1)
    center_thresh = 1*np.std(center_dist)
    # center_thresh = 0.05
    mask_center = center_dist < center_thresh

    # --- (2) normal 기반 ---
    normal_mean = np.mean(normals, axis=0)
    normal_mean /= np.linalg.norm(normal_mean)
    angle_diff = np.arccos(np.clip(normals @ normal_mean, -1.0, 1.0))  # 각도 차이
    angle_thresh = np.deg2rad(10)  # 10도 이상 차이 나는 normal 제거
    mask_normal = angle_diff < angle_thresh

    # --- (3) radius 기반 ---
    radius_mean = np.mean(radii)
    mask_radius = np.abs(radii - radius_mean) < 100 * np.std(radii)

    # --- (4) 종합 mask ---
    mask = mask_center & mask_normal & mask_radius

    # outlier 제거된 features
    features_filtered = features_matrix[mask]


    # ✅ Local → World 변환 (return 직전에)
    features_matrix_world = copy.deepcopy(features_filtered)
    # center 변환: (x, y, z) -> columns 0~2
    features_matrix_world[:, :3] = features_filtered[:, :3] @ initial_eigenvectors.T + initial_centroid
    # normal 변환: (nx, ny, nz) -> columns 3~5
    features_matrix_world[:, 3:6] = (initial_eigenvectors @ features_filtered[:, 3:6].T).T

    return t_opt, R_opt, lowest_point_opt, refined_box_vertices, mean_radius, features_matrix_world