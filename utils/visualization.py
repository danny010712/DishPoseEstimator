import numpy as np
import matplotlib.pyplot as plt
import copy
import open3d as o3d
from datetime import datetime
from config import RESULTS_DIR
import os

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])
    center = np.mean(limits, axis=1)
    radius = 0.5 * np.max(limits[:,1] - limits[:,0])
    for i in range(3):
        getattr(ax, f'set_{["x","y","z"][i]}lim')((center[i] - radius, center[i] + radius))


def show_pointcloud(pcd, elev=90, azim=-90, title="Point Cloud Visualization"): # almost deprecated
    # points_object = np.asarray(pcd.points)
    # colors_object = np.asarray(pcd.colors) if pcd.colors else 'red'
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points_object[:, 0], points_object[:, 1], points_object[:, 2], c=colors_object, s=1)
    # set_axes_equal(ax)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.view_init(elev=elev, azim=azim)
    # plt.title(title)
    # plt.tight_layout()
    # plt.show()
    o3d.visualization.draw_geometries([pcd])


def visualize_step_results(pcd, indices, black_indices = None, elev=-90, azim=-90, Title = "Intermediate result"): # For checking intermediate crop/segmented results
    """
    Visualizing function in segmentation step.
    pcd와 indices에 대해 indices에 해당하는 pcd를 green으로 highlight
    """
    points = copy.deepcopy(np.asarray(pcd.points))

    # 원래 색상 정보가 있으면 사용, 없으면 기본 색 (회색)
    if pcd.has_colors():
        colors = copy.deepcopy(np.asarray(pcd.colors))
    else:
        colors = np.full_like(points, fill_value=0.7)  # 회색 계열 (0.7, 0.7, 0.7)

    # all_indices = np.arange(len(points))
    # non_indices = np.setdiff1d(all_indices, indices)
    # rand_non_idx = np.random.choice(non_indices)
    # rand_high_idx = np.random.choice(indices)

    # # 선택된 점들의 정보 출력 (highlight 전 색상)
    # print(f"Non-highlight random point -> idx: {rand_non_idx}, xyzrgb: {np.concatenate([points[rand_non_idx], colors[rand_non_idx]])}")
    # print(f"Highlight random point (before green) -> idx: {rand_high_idx}, xyzrgb: {np.concatenate([points[rand_high_idx], colors[rand_high_idx]])}")

    # 하이라이트 색상 지정
    highlight_color = np.array([0.0, 1.0, 0.0])  # green
    elimination_color = np.array([0.0, 1.0, 1.0])  # cyan
    colors[indices] = highlight_color  # 인덱스에 해당하는 점만 초록색으로 덮기
    if black_indices is not None:
        colors[black_indices] = elimination_color
    # colors[[rand_non_idx, rand_high_idx]] = np.array([0, 0, 1])
    sizes = np.ones(len(points))*0.01
    # sizes[[rand_non_idx, rand_high_idx]] = 20

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=sizes)
    set_axes_equal(ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(Title)

    green_patch = plt.Line2D([0], [0], marker='o', color='w', label='Selected points',
                             markerfacecolor=highlight_color, markersize=8)
    
    cyan_patch = plt.Line2D([0], [0], marker='o', color='w', label='Masked points',
                            markerfacecolor=elimination_color, markersize=8)
    if black_indices is not None:
        ax.legend(handles=[green_patch, cyan_patch], loc='upper right')
    else:
        ax.legend(handles=[green_patch], loc='upper right')

    plt.tight_layout()
    plt.show()
    return


def visualize_pca_info(pcd, centroid = None, lowest_point = None, eigenvectors = None, true_pose = None, bounding_box_frame = None, features_matrix = None, Title = "PCA Info"):
    """
    Visualizes the point cloud with its centroid and principal axes.

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud.
        centroid (np.ndarray): The centroid of the point cloud.
        eigenvectors (np.ndarray): The principal axes of the point cloud.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    points = np.asarray(pcd.points)
    if len(pcd.colors) != 0:
        colors = np.asarray(pcd.colors)
    else:
        colors = 'black'
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors, alpha=0.5) #

    if centroid is not None:
        # Plot the centroid
        ax.scatter(centroid[0], centroid[1], centroid[2], c='black', s=100, label='Centroid')

    # Plot the lowest point
    ax.scatter(lowest_point[0], lowest_point[1], lowest_point[2], c='red', s=100, label='Lowest Point')

    # Plot the principal axes
    colors = ['r', 'g', 'b']  # Red for axis 1, Green for axis 2, Blue for axis 3
    for i in range(3):
        # The length of the axes can be scaled for better visualization
        axis_length = 0.05
        ax.quiver(
            lowest_point[0], lowest_point[1], lowest_point[2],
            eigenvectors[0, i] * axis_length,
            eigenvectors[1, i] * axis_length,
            eigenvectors[2, i] * axis_length,
            color=colors[i],
            label=f'Axis {i+1}'
        )
    
    if true_pose is not None:
        for i in range(3):
            # The length of the axes can be scaled for better visualization
            axis_length = 0.05
            ax.quiver(
                true_pose[0, 3], true_pose[1, 3], true_pose[2, 3],
                true_pose[0, i] * axis_length,
                true_pose[1, i] * axis_length,
                true_pose[2, i] * axis_length,
                color=colors[i],
                label=f'True Axis {i+1}',
                ls = '--'
            )

    # Plot the bounding box
    if bounding_box_frame is not None:
        edges = [
            (0, 1), (0, 2), (0, 4), # Edges from the first vertex
            (1, 3), (1, 5),          # Edges from the second vertex
            (2, 3), (2, 6),          # Edges from the third vertex
            (3, 7),                  # Edge from the fourth vertex
            (4, 5), (4, 6),          # Edges from the fifth vertex
            (5, 7),                  # Edge from the sixth vertex
            (6, 7)                   # Edge from the seventh vertex
        ]
        ax.scatter(bounding_box_frame[:, 0], bounding_box_frame[:, 1], bounding_box_frame[:, 2], c='b', marker='o')
        for i, j in edges:
            ax.plot(
                bounding_box_frame[[i, j], 0],
                bounding_box_frame[[i, j], 1],
                bounding_box_frame[[i, j], 2], 'k-'
            )

    if features_matrix is not None:
        # 각 circle의 중심과 반지름 시각화
        for row in features_matrix:
            center = row[:3]
            normal = row[3:6]
            radius = row[6]
            ax.scatter(*center, c='orange', s=30)
            # 원형 outline을 대략적으로 그려주기 (optional)
            circle_pts = []
            n = normal / np.linalg.norm(normal)
            # circle 평면에 수직인 벡터 하나 찾기
            u = np.cross(n, [1, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1, 0])
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
            for theta in np.linspace(0, 2*np.pi, 50):
                circle_pts.append(center + radius * (np.cos(theta)*u + np.sin(theta)*v))
            circle_pts = np.array(circle_pts)
            ax.plot(circle_pts[:, 0], circle_pts[:, 1], circle_pts[:, 2], 'orange', alpha=0.6)
            

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(Title)
    ax.legend()
    set_axes_equal(ax)
    # ax.view_init(elev=00, azim=00)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(RESULTS_DIR, f"pca_visualization_{timestamp}.png"), dpi=300)
    plt.show()