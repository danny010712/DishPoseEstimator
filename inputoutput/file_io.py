import os
import open3d as o3d
from config import RESULTS_DIR, SAVE_INTERMEDIATE

def save_pointcloud(pcd, name):
    if not SAVE_INTERMEDIATE:
        return
    file_path = os.path.join(RESULTS_DIR, f"{name}.ply")
    o3d.io.write_point_cloud(file_path, pcd)
    print(f"[Saved] {file_path}")

def load_pointcloud(path):
    return o3d.io.read_point_cloud(path)