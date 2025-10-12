import os

# === Base Paths ===
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_PATH, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Capture Parameters ===
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
DEPTH_SCALE_FACTOR = 1000.0 
DEPTH_TRUNCATION = 1.5
SCENE_NUM = 1 # number of scenes to use for merging

# === Segmentation ===
CROP_THRESHOLD = 0.2 # Crop the point cloud, centered with the gripper end point within this radius.
EPS = 0.015 # radius of the neighborhood in DBSCAN 0.015
MIN_POINTS = 25 # minimum number of points to create cluster
COLOR_WEIGHT = 0.5 # 0.5
COLOR_REF = [1, 1, 1] # 0,0,0
COLOR_THRESHOLD = 0.029  # for dark filtering 0.14~0.18

# === Registration ===
VOXEL_SIZE = 0.005
MAX_ITER_RANSAC = 100
THRESHOLD_RANSAC = 0.01 # For determining inliers

# === Pose Estimation ===
DISTANCE_THRESHOLD = 0.005
SAVE_INTERMEDIATE = True
