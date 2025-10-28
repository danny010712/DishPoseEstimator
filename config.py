import os

# === Base Paths ===
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_PATH, "results")
DATA_DIR = os.path.join(BASE_PATH, "data")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
SAVE_INTERMEDIATE = True # whether to save intermediate results

# === Capture Parameters ===
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
DEPTH_SCALE_FACTOR = 1000.0 
DEPTH_TRUNCATION = 1.5
SCENE_NUM = 1 # number of scenes to use for merging

# === Segmentation ===
CROP_THRESHOLD = 1.3 # Crop the point cloud, centered with the gripper end point within this radius.
EPS = 0.01 # radius of the neighborhood in DBSCAN 0.015
MIN_POINTS = 100 # minimum number of points to create cluster
# COLOR_WEIGHT = 0.0 # 0.5
# COLOR_REF = [1, 1, 1] # 0,0,0
# COLOR_THRESHOLD = 0.0  # for dark filtering 0.14~0.18 # std 0.029

MODEL_TYPE = 'FastSAM-x'
CONF = 0.1 # FastSAM parameters
IOU = 0.2 # FastSAM parameters

# === Registration ===
VOXEL_SIZE = 0.005
MAX_ITER_RANSAC = 200
THRESHOLD_RANSAC = 0.005 # For determining inliers

# === Pose Estimation ===
DISTANCE_THRESHOLD = 0.005


