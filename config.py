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
DEPTH_TRUNCATION = 1.5 # ignore points further than 1.5 m from the camera.
SCENE_NUM = 1 # number of scenes to use for merging

# === Segmentation ===
CROP_THRESHOLD = 1.0 # Crop the point cloud, centered with the gripper end point within this radius.
EPS = 0.015 # radius of the neighborhood in DBSCAN 0.015
MIN_POINTS = 100 # minimum number of points to create cluster

MODEL_TYPE = 'FastSAM-x'
CONF = 0.1 # FastSAM parameters
IOU = 0.2 # FastSAM parameters

# === Registration ===
VOXEL_SIZE = 0.005
MAX_ITER_RANSAC = 200
THRESHOLD_RANSAC = 0.001 # For determining inliers

# === Pose Estimation ===
DISTANCE_THRESHOLD = 0.005
# AprilTag
TAG_SIZE = 0.032
TAG_FAMILY = 'tag36h11'

ESTIMATION_MODE = 'camera' # choose from 'world', 'camera', 'gripper'