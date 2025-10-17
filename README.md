git clone https://github.com/danny010712/DishPoseEstimator.git

cd DishPoseEstimator

conda create -n myenv python=3.9

conda activate myenv

pip install -r requirements.txt

### How to Use ###
- Currently works on python=3.9
- Paths recommended not to include Korean(for now)
- Edit parameters in 'config.py' and save before running codes

# run for pose estimation
python main.py
-> visualizes intermediate results including cropping, clustering and finding optimal oriented bounding box(OBB) & final refined pose

# Able to run "capture.py" for checking quality of raw point clouds
python -m inputoutput.capture
-> Can check how the raw files are made

# Required data in folder "data", Results in folder "results"
raw point cloud files are saved in:
results/raw_{timestamp}.ply

FastSAM result visualizing image is saved in:
results/FastSAMresult_{timestamp}.png

color, depth map are also saved in:
results/color_{timestamp}.png
results/depth_map_{timestamp}.png

Segmented point cloud right after applying FastSAM is saved in:
results/fastsamsegmented_{i}.ply
