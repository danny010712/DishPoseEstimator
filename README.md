# Setup
git clone https://github.com/danny010712/DishPoseEstimator.git

cd DishPoseEstimator

conda create -n myenv python=3.9

conda activate myenv

pip install -r requirements.txt

- Connect RealSense
- Currently works on python=3.9
- Paths recommended not to include Korean(for now)
- Edit parameters in 'config.py' and save before running codes
    - Try to edit FastSAM parameters, estimation mode, AprilTag features
- If 'Could not find module apriltag.dll' error occurs, try to copy pthreadVC2.dll into the same env folder with apriltag.dll
    - Due to pupil_pthreads_win install, pthreadVC2.dll might be in ```...\anaconda3\envs\YOUR_ENV_NAME\Lib\site-packages\pupil_pthreads_win\pthreadVC2.dll```
    - copy this to ```...\anaconda3\envs\YOUR_ENV_NAME\Lib\site-packages\pupil_apriltags\lib```


# How to Use
## Quick pose estimation
```python main.py```

visualizes intermediate results including cropping, clustering and finding optimal oriented bounding box(OBB) & final refined pose

prints estimated pose and error

## Quick RealSense capture
```python -m inputoutput.capture```

Able to check quality of raw point clouds

-> Can check how the raw files are made

## Datasets and results folder

- Datasets in folder "data"
- Results in folder "results"

- raw point cloud files are saved in:
results/raw_{timestamp}.ply

- FastSAM result visualizing image is saved in:
results/FastSAMresult_{timestamp}.png

- color, depth map are also saved in:
results/color_{timestamp}.png
results/depth_map_{timestamp}.png

- Segmented point cloud right after applying FastSAM is saved in:
results/fastsamsegmented_{i}.ply

# Updates
## 1031 Updates
1) capture시 필터 적용 해제(hole filling 등의 필터가 depth noise 야기 가능성)
2) pose refinement 방식 변경(xy-plane projection 후 MEC(minimum enclosing circle) 찾기)
3) dataset2 SlicedBowl 추가 -> refinement 결과 확인 가능

## 1112 Updates
1) AprilTag detection -> true pose

## 1126 Updating
1) Calculate T_camera in world frame using known-pose AprilTags
2) Depth smoothing(?) using multi-view frames

## Future works
0) text prompt 적용?
1) free space constraint
2) Bounding Box fitting regularization?