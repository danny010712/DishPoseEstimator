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

## 1031 Updates
1) capture시 필터 적용 해제(hole filling 등의 필터가 depth noise 야기 가능성)
2) pose refinement 방식 변경(xy-plane projection 후 MEC(minimum enclosing circle) 찾기)
3) dataset2 SlicedBowl 추가 -> refinement 결과 확인 가능


## Future works
0) text prompt 적용?