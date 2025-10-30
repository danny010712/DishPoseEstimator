from fastsam import FastSAM, FastSAMPrompt
import numpy as np

model = FastSAM('./weights/FastSAM-x.pt')
IMAGE_PATH = './images/dogs.jpg'
DEVICE = 'cpu'
everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.3, iou=0.4,)
prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

# everything prompt
# ann = prompt_process.everything_prompt()
# point prompt
# points default [[0,0]] [[x1,y1],[x2,y2]]
# point_label default [0] [1,0] 0:background, 1:foreground

ann = prompt_process.point_prompt(points=[[620, 360]], pointlabel=[1])
# ann = prompt_process.text_prompt(text = 'the yellow dog')


print(ann[0].shape)
print(ann[0][355:365, 615:625])

cnt = 0
for i in range(1072):
    for j in range(603):
        if ann[0][j, i] == True: cnt = cnt+1

print(f"true : {cnt}")



prompt_process.plot(annotations=ann,output_path='./output/dog.jpg',)
