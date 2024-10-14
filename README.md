# yolov8-seg_lanedetection
YOLOv8-seg | Lane Detection | Easy to access

## Introduction
Topology lane recognition via Hough transform is difficult; therefore, lane recognition is achieved via accessible yolov8-segmentation.

## Prerequisites
It runs on Ubuntu 20.04, and the version below doesn't matter if you install the latest version.
```Shell
pip install -r requirements.txt
```
- torch
- scikit-learn
- opencv
- torch
- ultralytics

I recommend you to study through that blog because you also need to know information about camera calibration.
- https://darkpgmr.tistory.com/32

## Dataset
A custom dataset was constructed by fetching various topology lane images.
You can get the dataset through the code below.

```Shell
from roboflow import Roboflow
rf = Roboflow(api_key="hVnauuf3IfwkphEQsWBu")
project = rf.workspace("dddd-oew5d").project("lane-tpkna")
version = project.version(28)
dataset = version.download("yolov8")
```

## Result


## Reference
- [https://github.com/jinhoyoho](https://github.com/jinhoyoho/CLRNet_research)
- https://gaussian37.github.io/vision-concept-ipm/
