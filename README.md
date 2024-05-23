# About

This repo contains Python scripts for video annotating and annotation evaluation.

# How to use

## Annotator
1. Download the selected version of the pre-trained YOLOv8 model and tracker models (only for vit and nano usage).
2. Place the models in the repo root directory.
3. Open the command line and run:
```
./video_annotator.py --input_path <video path> --detector <selected version of pre-trained YOLOv8 model> --tracker <selected traker>
```
4. For multiple object tracking, add `--mot` argument.

## Evaluator
1. Open the command line and run:
```
./evaluator.py --predicted <generated annotation> --groundtruth <groundthruth annotation> --dataset <original dataset>
```

# Sources
1. Pre-trained YOLOv8 models: https://docs.ultralytics.com/datasets/detect/coco/
2. Nano model: https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack/models/nanotrackv3
3. Vit model: https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_vittrack
4. GOT-10K dataset: http://got-10k.aitestunion.com/index
5. GMOT-40 dataset: https://spritea.github.io/GMOT40/
