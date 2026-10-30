# Pretrained weights

YOLOX checkpoints (`*.pth`) that a training run can start from. The Train dialog lists every
file here under "Start from weights"; the v4 recipe starts from the COCO `yolox_m.pth`.

Checkpoints are gitignored. Download the stock ones from the YOLOX releases
(https://github.com/Megvii-BaseDetection/YOLOX/releases) or link an existing file:

    ln -s /mnt/hdd/rabeetakbar/rabeet/SLM/annotations/SLMmodelv2/models/yolox_m.pth projects/weights/
