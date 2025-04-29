from ultralytics import YOLO

# Load a COCO-pretrained YOLO12n model
model = YOLO("yolo12n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
import os
from os.path import join

data_path = join(os.environ["ROBOFLOW_DATASET"], "Chess Pieces.v24-416x416_aug.yolov12/data.yaml")

results = model.train(data=data_path, epochs=100, imgsz=640)
