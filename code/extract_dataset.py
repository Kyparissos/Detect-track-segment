# from ultralytics import YOLO
# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("path/to/last.pt")  # load a pretrained model (recommended for training)

# protists-v5
# from roboflow import Roboflow
# rf = Roboflow(api_key="Xe5DAMUxmRkVoMxy59e8")
# project = rf.workspace("protists").project("protists")
# dataset = project.version(5).download("yolov8")

# Thesis-1
# from roboflow import Roboflow
# rf = Roboflow(api_key="ljL3g7pxndRkadlrg0l3")
# project = rf.workspace("thesis-m9brm").project("thesis-e8upp")
# dataset = project.version(1).download("yolov8")

# without augmentation
# from roboflow import Roboflow
# rf = Roboflow(api_key="ljL3g7pxndRkadlrg0l3")
# project = rf.workspace("thesis-m9brm").project("protists-seg")
# version = project.version(1)
# version.deploy("yolov8", "C:/Users/jingm/Desktop/Yolo/train2/")

# from roboflow import Roboflow
# rf = Roboflow(api_key="ljL3g7pxndRkadlrg0l3")
# project = rf.workspace("thesis-m9brm").project("together-2")
# version = project.version(1)
# version.deploy("yolov8", "C:/Users/jingm/Desktop/Yolo/rightlabels_85mAP/")

# from roboflow import Roboflow
# rf = Roboflow(api_key="ljL3g7pxndRkadlrg0l3")
# project = rf.workspace("thesis-m9brm").project("test-uyxad")
# version = project.version(3)
# dataset = version.download("yolov8")

from roboflow import Roboflow
rf = Roboflow(api_key="Jxbh0mAHd0jlhrllGGkf")
project = rf.workspace("nepri").project("classification-t5fcm")
version = project.version(1)
dataset = version.download("folder")
