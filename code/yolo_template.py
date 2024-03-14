# from multiprocessing import freeze_support
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)


# from roboflow import Roboflow
# rf = Roboflow(api_key="Xe5DAMUxmRkVoMxy59e8")
# project = rf.workspace("protists").project("protists")
# dataset = project.version(3).download("yolov8")


if __name__ ==  '__main__':
    # freeze_support()
    # Use the model
    model.train(data='coco128.yaml', epochs=1)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # path = model.export(format="onnx")  # export the model to ONNX format