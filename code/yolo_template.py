# from multiprocessing import freeze_support
from ultralytics import YOLO
from ultralytics import RTDETR

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# model = YOLO(r'models\3.26.1\weights\best.pt')
model = RTDETR(r'models\3.28.1\weights\best.pt')

# model.train(data='coco128.yaml', epochs=1)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

# Run inference with the RT-DETR-l model on the 'bus.jpg' image
# results = model(r'D:\jingm\下载\Data\test\164215.png')
results = model.track(source=r'D:\jingm\下载\Data\test\datatest.mp4', show=True, conf=0.3, iou=0.5)