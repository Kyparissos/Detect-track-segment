from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO('yolov8n-pose.yaml')  # build a new model from YAML
    model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)
    model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data='coco8-pose.yaml', epochs=100, imgsz=640)