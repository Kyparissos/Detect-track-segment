import cv2
from ultralytics import YOLO

CLASS = (
    "amoeba",
    "apusomonas",
    "ciliate",
    "cysts",
    "flagellate-classical",
    "flagellate-hemi",
    "hemimastix",
    "hypotrich",
    "monothalamid",
    "nematode",
    "snail ciliate"
)

COLOR = (
    (255,191,0),
    (208,224,64),
    (0,255,127),
    (0,215,255),
    (34,34,178),
    (204,50,153),
    (255,255,187),
    (62,255,192),
    (8,101,139),
    (106,106,255),
    (155,211,255)
)

# Load YOLOv8 models
model_tracking = YOLO(r'G:\lund\master_thesis\train4\weights\best.pt')
model_classify = YOLO(r'G:\lund\master_thesis\YOLOv8\runs\classify\train\weights\best.pt')

# Open the video file
source = cv2.VideoCapture(r"H:\_videos\test3.13\2.mp4")
fps = int(source.get(cv2.CAP_PROP_FPS))
width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))

# save output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(r"H:\_videos\test3.13\2_label.mp4",
                               fourcc, fps, (width, height))

# Loop through the video frames
while source.isOpened():
    # Read a frame from the video
    success, frame = source.read()

    if success:
        # Tracking moving objects first
        results_tracking = model_tracking.track(frame, conf=0.5, persist=True)

        for i in range(0, len(results_tracking[0].boxes.xyxy)):
            # Save the coordinates of the detected bounding boxes
            bbox_coordinates = results_tracking[0].boxes.xyxy[i].tolist()
            bbox_coordinates = [int(x) for x in bbox_coordinates]

            # Crop frame for classification
            frame_crop = frame[bbox_coordinates[1]:bbox_coordinates[3],
                           bbox_coordinates[0]:bbox_coordinates[2]].copy()

            # Classification
            results_classify = model_classify(frame_crop, conf=0.5)  # predict on an image
            label = CLASS[results_classify[0].probs.top1]
            prob = round(float(results_classify[0].probs.top1conf), 2)
            whole_label = label + " " + str(prob)

            # Add label and prob to bounding box
            annotated_frame = frame
            if prob >= 0.5:
                cv2.rectangle(annotated_frame, (bbox_coordinates[0], bbox_coordinates[1]), (bbox_coordinates[2], bbox_coordinates[3]),
                              color=COLOR[results_classify[0].probs.top1], thickness=2)

                cv2.putText(
                    annotated_frame,
                    whole_label,
                    (bbox_coordinates[0], bbox_coordinates[1] - 15),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=frame.shape[1] / 1000,
                    color=COLOR[results_classify[0].probs.top1],
                    thickness=1
                )

        cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
        cv2.imshow("Tracking", frame)

        output_video.write(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
source.release()
cv2.destroyAllWindows()
