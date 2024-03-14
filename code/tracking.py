import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO(r'G:\lund\master_thesis\YOLOv8\best.pt')

# Open the video file
source = cv2.VideoCapture(r"H:\_videos\FrancoisINPUT\captured1908.mp4")
# fps = int(source.get(cv2.CAP_PROP_FPS))
width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(r"H:\_videos\FrancoisINPUT\captured1908_label.mp4", fourcc, 20.0, (width, height))

# Loop through the video frames
while source.isOpened():
    # Read a frame from the video
    success, frame = source.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, conf=0.55, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        output_video.write(annotated_frame)

        # Display the annotated frame
        cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
        cv2.imshow("Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
source.release()
cv2.destroyAllWindows()