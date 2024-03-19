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

CONF_SIZE = 12


def update_list(lst, nbr):
    lst = lst[1:]
    lst.append(nbr)
    return lst


def update_dict(dict, id, label, bbox, flag):
    flag[id] = True
    if id not in dict:
        dict[id] = [[label], [bbox]]
    elif len(dict[id][0]) < CONF_SIZE:
        dict[id][0].append(label)
        dict[id][1].append(bbox)
    else:
        del dict[id][0][0]
        dict[id][0].append(label)
        del dict[id][1][0]
        dict[id][1].append(bbox)
    return dict, flag

def draw_bbox(frame, label, bbox_coordinates):
    annotated_frame = frame
    label_index = CLASS.index(label)
    cv2.rectangle(annotated_frame, (bbox_coordinates[0], bbox_coordinates[1]),
                  (bbox_coordinates[2], bbox_coordinates[3]),
                  color=COLOR[label_index], thickness=2)

    cv2.putText(
        annotated_frame,
        label,
        (bbox_coordinates[0], bbox_coordinates[1] - 15),
        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
        fontScale=frame.shape[1] / 1000,
        color=COLOR[label_index],
        thickness=1
    )
    return annotated_frame


# Load YOLOv8 models
model_tracking = YOLO(r'G:\lund\master_thesis\train3\weights\best.pt')
model_classify = YOLO(r'G:\lund\master_thesis\train2\weights\best.pt')

# Open the video file
source = cv2.VideoCapture(r"H:\_videos\test3.13\1.mp4")
fps = int(source.get(cv2.CAP_PROP_FPS))
width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))

# save output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(r"H:\_videos\test3.13\1_label.mp4",
                               fourcc, fps, (width, height))

# Loop through the video frames
frames = [0] * CONF_SIZE
successes = [0] * CONF_SIZE
bbox_coordinates = [0] * CONF_SIZE
dicts = {"-1": [["init"], ["init"]]}
flags = [False] * 20
while source.isOpened():
    # Read a frame from the video
    now_fps = source.get(1)
    if now_fps < CONF_SIZE:
       for n in range(0, CONF_SIZE):
            success0, frame0 = source.read()
            frames = update_list(frames, frame0)
            successes = update_list(successes, success0)

            if successes[n]:
                results_tracking = model_tracking.track(frames[n], conf=0.3, persist=True)

                if results_tracking[0].boxes.id != None:
                    ids = results_tracking[0].boxes.id.tolist()
                    for i in range(0, len(ids)):
                        # Save the coordinates of the detected bounding boxes
                        bbox_coordinates = results_tracking[0].boxes.xyxy[i].tolist()
                        bbox_coordinates = [int(x) for x in bbox_coordinates]

                        # Crop frame for classification
                        frame_crop = frames[n][bbox_coordinates[1]:bbox_coordinates[3],
                                     bbox_coordinates[0]:bbox_coordinates[2]].copy()

                        # Classification
                        results_classify = model_classify(frame_crop, conf=0.5)  # predict on an image
                        label = CLASS[results_classify[0].probs.top1]
                        # prob = round(float(results_classify[0].probs.top1conf), 2)
                        # whole_label = label + " " + str(prob)

                        dicts, flags = update_dict(dicts, int(ids[i]), label, bbox_coordinates, flags)
                        if "-1" in dicts:
                            del dicts["-1"]

    else:
        success1, frame1 = source.read()
        frames = update_list(frames, frame1)
        successes = update_list(successes, success1)

        if successes[-1]:
            # Tracking moving objects first
            results_tracking = model_tracking.track(frames[-1], conf=0.3, persist=True)

            if results_tracking[0].boxes.id != None:
                ids = results_tracking[0].boxes.id.tolist()
                for i in range(0, len(ids)):
                    # Save the coordinates of the detected bounding boxes
                    bbox_coordinates = results_tracking[0].boxes.xyxy[i].tolist()
                    bbox_coordinates = [int(x) for x in bbox_coordinates]

                    # Crop frame for classification
                    frame_crop = frames[-1][bbox_coordinates[1]:bbox_coordinates[3],
                                 bbox_coordinates[0]:bbox_coordinates[2]].copy()

                    # Classification
                    results_classify = model_classify(frame_crop, conf=0.5)  # predict on an image
                    label = CLASS[results_classify[0].probs.top1]
                    # prob = round(float(results_classify[0].probs.top1conf), 2)
                    # whole_label = label + " " + str(prob)

                    dicts, flags = update_dict(dicts, int(ids[i]), label, bbox_coordinates, flags)
        else:
            # Break the loop if the end of the video is reached
            break

            # Find the most likely class
    keys = list(dicts.keys())
    for i in range(0, len(keys)):
        if flags[keys[i]] == True:
            label_list = dicts[keys[i]][0]
            bbox_list = dicts[keys[i]][1]
            max_label = max(label_list, key=label_list.count)
            annotated_frame = draw_bbox(frames[0], max_label, bbox_list[0])
            flags[keys[i]] = False

    cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
    cv2.imshow("Tracking", annotated_frame)

    output_video.write(annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
source.release()
cv2.destroyAllWindows()
