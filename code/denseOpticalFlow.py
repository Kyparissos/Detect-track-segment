import numpy as np
import cv2
import time


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    # maskx = (np.abs(fx) > 1) & (30> np.abs(fx))
    # masky = (np.abs(fy) > 1) & (30> np.abs(fy))
    fx[fx < 1] = 0
    fy[fy < 1] = 0
    lengthx =len(fx[fx.nonzero()])
    lengthy =len(fy[fy.nonzero()])
    fxa = np.sum(fx)/lengthx
    fya = np.sum(fy)/lengthy
    # most_frequent_fx = np.bincount(fx).argmax()
    # ffx = np.full_like(fx, most_frequent_fx)
    # most_frequent_fy = np.bincount(fy).argmax()
    # ffy = np.full_like(fy, most_frequent_fy)

    lines = np.vstack([x, y, x - fx + fxa, y - fy + fya]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    # maskx = (np.abs(fx) > 1) & (30> np.abs(fx))
    # masky = (np.abs(fy) > 1) & (30> np.abs(fy)) 
    # ffx = fx[maskx]
    # ffy = fy[masky]
    fxa = np.mean(fx)
    fya = np.mean(fy)
    # most_frequent_fx = np.bincount(fx).argmax()
    # ffx = np.full_like(fx, most_frequent_fx)
    # most_frequent_fy = np.bincount(fy).argmax()
    # ffy = np.full_like(fy, most_frequent_fy)
    fx = fx - fxa
    fy = fy - fya

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


cap = cv2.VideoCapture(r"D:\jingm\下载\Data\demo\demo-nematode.mp4")

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

while True:

    suc, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # start time to calculate FPS
    start = time.time()

    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    prevgray = gray

    cv2.namedWindow('flow HSV', 0)
    cv2.resizeWindow('flow HSV', 600, 600)
    cv2.namedWindow('flow', 0)
    cv2.resizeWindow('flow', 600, 600)
    cv2.imshow('flow', draw_flow(gray, flow))
    cv2.imshow('flow HSV', draw_hsv(flow))

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()