#!/usr/bin/env python
"""STARTER (needs hardware) - overhead OAK-D -> object detection -> pixel (bottom-center).

First perception milestone (per Brainstorm 2): detect a colored/shaped object on the
OAK-D RGB stream. Later: swap the color blob for YOLO / YOLO-World, and project the
bottom-center pixel through the camera->floor homography (M1) to get Goal{x, y}.

Run (on a machine with the OAK-D plugged in):
    uv run --with depthai --with opencv-python --with numpy python starters/oakd_detect.py

STATUS: written, NOT hardware-verified. Tune the HSV range to your object; wire the
homography to turn the pixel into a floor coordinate and publish Goal (M3a).
"""
from __future__ import annotations
import sys

# Default: detect a saturated RED object (red wraps the hue circle -> two ranges). Tune to your target.
HSV_LO_1, HSV_HI_1 = (0, 120, 70), (10, 255, 255)
HSV_LO_2, HSV_HI_2 = (170, 120, 70), (180, 255, 255)
MIN_AREA_PX = 500


def detect_color_blob(bgr):
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(HSV_LO_1), np.array(HSV_HI_1)) | \
        cv2.inRange(hsv, np.array(HSV_LO_2), np.array(HSV_HI_2))
    mask = cv2.medianBlur(mask, 5)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA_PX:
        return None
    x, y, w, h = cv2.boundingRect(c)
    return (x, y, x + w, y + h), (x + w // 2, y + h)  # bbox, bottom-center (floor contact point)


def main():
    try:
        import depthai as dai
    except ImportError:
        print("depthai missing. Run: uv run --with depthai --with opencv-python --with numpy python starters/oakd_detect.py")
        sys.exit(1)
    import cv2

    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setInterleaved(False)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam.video.link(xout.input)

    with dai.Device(pipeline) as device:
        q = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        print("OAK-D streaming. Press 'q' to quit.")
        while True:
            frame = q.get().getCvFrame()
            det = detect_color_blob(frame)
            if det is not None:
                (x0, y0, x1, y1), (cx, by) = det
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.circle(frame, (cx, by), 5, (0, 0, 255), -1)
                print(f"object bbox=({x0},{y0},{x1},{y1}) bottom-center=({cx},{by})")
                # TODO: pixel (cx, by) -> floor (X, Y) via CalibConfig.H_cam2floor (M1),
                #       then publish Goal(x=X, y=Y, label="object", t=time.time()) on the mrhack bus (M3a).
                # TODO: swap detect_color_blob for YOLO / YOLO-World (open-vocab "the wrench").
            cv2.imshow("OAK-D", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
