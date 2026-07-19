# ArUco object navigation

ArUco is a lightweight deterministic alternative to YOLO for objects that can
carry a printed marker. It uses the K1 Geek's existing RGB, registered depth,
and camera calibration streams; there are no camera or target-distance flags.

Copy `config/aruco_markers.example.json` and map each printed marker ID to the
name a person will type or speak. Names must be unique. Then run:

```bash
uv run nero-orb-slam --no-display \
  --object-backend aruco \
  --aruco-map config/aruco_markers.json
```

The equivalent environment configuration is:

```bash
export NERO_OBJECT_BACKEND=aruco
export NERO_ARUCO_MAP=config/aruco_markers.json
export NERO_ARUCO_DICTIONARY=DICT_4X4_50
uv run nero-orb-slam --no-display
```

The dictionary defaults to OpenCV `DICT_4X4_50`. Use matching dictionary and
marker IDs when generating/printing markers. A command such as `go to the green
cup` is accepted only if `green cup` exists in the mapping. Detected marker
corners become the standard Nero bounding box, while registered K1 depth and
`CameraInfo` provide its camera-frame 3D position and distance. Consequently,
the existing `/nero/navigation/detections` topic and Rerun RGB overlay work
without an ArUco-specific bridge.

If the marker is not detected before the policy's search limit, the robot stops,
announces that it could not detect the requested object exactly once, and waits
for another command.
