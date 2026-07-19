# VERBATIM working box helpers. Run on the box with ~/Prismos-x/venv/bin/python.
import pyrealsense2 as rs, numpy as np, cv2, subprocess, os, json
ENV={**os.environ,"WAYLAND_DISPLAY":"wayland-1","XDG_RUNTIME_DIR":"/run/user/1000"}
DICT=cv2.aruco.DICT_4X4_50; PROJ_W,PROJ_H=1920,1080
def capture_color():
    pl=rs.pipeline(); c=rs.config(); c.enable_stream(rs.stream.color,1280,720,rs.format.bgr8,30); pl.start(c)
    for _ in range(15): fr=pl.wait_for_frames()
    img=np.asanyarray(fr.get_color_frame().get_data()).copy(); pl.stop(); return img
def detect_tags(img):
    d=cv2.aruco.getPredefinedDictionary(DICT); g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    try: cs,ids,_=cv2.aruco.ArucoDetector(d).detectMarkers(g)
    except AttributeError: cs,ids,_=cv2.aruco.detectMarkers(g,d)
    return {int(i):c[0] for i,c in zip(ids.flatten(),cs)} if ids is not None else {}  # id -> 4x2 (TL,TR,BR,BL)
def project_png(bgr):
    cv2.imwrite('/tmp/_true.png',bgr)
    subprocess.run(["swaymsg","output","HDMI-A-1","bg","/tmp/_true.png","fill"],env=ENV,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
def load_handles():
    return json.load(open('/tmp/procam_calib.json'))["handles"]  # 4 [x,y] projector px, dragged onto the 4 tags
