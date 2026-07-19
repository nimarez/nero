from types import SimpleNamespace

from nero.k1_preflight import CameraReadiness


def message(stamp: float, width: int = 544, height: int = 448):
    seconds = int(stamp)
    return SimpleNamespace(
        width=width,
        height=height,
        header=SimpleNamespace(
            stamp=SimpleNamespace(
                sec=seconds,
                nanosec=int(round((stamp - seconds) * 1_000_000_000)),
            )
        ),
    )


def test_camera_readiness_requires_messages_and_a_close_rgbd_pair():
    readiness = CameraReadiness(tolerance_seconds=0.02)
    readiness.observe("raw_rgb", message(1.0))
    readiness.observe("rgb", message(1.0))
    readiness.observe("depth", message(1.03))
    readiness.observe("camera_info", message(1.0))

    assert not readiness.ready
    assert "closest RGB-D offset: 30.0ms" in readiness.summary()

    for stamp in (1.01, 1.06, 1.11):
        readiness.observe("rgb", message(stamp - 0.01))
        readiness.observe("depth", message(stamp))
    assert readiness.ready
    assert "RGB rate:" in readiness.summary()


def test_camera_readiness_rejects_wrong_k1_resolution_and_explains_silent_topics():
    readiness = CameraReadiness()
    readiness.observe("rgb", message(1.0, width=640, height=480))

    assert not readiness.ready
    summary = readiness.summary({"rgb": 1, "depth": 1, "camera_info": 1, "raw_rgb": 1})
    assert "rgb is 640x480, expected maximum 544x448" in summary
    assert "depth: no messages (1 publisher(s) discovered)" in summary
