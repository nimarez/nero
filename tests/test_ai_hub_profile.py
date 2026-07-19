from nero.perception.ai_hub_profile import summarize_profile
from nero.perception.yolo_world_export import _validate_imgsz


def test_summarize_ai_hub_profile():
    profile = {
        "execution_summary": {
            "estimated_inference_time": 1495,
            "estimated_inference_peak_memory": 83132416,
            "first_load_time": 1170893,
            "warm_load_time": 357277,
        },
        "execution_detail": [
            {"compute_unit": "NPU"},
            {"compute_unit": "NPU"},
        ],
    }

    assert summarize_profile(profile) == {
        "inference_time_us": 1495,
        "inference_peak_memory_bytes": 83132416,
        "first_load_time_us": 1170893,
        "warm_load_time_us": 357277,
        "compute_units": ["NPU"],
    }


def test_yolo_world_export_size_validation():
    _validate_imgsz(256)


def test_yolo_world_export_rejects_invalid_size():
    import pytest

    with pytest.raises(ValueError, match="at least 256 and divisible by 32"):
        _validate_imgsz(224)
    with pytest.raises(ValueError, match="at least 256 and divisible by 32"):
        _validate_imgsz(300)
