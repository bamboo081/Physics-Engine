"""Marker tests use generated pixels and require no downloadable model."""

import importlib.util

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def detector_class():
    assert importlib.util.find_spec("physics_motion.detection") is not None, (
        "Offline color detector is not implemented"
    )
    from physics_motion.detection import ColorMarkerDetector

    return ColorMarkerDetector


def frame_with_marker(center=(40, 50), hue=0):
    hsv = np.zeros((120, 180, 3), np.uint8)
    cv2.circle(hsv, center, 10, (hue, 220, 230), -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_marker_requires_user_sample_then_tracks_centroid():
    detector = detector_class()()
    assert detector.detect(frame_with_marker()) == []
    detector.select_color(frame_with_marker(), 40, 50)
    found = detector.detect(frame_with_marker(center=(53, 65)))
    assert len(found) == 1
    assert found[0].center == pytest.approx((53, 65), abs=0.1)


def test_hue_wraparound_tracks_red_on_both_sides_of_hsv_zero():
    detector = detector_class()()
    detector.select_color(frame_with_marker(hue=179), 40, 50)
    found = detector.detect(frame_with_marker(center=(57, 64), hue=1))
    assert len(found) == 1
    assert found[0].center == pytest.approx((57, 64), abs=0.1)


def test_white_background_and_tiny_color_noise_do_not_become_targets():
    detector = detector_class()()
    detector.select_color(frame_with_marker(), 40, 50)
    frame = np.full((120, 180, 3), 255, np.uint8)
    frame[10, 10] = (0, 0, 255)
    assert detector.detect(frame) == []
    with pytest.raises(ValueError, match="saturat|color|bright"):
        detector.select_color(frame, 40, 50)
    assert not detector.selected


def test_multiple_matching_components_are_preserved_for_selector():
    detector = detector_class()()
    frame = frame_with_marker()
    cv2.circle(frame, (110, 50), 10, (0, 0, 240), -1)
    detector.select_color(frame, 40, 50)
    centers = [item.center for item in detector.detect(frame)]
    assert len(centers) == 2
    assert np.asarray(centers) == pytest.approx(np.asarray([(40, 50), (110, 50)]), abs=0.1)


def test_marker_uses_area_centroid_instead_of_bounding_box_midpoint():
    detector = detector_class()()
    frame = np.zeros((120, 180, 3), np.uint8)
    cv2.rectangle(frame, (20, 20), (30, 70), (0, 0, 240), -1)
    cv2.rectangle(frame, (20, 60), (70, 70), (0, 0, 240), -1)
    detector.select_color(frame, 25, 30)
    found = detector.detect(frame)
    assert len(found) == 1
    assert found[0].center[0] < 40
    assert found[0].center[1] > 50


def test_selection_outside_frame_fails_clearly():
    detector = detector_class()()
    with pytest.raises(ValueError, match="frame"):
        detector.select_color(frame_with_marker(), 200, 50)


def test_missing_yolo_dependency_recommends_the_real_optional_extra(monkeypatch):
    import builtins

    from physics_motion.detection import YoloDetector

    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "ultralytics":
            raise ModuleNotFoundError("No module named ultralytics")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(RuntimeError, match="desktop,detector"):
        YoloDetector("unused-model.pt", "cpu", 0.35)
