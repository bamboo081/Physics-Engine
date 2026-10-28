"""Identity association regressions; no camera, network, or neural detector."""

import math

import pytest

from physics_motion.tracking import Detection, TrackSelector


def d(x, y, conf=0.95):
    return Detection((x - 5, y - 5, x + 5, y + 5), conf)


def test_single_target_nearest_neighbor_not_first_result():
    selector = TrackSelector(max_speed_px_s=150)
    assert selector.choose([d(10, 10)], 1.0)[0].center == (10, 10)
    chosen, _ = selector.choose([d(200, 200), d(14, 10)], 1.1)
    assert chosen.center == (14, 10)
    assert selector.choose([d(500, 500)], 1.2)[0] is None


def test_ambiguous_requires_click_and_lost_requires_reselection():
    selector = TrackSelector(max_gap_s=0.5)
    chosen, status = selector.choose([d(20, 20), d(90, 20)], 0.1)
    assert chosen is None and "click" in status.lower()
    selector.request_selection(20, 20)
    assert selector.choose([d(20, 20), d(90, 20)], 0.2)[0].center == (20, 20)
    selector.choose([], 1.0)
    assert selector.choose([d(21, 20)], 1.1)[0] is None
    selector.request_selection(21, 20)
    assert selector.choose([d(21, 20)], 1.2)[0] is not None


def test_missed_click_does_not_auto_select_different_object_next_frame():
    selector = TrackSelector()
    selector.request_selection(400, 400)
    chosen, status = selector.choose([d(20, 20)], 0.1)
    assert chosen is None and "missed" in status.lower()
    assert selector.choose([d(20, 20)], 0.2)[0] is None


@pytest.mark.parametrize(
    "box,confidence",
    [
        ((0, 0, math.nan, 10), 0.9),
        ((0, 0, math.inf, 10), 0.9),
        ((10, 0, 0, 10), 0.9),
        ((0, 0, 0, 10), 0.9),
        ((0, 0, 10, 10), math.nan),
        ((0, 0, 10, 10), 1.1),
    ],
)
def test_invalid_detection_cannot_enter_association(box, confidence):
    with pytest.raises(ValueError):
        Detection(box, confidence)


@pytest.mark.parametrize(
    "speed,gap", [(math.nan, 0.5), (math.inf, 0.5), (100, math.nan), (100, math.inf)]
)
def test_nonfinite_association_limits_are_rejected(speed, gap):
    with pytest.raises(ValueError):
        TrackSelector(max_speed_px_s=speed, max_gap_s=gap)


def test_overlapping_clicked_objects_do_not_choose_by_confidence():
    selector = TrackSelector()
    selector.request_selection(20, 20)
    chosen, status = selector.choose([d(19, 20, 0.6), d(21, 20, 0.99)], 0)
    assert chosen is None
    assert "multiple" in status.lower() or "overlap" in status.lower()


def test_near_equal_candidates_suspend_identity_until_manual_selection():
    selector = TrackSelector(max_speed_px_s=200)
    selector.choose([d(20, 20)], 0)
    chosen, status = selector.choose([d(28, 20), d(30, 20)], 0.1)
    assert chosen is None
    assert "click" in status.lower()
    assert selector.choose([d(30, 20)], 0.2)[0] is None
    selector.request_selection(30, 20)
    assert selector.choose([d(30, 20)], 0.3)[0].center == (30, 20)


def test_reselection_and_missing_observation_open_new_track_ids():
    selector = TrackSelector()
    selector.choose([d(20, 20)], 0)
    assert selector.track_id == 1
    selector.choose([d(21, 20)], 0.1)
    assert selector.track_id == 1
    selector.choose([], 0.2)
    selector.choose([d(23, 20)], 0.3)
    assert selector.track_id == 2
    selector.request_selection(24, 20)
    selector.choose([d(24, 20)], 0.4)
    assert selector.track_id == 3


def test_centroid_override_is_used_for_asymmetric_color_marker():
    observation = Detection((10, 20, 30, 40), 1, centroid=(16.5, 29.25))
    assert observation.center == (16.5, 29.25)
    with pytest.raises(ValueError):
        Detection((10, 20, 30, 40), 1, centroid=(31, 25))


def test_nonmonotonic_time_never_updates_association():
    selector = TrackSelector()
    selector.choose([d(20, 20)], 1)
    assert selector.choose([d(21, 20)], 1)[0] is None
    assert selector.last_time == 1


def test_rejected_distant_target_requires_selection_even_as_time_gate_expands():
    selector = TrackSelector()
    original = Detection((15, 15, 35, 35), 0.9)
    stranger = Detection((215, 15, 235, 35), 0.95)
    assert selector.choose([original], 0.0)[0] is original
    for timestamp in (0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.4):
        chosen, _ = selector.choose([stranger], timestamp)
        assert chosen is None, "A rejected jump must not become an automatic target switch"
    selector.request_selection(225, 25)
    assert selector.choose([stranger], 0.44)[0] is stranger
