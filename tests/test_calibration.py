import pytest

from physics_motion.calibration import Calibration


def test_scale_coordinate_orientation_and_serialization():
    cal = Calibration.from_points((20, 20), (220, 20), 0.5, (640, 480))
    assert cal.pixels_per_meter == pytest.approx(400)
    assert cal.pixel_to_meters(200, 80) == pytest.approx((0.5, 1.0))
    assert cal.to_dict()["reference_length_m"] == 0.5


@pytest.mark.parametrize(
    "a,b,length,size",
    [
        ((0, 0), (0, 0), 1, (640, 480)),
        ((0, 0), (3, 2), 1, (640, 480)),
        ((0, 0), (200, 0), 0, (640, 480)),
        ((0, 0), (200, 0), -1, (640, 480)),
        ((-1, 0), (200, 0), 0.3, (640, 480)),
        ((0, 0), (700, 0), 0.3, (640, 480)),
        ((0, 0), (200, 0), 0.3, (0, 480)),
        ((0, 0), (200, 0), float("nan"), (640, 480)),
    ],
)
def test_reject_bad_calibration(a, b, length, size):
    with pytest.raises(ValueError):
        Calibration.from_points(a, b, length, size)


@pytest.mark.parametrize("minimum", [0, -1, float("nan"), float("inf"), True, "10"])
def test_invalid_reference_pixel_threshold_cannot_allow_zero_scale(minimum):
    with pytest.raises(ValueError):
        Calibration.from_points((20, 20), (20, 20), 0.5, (640, 480), minimum)


@pytest.mark.parametrize(
    "size", [(640.5, 480), (640, float("nan")), (float("inf"), 480), (True, 480), (640,), "640x480"]
)
def test_frame_dimensions_must_be_two_finite_positive_integers(size):
    with pytest.raises(ValueError):
        Calibration.from_points((0, 0), (20, 0), 0.5, size)


@pytest.mark.parametrize("point", [(0,), (0, 1, 2), (True, 0), ("0", 0), None])
def test_malformed_points_raise_a_clear_value_error(point):
    with pytest.raises(ValueError):
        Calibration.from_points(point, (200, 0), 0.5, (640, 480))


def test_direct_construction_cannot_bypass_scale_validation():
    with pytest.raises(ValueError):
        Calibration(640, 480, (0, 0), (200, 0), 0.5, 0)
    with pytest.raises(ValueError):
        Calibration(640, 480, (0, 0), (200, 0), 0.5, 500)


@pytest.mark.parametrize("length", [1e-320, 1e308, True, "0.5"])
def test_unrepresentable_or_wrongly_typed_physical_scale_is_rejected(length):
    with pytest.raises(ValueError):
        Calibration.from_points((0, 0), (200, 0), length, (640, 480))


@pytest.mark.parametrize(
    "point", [(-0.1, 0), (641, 0), (0, 481), (float("inf"), 0), (True, 0), ("0", 0)]
)
def test_pixel_conversion_rejects_invalid_or_outside_frame_coordinates(point):
    cal = Calibration.from_points((20, 20), (220, 20), 0.5, (640, 480))
    with pytest.raises(ValueError):
        cal.pixel_to_meters(*point)


def test_diagonal_reference_and_geometric_frame_edges_preserve_coordinate_orientation():
    cal = Calibration.from_points((30, 40), (60, 80), 0.5, (640, 480))
    assert cal.pixels_per_meter == pytest.approx(100)
    assert cal.pixel_to_meters(0, 480) == pytest.approx((0, 0))
    assert cal.pixel_to_meters(640, 0) == pytest.approx((6.4, 4.8))
