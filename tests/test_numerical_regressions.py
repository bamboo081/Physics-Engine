"""Ground-truth regressions for the public numerical analysis contract."""

import numpy as np
import pandas as pd
import pytest

from physics_motion.engine import AnalysisSettings, analyze_samples


def frame(t, x, y, **metadata):
    return pd.DataFrame({"Time": t, "X": x, "Y": y, **metadata})


def test_general_motion_recovers_horizontal_and_vertical_acceleration():
    t = np.linspace(0, 1, 51)
    result = analyze_samples(frame(t, 1 + 2 * t + 2 * t * t, 2 - t - 1.5 * t * t))
    np.testing.assert_allclose(result.data.Vx_model, 2 + 4 * t, atol=1e-10)
    np.testing.assert_allclose(result.data.Vy_model, -1 - 3 * t, atol=1e-10)
    np.testing.assert_allclose(result.data.Ax_model, 4, atol=1e-9)
    np.testing.assert_allclose(result.data.Ay_model, -3, atol=1e-9)
    np.testing.assert_allclose(result.data.X_model, result.data.X, atol=1e-12)
    assert result.stats["gravity_estimate_m_s2"] is None


def test_local_cubic_derivatives_use_irregular_timestamps_including_endpoints():
    t = np.r_[0.0, np.sort(np.random.default_rng(3).uniform(0.01, 0.99, 59)), 1.0]
    result = analyze_samples(
        frame(
            t,
            0.3 + 2 * t + 0.7 * t * t - 0.4 * t**3,
            1 - 0.3 * t + 0.2 * t * t + 0.6 * t**3,
        )
    )
    np.testing.assert_allclose(result.data.Vx_model, 2 + 1.4 * t - 1.2 * t * t, atol=1e-10)
    np.testing.assert_allclose(result.data.Vy_model, -0.3 + 0.4 * t + 1.8 * t * t, atol=1e-10)
    np.testing.assert_allclose(result.data.Ax_model, 1.4 - 2.4 * t, atol=1e-8)
    np.testing.assert_allclose(result.data.Ay_model, 0.4 + 3.6 * t, atol=1e-8)


def test_general_curved_motion_is_not_reduced_to_one_global_parabola():
    t = np.linspace(0, 4, 401)
    result = analyze_samples(frame(t, np.sin(2 * t), np.cos(1.3 * t)))
    np.testing.assert_allclose(result.data.Vx_model, 2 * np.cos(2 * t), atol=0.001)
    np.testing.assert_allclose(result.data.Ax_model, -4 * np.sin(2 * t), atol=0.02)
    np.testing.assert_allclose(result.data.Ay_model, -1.69 * np.cos(1.3 * t), atol=0.01)
    assert len(result.segments) == 1


@pytest.mark.parametrize("mode", ["General motion", "Horizontal toss"])
def test_large_absolute_time_does_not_change_physics(mode):
    t = 1_700_000_000.0 + np.linspace(0, 0.6, 61)
    elapsed = t - t[0]  # the actual representable input times
    result = analyze_samples(
        frame(t, 0.3 + 2 * elapsed, 2 + 1.2 * elapsed - 4.905 * elapsed**2),
        AnalysisSettings(experiment_type=mode),
    )
    np.testing.assert_allclose(result.data.Vx_model, 2, atol=1e-8)
    np.testing.assert_allclose(result.data.Vy_model, 1.2 - 9.81 * elapsed, atol=1e-8)
    np.testing.assert_allclose(result.data.Ay_model, -9.81, atol=1e-7)
    if mode == "Horizontal toss":
        assert result.stats["gravity_estimate_m_s2"] == pytest.approx(9.81, abs=1e-8)
        seg = result.segments[0]
        px, py = seg.evaluate_position(t)
        np.testing.assert_allclose(px, result.data.X, atol=1e-12)
        np.testing.assert_allclose(py, result.data.Y, atol=1e-12)
        assert seg.to_dict()["time_origin_s"] == t[0]


def test_gap_cannot_be_reinterpreted_as_a_bounce_even_with_adjacent_dataframe_rows():
    first = np.linspace(0, 0.4, 41)
    second = np.linspace(2, 2.4, 41)
    t = np.r_[first, second]
    y = np.r_[2 - 2 * first - 4.905 * first**2, 1 + (second - 2) - 4.905 * (second - 2) ** 2]
    result = analyze_samples(
        frame(t, 0.2 * t, y),
        AnalysisSettings(experiment_type="Free-fall / bounce", max_gap_seconds=0.5),
    )
    assert len(result.segments) == 2
    assert result.stats["restitution_estimate"] is None
    assert result.segments[1].boundary_reason == "time_gap"


def test_track_identity_break_prevents_smoothing_across_a_small_gap_or_reselection():
    t = np.linspace(0, 0.39, 40)
    x = np.r_[t[:20], 10 + 2 * t[20:]]
    source = frame(t, x, np.ones_like(t), TrackID=[1] * 20 + [2] * 20)
    result = analyze_samples(source)
    assert len(result.segments) == 2
    np.testing.assert_allclose(result.data.Vx_model[:20], 1, atol=1e-10)
    np.testing.assert_allclose(result.data.Vx_model[20:], 2, atol=1e-10)
    assert result.segments[1].boundary_reason == "track_change"
    assert result.stats["restitution_estimate"] is None


@pytest.mark.parametrize("mode", ["General motion", "Lift and lower", "Custom experiment"])
def test_a_smooth_direction_reversal_is_not_an_impact_in_general_modes(mode):
    t = np.linspace(0, 1, 101)
    result = analyze_samples(
        frame(t, t, 10 * (t - 0.5) ** 2), AnalysisSettings(experiment_type=mode)
    )
    assert len(result.segments) == 1
    np.testing.assert_allclose(result.data.Ay_model, 20, atol=1e-8)
    assert result.stats["restitution_estimate"] is None


@pytest.mark.parametrize("impact", [0.4, 0.405])
def test_ballistic_bounce_recovers_restitution_at_an_observed_or_between_frame_impact(impact):
    t = np.linspace(0, 0.8, 81)
    u = t - impact
    y = 2 + np.where(u < 0, -2 * u, 1.4 * u) - 4.905 * u * u
    result = analyze_samples(
        frame(t, 0.2 * t, y), AnalysisSettings(experiment_type="Free-fall / bounce")
    )
    assert len(result.segments) == 2
    assert result.stats["gravity_estimate_m_s2"] == pytest.approx(9.81, abs=1e-8)
    assert result.stats["restitution_estimate"] == pytest.approx(0.7, abs=1e-8)
    assert result.stats["restitution_count"] == 1
    assert result.stats["restitution_events"][0]["impact_time_s"] == pytest.approx(impact, abs=1e-8)


def test_discontinuous_height_jump_cannot_yield_a_restitution_estimate():
    t = np.linspace(0, 0.8, 81)
    u = t - 0.4
    y = np.where(u < 0, 2 - 2 * u, 2.2 + 1.4 * u) - 4.905 * u * u
    result = analyze_samples(
        frame(t, 0.2 * t, y), AnalysisSettings(experiment_type="Free-fall / bounce")
    )
    assert result.stats["restitution_estimate"] is None


def test_horizontal_acceleration_is_measured_and_disqualifies_gravity_only_motion():
    t = np.linspace(0, 0.6, 61)
    result = analyze_samples(
        frame(t, 2 * t + t * t, 2 + 1.2 * t - 4.905 * t * t),
        AnalysisSettings(experiment_type="Horizontal toss"),
    )
    np.testing.assert_allclose(result.data.Ax_model, 2, atol=1e-10)
    assert result.stats["gravity_estimate_m_s2"] is None


def test_high_position_r_squared_without_resolved_curvature_does_not_establish_gravity():
    t = np.linspace(0, 0.1, 61)
    y = 2 + 100 * t - 4.905 * t * t + 0.05 * np.sin(2 * np.pi * t / 0.02)
    result = analyze_samples(
        frame(t, 2 * t, y), AnalysisSettings(experiment_type="Horizontal toss")
    )
    assert result.segments[0].r2_y > 0.99
    assert result.stats["gravity_estimate_m_s2"] is None


def test_systematic_cubic_acceleration_drift_disqualifies_a_high_r_squared_ballistic_fit():
    t = np.linspace(0, 0.6, 61)
    y = 2 + 10 * t - 4.905 * t * t + 2 * t**3
    result = analyze_samples(
        frame(t, 2 * t, y), AnalysisSettings(experiment_type="Horizontal toss")
    )
    assert result.segments[0].r2_y > 0.99
    assert result.stats["gravity_estimate_m_s2"] is None


def test_horizontal_cubic_drift_cannot_hide_behind_zero_mean_horizontal_acceleration():
    t = np.linspace(0, 0.6, 61)
    result = analyze_samples(
        frame(t, 2 * t + (t - 0.3) ** 3, 2 + 1.2 * t - 4.905 * t * t),
        AnalysisSettings(experiment_type="Horizontal toss"),
    )
    assert result.segments[0].r2_x > 0.99
    assert result.stats["gravity_estimate_m_s2"] is None


def test_fit_standard_error_matches_closed_form_quadratic_regression():
    t = np.arange(12) / 10
    residual = np.r_[0.01, -0.03, 0.03, -0.01, np.zeros(8)]
    result = analyze_samples(
        frame(t, 3 * t, 2 + 1.2 * t - 4.905 * t * t + residual),
        AnalysisSettings(experiment_type="Horizontal toss"),
    )
    # The residual vector is orthogonal to 1,t,t². For these centered times,
    # sum(tau²)=1.43, sum(tau⁴)=0.303875; SSE=0.002 with 12-3=9 degrees of freedom.
    expected = 2 * np.sqrt((0.002 / 9) / (0.303875 - 1.43**2 / 12))
    assert result.stats["gravity_estimate_m_s2"] == pytest.approx(9.81, abs=1e-10)
    assert result.stats["gravity_fit_standard_error_m_s2"] == pytest.approx(expected, rel=1e-10)
    np.testing.assert_allclose(result.data.Ay_fit_se, expected, rtol=1e-10)
    assert result.stats["gravity_estimate_std_m_s2"] is None
    assert any("measurement uncertainty" in warning for warning in result.warnings)


def test_short_intervals_keep_raw_positions_without_fabricated_derivatives():
    t = np.r_[0.0, 0.1, 0.2, np.linspace(2, 2.3, 16)]
    source = frame(t, t, 1 + t, Note=["measured"] * len(t))
    before = source.copy(deep=True)
    result = analyze_samples(source, AnalysisSettings(max_gap_seconds=0.5))
    pd.testing.assert_frame_equal(source, before)
    pd.testing.assert_frame_equal(result.raw_samples, before)
    assert len(result.data) == len(source)
    assert result.data.SegmentID[:3].isna().all()
    assert result.data.Vx_model[:3].isna().all()
    np.testing.assert_allclose(result.data.Vx_model[3:], 1, atol=1e-10)


@pytest.mark.parametrize(
    "overrides",
    [
        {"mass_kg": True},
        {"mass_kg": "0.2"},
        {"mass_kg": np.inf},
        {"min_data_points": 12.5},
        {"min_segment_points": True},
        {"max_gap_seconds": np.nan},
        {"max_gap_seconds": np.inf},
        {"bounce_velocity_threshold": np.nan},
        {"bounce_velocity_threshold": np.inf},
        {"fit_quality_threshold": np.nan},
        {"experiment_type": None},
        {"experiment_type": "  "},
        {"smoothing_window": 4},
        {"smoothing_window": 8},
        {"smoothing_window": 9.5},
        {"smoothing_window": True},
    ],
)
def test_settings_reject_nonfinite_or_wrongly_typed_parameters(overrides):
    with pytest.raises(ValueError):
        AnalysisSettings(**overrides).validate()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, "missing"])
def test_measurements_and_optional_confidence_are_validated(bad):
    t = np.linspace(0, 1, 20)
    for column in ["Time", "X", "Y", "Confidence"]:
        source = frame(t, t, t, Confidence=np.ones_like(t)).astype(object)
        source.loc[7, column] = bad
        with pytest.raises(ValueError):
            analyze_samples(source)


@pytest.mark.parametrize("track_id", [None, np.nan, np.inf, ""])
def test_missing_or_nonfinite_track_identifiers_are_rejected(track_id):
    t = np.linspace(0, 1, 20)
    ids = [1] * len(t)
    ids[10] = track_id
    with pytest.raises(ValueError, match="TrackID"):
        analyze_samples(frame(t, t, t, TrackID=ids))


def test_duplicate_headers_or_complex_measurements_fail_clearly():
    t = np.linspace(0, 1, 20)
    source = frame(t, t, t)
    with pytest.raises(ValueError, match="duplicate"):
        analyze_samples(pd.concat([source, source[["X"]]], axis=1))
    source["Y"] = t + 1j
    with pytest.raises(ValueError):
        analyze_samples(source)


def test_extreme_input_magnitude_fails_instead_of_exporting_infinite_physics():
    t = np.linspace(0, 1, 20)
    with pytest.raises(ValueError, match="numerical|magnitude|finite"):
        analyze_samples(frame(t, np.tile([-1e308, 1e308], 10), t))
