import numpy as np
import pandas as pd
import pytest

from physics_motion.engine import AnalysisSettings, analyze_samples


def samples(t, x, y):
    return pd.DataFrame({"Time": t, "X": x, "Y": y})


def test_projectile_gravity_velocity_energy_and_units():
    t = np.linspace(0, 0.6, 61)
    x = 0.3 + 2.0 * t
    y = 2.0 + 1.2 * t - 4.905 * t**2
    analysis = analyze_samples(
        samples(t, x, y), AnalysisSettings(mass_kg=0.5, experiment_type="Horizontal toss")
    )
    assert analysis.stats["gravity_estimate_m_s2"] == pytest.approx(9.81, rel=1e-8)
    assert analysis.stats["max_modeled_speed_m_s"] is not None
    assert analysis.segments[0].r2_y == pytest.approx(1.0)
    assert analysis.data.iloc[0].Vx_model == pytest.approx(2.0)
    assert analysis.data.iloc[0].Vy_model == pytest.approx(1.2)
    assert analysis.data.iloc[0].PE == pytest.approx(0.5 * 9.81 * 2.0)
    assert analysis.data.iloc[0].KE == pytest.approx(0.5 * 0.5 * (2**2 + 1.2**2))
    assert np.allclose(analysis.data.E_total, analysis.data.KE + analysis.data.PE)


def test_horizontal_constant_velocity_does_not_claim_gravity():
    t = np.linspace(0, 1, 60)
    result = analyze_samples(samples(t, 3 * t, np.ones_like(t)), AnalysisSettings())
    assert result.stats["gravity_estimate_m_s2"] is None
    assert result.stats["restitution_estimate"] is None
    assert result.data.Vx_model.to_numpy() == pytest.approx(np.full_like(t, 3))
    assert result.data.Ay_model.abs().max() < 1e-7


def test_stationary_is_finite_and_zero_speed():
    t = np.linspace(0, 1, 50)
    result = analyze_samples(samples(t, np.full_like(t, 0.4), np.full_like(t, 0.6)))
    assert result.stats["max_modeled_speed_m_s"] < 1e-8
    assert result.segments[0].r2_y == pytest.approx(1.0)


def test_gap_breaks_fits_and_does_not_interpolate():
    t = np.r_[np.linspace(0, 0.2, 11), np.linspace(2, 2.2, 11)]
    result = analyze_samples(samples(t, 2 * t, 1 + t), AnalysisSettings(max_gap_seconds=0.5))
    assert len(result.segments) == 2
    assert any("time gap" in w for w in result.warnings)
    assert result.data.shape[0] == len(t)


def test_does_not_infer_restitution_from_apex():
    t = np.linspace(0, 0.6, 61)
    result = analyze_samples(
        samples(t, 0.5 * t, 1 + 2 * t - 4.905 * t * t),
        AnalysisSettings(experiment_type="Free-fall / bounce"),
    )
    assert result.stats["restitution_estimate"] is None


def test_reject_duplicate_nonnumeric_and_insufficient_samples():
    t = np.linspace(0, 1, 15)
    df = samples(t, t, t)
    with pytest.raises(ValueError, match="strictly increasing"):
        analyze_samples(pd.concat([df.iloc[:5], df.iloc[[4]], df.iloc[5:]]))
    df["X"] = df["X"].astype(object)
    df.loc[4, "X"] = "invalid"
    with pytest.raises(ValueError, match="nonnumeric"):
        analyze_samples(df)
    with pytest.raises(ValueError, match="at least"):
        analyze_samples(samples(t[:4], t[:4], t[:4]))
    with pytest.raises(ValueError, match="Mass"):
        analyze_samples(samples(t, t, t), AnalysisSettings(mass_kg=0))


def test_bounce_candidate_detected_without_unphysical_claim():
    t = np.linspace(0, 0.8, 81)
    impact = 0.4
    # Downward motion before the impact and rebound at 70% speed afterwards.
    v_before = -1.8
    y = np.where(
        t < impact,
        1 + v_before * (t - impact) - 4.905 * (t - impact) ** 2,
        1 + 1.26 * (t - impact) - 4.905 * (t - impact) ** 2,
    )
    result = analyze_samples(
        samples(t, 0.2 * t, y), AnalysisSettings(experiment_type="Free-fall / bounce")
    )
    assert len(result.segments) >= 2
    # A fit can be contaminated near the impact; don't assert spurious precision.
    assert (
        result.stats["restitution_estimate"] is None
        or 0 <= result.stats["restitution_estimate"] <= 1
    )
