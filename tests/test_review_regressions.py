"""Counterexamples found by independent review of the numerical/export contracts."""

import numpy as np
import pandas as pd
import pytest

from physics_motion.cli import demo_samples
from physics_motion.engine import AnalysisSettings, analyze_samples
from physics_motion.output import export_run


def test_symmetric_acceleration_change_does_not_qualify_as_gravity():
    t = np.linspace(0, 1, 101)
    # True ay ranges from -9.81 to +14.19, although quadratic R² > .999.
    frame = pd.DataFrame(
        {"Time": t, "X": 2 * t, "Y": 10 + 10 * t - 4.905 * t * t + 8 * (t - 0.5) ** 4}
    )
    result = analyze_samples(frame, AnalysisSettings(experiment_type="Horizontal toss"))
    assert result.segments[0].r2_y > 0.999
    assert result.stats["gravity_estimate_m_s2"] is None
    assert np.isfinite(result.data["Ay_model"]).all()


@pytest.mark.parametrize("spacing", [1e-14, 1e-15])
def test_unresolved_higher_order_diagnostic_does_not_destroy_valid_quadratic(spacing):
    t = np.r_[np.arange(4) * spacing, 0.3 + np.arange(4) * spacing, 0.6 + np.arange(4) * spacing]
    frame = pd.DataFrame({"Time": t, "X": 0.2 * t, "Y": 2 + t - 4.905 * t * t})
    result = analyze_samples(frame, AnalysisSettings(experiment_type="Horizontal toss"))
    assert np.allclose(result.data["Ay_model"], -9.81, atol=1e-10)
    assert result.stats["gravity_estimate_m_s2"] is None


def test_export_rejects_different_settings_from_the_calculation(tmp_path):
    used = AnalysisSettings(mass_kg=0.3, experiment_type="Horizontal toss")
    result = analyze_samples(demo_samples(), used)
    stated = AnalysisSettings(mass_kg=0.2, experiment_type="Horizontal toss")
    with pytest.raises(ValueError, match="settings"):
        export_run(result, stated, tmp_path)
    assert not list(tmp_path.iterdir())
