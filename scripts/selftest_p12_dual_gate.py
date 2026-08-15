from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dual_gate_alarm import (  # noqa: E402
    apply_alarm_rule,
    assess_support,
    audit_calibration,
    fit_support_model,
)


def main() -> None:
    rng = np.random.default_rng(20260811)

    stable_blocks = np.repeat(np.arange(12), 20)
    stable_scores = rng.lognormal(mean=0.0, sigma=0.15, size=stable_blocks.size)
    stable = audit_calibration(stable_scores, stable_blocks)
    assert stable.reliable, stable.to_dict()

    spike_blocks = np.repeat(np.arange(10), 10)
    spike_scores = np.zeros(spike_blocks.size, dtype=np.float64)
    spike_scores[spike_blocks == 9] = 100.0
    unstable = audit_calibration(spike_scores, spike_blocks)
    assert not unstable.reliable

    reference_blocks = np.repeat(np.arange(12), 20)
    base_temperature = np.tile(np.linspace(20.0, 30.0, 20), 12)
    temperature = base_temperature + rng.normal(0.0, 0.03, base_temperature.size)
    load = np.tile(np.linspace(0.9, 1.1, 20), 12) + rng.normal(0.0, 0.002, base_temperature.size)
    reference_numeric = np.column_stack([temperature, load])
    reference_categories = np.full((reference_numeric.shape[0], 1), "static")

    support_model = fit_support_model(
        reference_numeric,
        reference_blocks,
        reference_categories,
    )
    in_domain = assess_support(
        support_model,
        [[22.0, 0.94], [25.0, 1.00], [28.0, 1.06]],
        [["static"], ["static"], ["static"]],
    )
    assert np.all(in_domain.supported), in_domain

    shifted = assess_support(support_model, [[50.0, 1.0]], [["static"]])
    assert not shifted.supported[0]
    assert shifted.reason[0] == "outside_numeric_support"

    unseen = assess_support(support_model, [[25.0, 1.0]], [["dynamic"]])
    assert not unseen.supported[0]
    assert unseen.reason[0] == "unseen_category"

    high = float(stable.threshold + 1.0)
    low = float(stable.threshold - 1.0)
    alarm = apply_alarm_rule(
        [high, high, low, high, high, high],
        stable,
        [True, True, True, False, True, True],
    )
    assert alarm.final_status[:3] == ("no_alarm", "no_alarm", "alarm")
    assert alarm.final_status[3] == "abstain_support"
    assert alarm.final_status[4:] == ("no_alarm", "no_alarm")

    rejected_calibration = apply_alarm_rule([100.0], unstable, [True])
    assert rejected_calibration.final_status == ("abstain_calibration",)

    result = {
        "schema_version": "p12-dual-gate-selftest-v1",
        "status": "pass",
        "checks": {
            "stable_calibration_passes": True,
            "zero_scale_spike_calibration_rejected": True,
            "in_domain_targets_supported": True,
            "shifted_numeric_target_rejected": True,
            "unseen_category_rejected": True,
            "two_of_three_alarm_applied": True,
            "unsupported_sample_breaks_sequence": True,
            "unreliable_calibration_abstains": True,
        },
        "stable_calibration": stable.to_dict(),
        "support_model": {
            "reference_samples": int(reference_numeric.shape[0]),
            "features": int(reference_numeric.shape[1]),
            "k": support_model.k,
            "quantile": support_model.quantile,
            "distance_threshold": support_model.distance_threshold,
        },
    }
    report = ROOT / "data" / "reports" / "p12_dual_gate_selftest_v1.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({**result, "report": str(report)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
