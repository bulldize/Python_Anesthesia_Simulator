#!/usr/bin/env python3
"""Debug-friendly wrapper around pkpd_pid_pipeline.

If the CSV path does not exist, this script generates a synthetic BIS dataset so
the full fitting + closed-loop pipeline can be exercised immediately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pkpd_pid_pipeline import (
    SimpleMainConfig,
    _namespace_from_simple_config,
    bis_model_from_ce,
    run_pipeline,
    simulate_ce_propo_schnider,
)


def generate_demo_csv(
    csv_path: Path,
    age: float,
    height: float,
    weight: float,
    sex: int,
    ts: float = 5.0,
    duration_s: float = 1800.0,
) -> Path:
    """Create a synthetic propofol/BIS dataset for quick debugging."""
    time_s = np.arange(0.0, duration_s + ts, ts, dtype=float)
    u_propo = np.zeros_like(time_s)

    u_propo[(time_s >= 60) & (time_s < 180)] = 0.9
    u_propo[(time_s >= 180) & (time_s < 420)] = 0.7
    u_propo[(time_s >= 420) & (time_s < 900)] = 0.5
    u_propo[(time_s >= 900) & (time_s < 1260)] = 0.3
    u_propo[(time_s >= 1260) & (time_s < 1500)] = 0.2
    u_propo[(time_s >= 1500)] = 0.2

    patient_characteristic = [age, height, weight, sex]
    ce_propo = simulate_ce_propo_schnider(u_propo, patient_characteristic, ts=ts)
    bis_clean = bis_model_from_ce(ce=ce_propo, c50=4.7, gamma=2.69, e0=95.9, emax=87.5)

    rng = np.random.default_rng(42)
    bis_noisy = np.clip(bis_clean + rng.normal(loc=0.0, scale=1.2, size=bis_clean.shape), 25.0, 98.0)

    df = pd.DataFrame(
        {
            "time": time_s,
            "u_propo": u_propo,
            "bis": bis_noisy,
            "bis_clean": bis_clean,
            "ce_propo": ce_propo,
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path


def main() -> None:
    output_dir = Path("outputs/pkpd_pid_debug")
    csv_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else output_dir / "demo_patient_timeseries.csv"

    age = 40.0
    height = 170.0
    weight = 70.0
    sex = 1

    if not csv_arg.exists():
        generate_demo_csv(
            csv_path=csv_arg,
            age=age,
            height=height,
            weight=weight,
            sex=sex,
        )
        print(f"[debug-wrapper] generated demo CSV: {csv_arg}")

    cfg = SimpleMainConfig(
        csv=str(csv_arg),
        age=age,
        height=height,
        weight=weight,
        sex=sex,
        output_dir=str(output_dir),
        control_duration=1800.0,
        bis_target=50.0,
        enable_tof_control=True,
        tof_target=10.0,
        debug_print_every=30,
        bis_band=5.0,
        tof_band=5.0,
        settle_window=120.0,
    )

    result = run_pipeline(_namespace_from_simple_config(cfg))
    summary = result["summary"]

    print("===== Debug pipeline done =====")
    print(f"input csv: {csv_arg}")
    print(f"sampling time ts: {result['ts']:.3f} s")
    print(f"fitted C50: {summary['fit']['c50']:.4f}")
    print(f"fit RMSE: {summary['fit']['metrics']['rmse']:.4f}")
    print(f"BIS settling time: {summary['closed_loop']['bis']['settling_time_s']}")
    print(f"outputs saved to: {result['output_dir']}")
    print("key artifacts: fit_diagnostics.png, closed_loop_overview.png, closed_loop_pid_debug.png, summary.json")


if __name__ == "__main__":
    main()
