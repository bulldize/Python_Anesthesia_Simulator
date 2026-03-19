#!/usr/bin/env python3
"""PKPD training + PID control pipeline.

Workflow:
1) Read CSV with pandas and plot dual-axis curves for u(t) and y(t) (BIS / TOF).
2) Assume Schnider PK model (Propofol), write a manual residual, and fit BIS C50 via least squares.
3) Build a fitted virtual patient class.
4) Run a simple PID closed loop, feed computed controls into PKPD model, and collect outputs.
5) Expose a main() entrypoint for the complete process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

_mpl_config_dir = Path(os.environ.get("MPLCONFIGDIR", Path(tempfile.gettempdir()) / "pas_mplconfig"))
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class _FallbackBISModel:
    """Minimal BIS model for this script when pd_models has extra dependencies."""

    def __init__(self, hill_param: list[float], ts: float):
        if len(hill_param) == 7:
            self.c50p, self.c50r, self.gamma, self.beta, self.E0, self.Emax, self.bis_delay = hill_param
        elif len(hill_param) == 6:
            self.c50p, self.c50r, self.gamma, self.beta, self.E0, self.Emax = hill_param
            self.bis_delay = 0.0
        else:
            raise ValueError("Fallback BIS model expects 6 or 7 hill parameters.")
        delay_steps = max(1, int(round(float(self.bis_delay) / max(float(ts), 1e-9))))
        self.bis_buffer = np.ones(delay_steps, dtype=float) * float(self.E0)

    def compute_bis(self, c_es_propo: float, c_es_remi: Optional[float] = 0.0) -> float:
        _ = c_es_remi
        c50 = max(float(self.c50p), 1e-6)
        ce = max(float(c_es_propo), 0.0)
        bis = float(self.E0) - float(self.Emax) * (ce**float(self.gamma)) / (c50**float(self.gamma) + ce**float(self.gamma))
        return float(bis)

    def one_step(self, c_es_propo: float, c_es_remi: Optional[float] = 0.0) -> float:
        bis_now = self.compute_bis(c_es_propo=c_es_propo, c_es_remi=c_es_remi)
        self.bis_buffer = np.roll(self.bis_buffer, -1)
        self.bis_buffer[-1] = bis_now
        return float(self.bis_buffer[0])


class _FallbackTOFModel:
    """Minimal TOF model for this script when pd_models has extra dependencies."""

    def __init__(self, hill_model: str = "Weatherley", hill_param: Optional[dict] = None):
        hill_param = hill_param or {}
        self.hill_model = hill_model
        self.C50 = float(hill_param.get("C50", 0.625))
        self.gamma = float(hill_param.get("gamma", 4.25))

    def compute_tof(self, Ce):
        ce = np.asarray(Ce, dtype=float)
        tof = (100.0 * self.C50**self.gamma) / (self.C50**self.gamma + ce**self.gamma)
        if np.ndim(tof) == 0:
            return float(tof)
        return tof


def _import_pas_modules():
    """Import PAS modules with a source-tree fallback."""
    try:
        from python_anesthesia_simulator.pk_models import CompartmentModel, AtracuriumModel
        from python_anesthesia_simulator.pd_models import BIS_model, TOF_model
    except Exception:
        project_root = Path(__file__).resolve().parents[1]
        module_path = project_root / "src" / "python_anesthesia_simulator"
        if str(module_path) not in sys.path:
            sys.path.insert(0, str(module_path))
        from pk_models import CompartmentModel, AtracuriumModel
        try:
            from pd_models import BIS_model, TOF_model
        except Exception:
            BIS_model = _FallbackBISModel
            TOF_model = _FallbackTOFModel
    return CompartmentModel, AtracuriumModel, BIS_model, TOF_model


CompartmentModel, AtracuriumModel, BIS_model, TOF_model = _import_pas_modules()


def compute_lbm(weight_kg: float, height_cm: float, sex: int) -> float:
    """Compute lean body mass with PAS convention: sex=0 female, sex=1 male."""
    if sex == 1:
        return 1.1 * weight_kg - 128 * (weight_kg / height_cm) ** 2
    return 1.07 * weight_kg - 148 * (weight_kg / height_cm) ** 2


def infer_sampling_time(time_s: np.ndarray, default: float = 1.0) -> float:
    """Infer dt from time array, fallback to default."""
    if len(time_s) < 2:
        return default
    diffs = np.diff(time_s.astype(float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return default
    return float(np.median(diffs))


def _find_column(df: pd.DataFrame, explicit: Optional[str], candidates: list[str]) -> Optional[str]:
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(f"Column '{explicit}' not found in CSV. Available: {list(df.columns)}")
        return explicit
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def load_csv_timeseries(
    csv_path: Path,
    time_col: Optional[str],
    u_col: Optional[str],
    bis_col: Optional[str],
    tof_col: Optional[str],
) -> tuple[pd.DataFrame, str, str, Optional[str], Optional[str]]:
    """Load CSV and resolve required columns."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    t_name = _find_column(df, time_col, ["time", "t", "seconds", "sec", "timestamp"])
    if t_name is None:
        t_name = "time"
        df[t_name] = np.arange(len(df), dtype=float)

    u_name = _find_column(
        df,
        u_col,
        ["u", "u_t", "u_propo", "u_propofol", "propofol_rate", "infusion_rate", "rate"],
    )
    if u_name is None:
        raise ValueError("Cannot infer control column u(t). Please pass --u-col explicitly.")

    bis_name = _find_column(df, bis_col, ["bis", "y", "y_bis", "bis_value"])
    tof_name = _find_column(df, tof_col, ["tof", "nmb", "muscle", "muscle_relax", "y_tof", "tof_value"])

    # Convert to numeric where possible.
    for col in [t_name, u_name, bis_name, tof_name]:
        if col is not None:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep rows with valid time/u.
    df = df.dropna(subset=[t_name, u_name]).reset_index(drop=True)
    if bis_name is None and tof_name is None:
        raise ValueError("Need at least one output column: BIS or TOF.")

    return df, t_name, u_name, bis_name, tof_name


def plot_dual_axis_u_y(
    time_s: np.ndarray,
    u_t: np.ndarray,
    bis: Optional[np.ndarray],
    tof: Optional[np.ndarray],
    out_path: Path,
    title_prefix: str,
) -> None:
    """Plot dual-axis curves for u(t) with BIS/TOF."""
    nrows = 2 if tof is not None else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(11, 7), sharex=True)
    if nrows == 1:
        axes = [axes]

    # u(t) + BIS
    ax_u = axes[0]
    ax_y = ax_u.twinx()
    ax_u.plot(time_s, u_t, color="tab:blue", lw=1.8, label="u(t) - infusion")
    if bis is not None:
        ax_y.plot(time_s, bis, color="tab:red", lw=1.5, label="BIS")
    ax_u.set_ylabel("u(t)")
    ax_y.set_ylabel("BIS")
    ax_u.grid(alpha=0.25)
    ax_u.set_title(f"{title_prefix} | u(t) vs BIS")

    lines = ax_u.get_lines() + ax_y.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax_u.legend(lines, labels, loc="best")

    # u(t) + TOF
    if tof is not None:
        ax_u2 = axes[1]
        ax_y2 = ax_u2.twinx()
        ax_u2.plot(time_s, u_t, color="tab:blue", lw=1.8, label="u(t) - infusion")
        ax_y2.plot(time_s, tof, color="tab:green", lw=1.5, label="TOF")
        ax_u2.set_ylabel("u(t)")
        ax_y2.set_ylabel("TOF (%)")
        ax_u2.grid(alpha=0.25)
        ax_u2.set_title(f"{title_prefix} | u(t) vs TOF")
        lines2 = ax_u2.get_lines() + ax_y2.get_lines()
        labels2 = [ln.get_label() for ln in lines2]
        ax_u2.legend(lines2, labels2, loc="best")

    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def simulate_ce_propo_schnider(
    u_propo: np.ndarray,
    patient_characteristic: list[float],
    ts: float,
) -> np.ndarray:
    """Simulate propofol effect-site concentration Ce under Schnider PK."""
    age, height, weight, sex = patient_characteristic
    lbm = compute_lbm(weight_kg=weight, height_cm=height, sex=int(sex))
    pk = CompartmentModel(
        Patient_characteristic=[age, height, weight, sex],
        lbm=lbm,
        drug="Propofol",
        model="Schnider",
        ts=ts,
    )
    ce = np.zeros_like(u_propo, dtype=float)
    for k, uk in enumerate(u_propo):
        ce[k] = float(pk.one_step(float(uk)))
    return ce


def bis_model_from_ce(ce: np.ndarray, c50: float, gamma: float, e0: float, emax: float) -> np.ndarray:
    """Simple one-drug Hill BIS model."""
    ce = np.asarray(ce, dtype=float)
    c50 = max(float(c50), 1e-6)
    return e0 - emax * (ce**gamma) / (c50**gamma + ce**gamma)


def bis_residual(c50_arr: np.ndarray, ce: np.ndarray, bis_meas: np.ndarray, gamma: float, e0: float, emax: float) -> np.ndarray:
    """Manual residual for least squares fitting."""
    c50 = float(c50_arr[0])
    pred = bis_model_from_ce(ce=ce, c50=c50, gamma=gamma, e0=e0, emax=emax)
    return pred - bis_meas


@dataclass
class C50FitResult:
    c50: float
    success: bool
    cost: float
    nfev: int


@dataclass
class SimpleMainConfig:
    """Minimal configuration for a clean main entrypoint."""
    csv: str
    age: float
    height: float
    weight: float
    sex: int
    output_dir: str = "outputs/pkpd_pid"
    control_duration: float = 1800.0
    bis_target: float = 50.0
    enable_tof_control: bool = False
    tof_target: float = 10.0
    time_col: Optional[str] = None
    u_col: Optional[str] = None
    bis_col: Optional[str] = None
    tof_col: Optional[str] = None
    ts: Optional[float] = None
    debug_print_every: int = 0
    bis_band: float = 5.0
    tof_band: float = 5.0
    settle_window: float = 120.0


def _namespace_from_simple_config(cfg: SimpleMainConfig) -> argparse.Namespace:
    """Expand minimal config into full pipeline parameters with stable defaults."""
    return argparse.Namespace(
        csv=cfg.csv,
        output_dir=cfg.output_dir,
        time_col=cfg.time_col,
        u_col=cfg.u_col,
        bis_col=cfg.bis_col,
        tof_col=cfg.tof_col,
        age=cfg.age,
        height=cfg.height,
        weight=cfg.weight,
        sex=cfg.sex,
        ts=cfg.ts,
        gamma_bis=2.69,
        e0_bis=95.9,
        emax_bis=87.5,
        c50_init=4.5,
        c50_lb=0.1,
        c50_ub=20.0,
        tof_c50=0.625,
        tof_gamma=4.25,
        control_duration=cfg.control_duration,
        bis_target=cfg.bis_target,
        tof_target=cfg.tof_target,
        kp_bis=0.08,
        ki_bis=0.002,
        kd_bis=0.0,
        u_propo_min=0.0,
        u_propo_max=6.67,
        enable_tof_control=cfg.enable_tof_control,
        kp_tof=0.04,
        ki_tof=0.001,
        kd_tof=0.0,
        u_atra_min=0.0,
        u_atra_max=20.0,
        debug_print_every=cfg.debug_print_every,
        bis_band=cfg.bis_band,
        tof_band=cfg.tof_band,
        settle_window=cfg.settle_window,
    )


def fit_c50_least_squares(
    ce: np.ndarray,
    bis_meas: np.ndarray,
    gamma: float,
    e0: float,
    emax: float,
    c50_init: float,
    c50_lb: float,
    c50_ub: float,
) -> C50FitResult:
    """Fit C50 from (u, BIS) using Schnider PK + manual residual + least squares."""
    mask = np.isfinite(ce) & np.isfinite(bis_meas)
    ce_fit = ce[mask]
    bis_fit = bis_meas[mask]
    if len(ce_fit) < 10:
        raise ValueError("Not enough valid samples for fitting (need >= 10).")

    opt = least_squares(
        fun=bis_residual,
        x0=np.array([c50_init], dtype=float),
        bounds=(np.array([c50_lb], dtype=float), np.array([c50_ub], dtype=float)),
        args=(ce_fit, bis_fit, gamma, e0, emax),
    )
    return C50FitResult(
        c50=float(opt.x[0]),
        success=bool(opt.success),
        cost=float(opt.cost),
        nfev=int(opt.nfev),
    )


class FittedVirtualPatient:
    """Virtual patient built from fitted BIS C50 and fixed PK/PD structures."""

    def __init__(
        self,
        patient_characteristic: list[float],
        c50_bis: float,
        gamma_bis: float,
        e0_bis: float,
        emax_bis: float,
        tof_c50: float,
        tof_gamma: float,
        ts: float,
    ):
        self.patient_characteristic = patient_characteristic
        self.ts = float(ts)
        age, height, weight, sex = patient_characteristic
        lbm = compute_lbm(weight_kg=weight, height_cm=height, sex=int(sex))

        self.propo_pk = CompartmentModel(
            Patient_characteristic=[age, height, weight, sex],
            lbm=lbm,
            drug="Propofol",
            model="Schnider",
            ts=self.ts,
        )
        self.atra_pk = AtracuriumModel(
            Patient_characteristic=[age, height, weight, sex],
            model="WardWeatherleyLago",
            ts=self.ts,
        )
        # Fixed one-drug BIS model with fitted C50.
        self.bis_pd = BIS_model(
            hill_param=[c50_bis, 0.0, gamma_bis, 0.0, e0_bis, emax_bis, 0.0],
            ts=self.ts,
        )
        self.tof_pd = TOF_model(
            hill_model="Weatherley",
            hill_param={"C50": tof_c50, "gamma": tof_gamma},
        )
        self.reset()

    def reset(self) -> None:
        self.propo_pk.x = np.zeros_like(self.propo_pk.x)
        self.propo_pk.y = np.dot(self.propo_pk.discretize_sys.C, self.propo_pk.x)
        self.atra_pk.x = np.zeros_like(self.atra_pk.x)
        self.atra_pk.y = np.dot(self.atra_pk.discretize_sys.C, self.atra_pk.x)
        self.bis_pd.bis_buffer = np.ones_like(self.bis_pd.bis_buffer) * self.bis_pd.E0
        self.time = 0.0
        self.last = {
            "time": 0.0,
            "u_propo": 0.0,
            "u_atra": 0.0,
            "ce_propo": 0.0,
            "ce_atra": 0.0,
            "bis": float(self.bis_pd.compute_bis(0.0, 0.0)),
            "tof": float(self.tof_pd.compute_tof(0.0)),
        }

    def step(self, u_propo: float, u_atra: float) -> dict[str, float]:
        ce_propo = float(self.propo_pk.one_step(float(u_propo)))
        ce_atra = float(self.atra_pk.one_step(float(u_atra)))
        bis = float(self.bis_pd.one_step(ce_propo, 0.0))
        tof = float(self.tof_pd.compute_tof(ce_atra))
        self.time += self.ts
        self.last = {
            "time": self.time,
            "u_propo": float(u_propo),
            "u_atra": float(u_atra),
            "ce_propo": ce_propo,
            "ce_atra": ce_atra,
            "bis": bis,
            "tof": tof,
        }
        return dict(self.last)


class PIDController:
    """Simple PID with anti-windup and optional reverse action."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        dt: float,
        u_min: float,
        u_max: float,
        reverse_acting: bool = True,
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = float(dt)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.reverse_acting = bool(reverse_acting)
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def compute(self, setpoint: float, measurement: float) -> dict[str, float | bool]:
        error = (measurement - setpoint) if self.reverse_acting else (setpoint - measurement)
        if not self.initialized:
            self.prev_error = error
            self.initialized = True

        integral_candidate = self.integral + error * self.dt
        derivative = (error - self.prev_error) / self.dt
        p_term = self.kp * error
        i_term = self.ki * integral_candidate
        d_term = self.kd * derivative
        u_raw = p_term + i_term + d_term
        u_sat = float(np.clip(u_raw, self.u_min, self.u_max))

        # Anti-windup: stop integrating when saturated in the same direction.
        anti_windup = (u_raw > self.u_max and error > 0) or (u_raw < self.u_min and error < 0)
        if anti_windup:
            i_term = self.ki * self.integral
        else:
            self.integral = integral_candidate

        self.prev_error = error
        return {
            "error": float(error),
            "derivative": float(derivative),
            "p_term": float(p_term),
            "i_term": float(i_term),
            "d_term": float(d_term),
            "u_raw": float(u_raw),
            "u_sat": float(u_sat),
            "saturated": bool(abs(u_raw - u_sat) > 1e-12),
            "anti_windup": bool(anti_windup),
            "integral_state": float(self.integral),
        }

    def step(self, setpoint: float, measurement: float) -> float:
        return float(self.compute(setpoint=setpoint, measurement=measurement)["u_sat"])


def run_closed_loop_control(
    patient: FittedVirtualPatient,
    duration_s: float,
    bis_target: float,
    pid_bis: PIDController,
    tof_target: Optional[float] = None,
    pid_tof: Optional[PIDController] = None,
    debug_print_every: int = 0,
) -> pd.DataFrame:
    """Run PID closed loop and return trajectory dataframe."""
    n_steps = int(np.ceil(duration_s / patient.ts))
    records: list[dict[str, float]] = []

    y_bis = float(patient.last["bis"])
    y_tof = float(patient.last["tof"])
    for step_idx in range(n_steps):
        bis_pid = pid_bis.compute(setpoint=bis_target, measurement=y_bis)
        u_propo = float(bis_pid["u_sat"])
        if pid_tof is not None and tof_target is not None:
            tof_pid = pid_tof.compute(setpoint=tof_target, measurement=y_tof)
            u_atra = float(tof_pid["u_sat"])
        else:
            u_atra = 0.0
            tof_pid = {
                "error": np.nan,
                "derivative": np.nan,
                "p_term": np.nan,
                "i_term": np.nan,
                "d_term": np.nan,
                "u_raw": np.nan,
                "u_sat": 0.0,
                "saturated": False,
                "anti_windup": False,
                "integral_state": np.nan,
            }

        out = patient.step(u_propo=u_propo, u_atra=u_atra)
        record = {
            "step_index": int(step_idx),
            "bis_pre": float(y_bis),
            "tof_pre": float(y_tof),
            "bis_target": float(bis_target),
            "tof_target": float(tof_target) if tof_target is not None else np.nan,
            "bis_error": float(bis_pid["error"]),
            "tof_error": float(tof_pid["error"]),
            "u_propo_p_term": float(bis_pid["p_term"]),
            "u_propo_i_term": float(bis_pid["i_term"]),
            "u_propo_d_term": float(bis_pid["d_term"]),
            "u_propo_raw": float(bis_pid["u_raw"]),
            "u_propo_sat": float(bis_pid["u_sat"]),
            "u_propo_saturated": bool(bis_pid["saturated"]),
            "u_propo_anti_windup": bool(bis_pid["anti_windup"]),
            "u_propo_integral_state": float(bis_pid["integral_state"]),
            "u_atra_p_term": float(tof_pid["p_term"]),
            "u_atra_i_term": float(tof_pid["i_term"]),
            "u_atra_d_term": float(tof_pid["d_term"]),
            "u_atra_raw": float(tof_pid["u_raw"]),
            "u_atra_sat": float(tof_pid["u_sat"]),
            "u_atra_saturated": bool(tof_pid["saturated"]),
            "u_atra_anti_windup": bool(tof_pid["anti_windup"]),
            "u_atra_integral_state": float(tof_pid["integral_state"]),
        }
        record.update(out)
        records.append(record)
        if debug_print_every > 0 and ((step_idx + 1) % debug_print_every == 0 or step_idx == n_steps - 1):
            print(
                "[closed-loop]"
                f" step={step_idx + 1:04d}"
                f" t={out['time']:7.1f}s"
                f" BIS={out['bis']:6.2f} (err={bis_pid['error']:7.3f})"
                f" u_propo={u_propo:6.3f}"
                f" Ce_propo={out['ce_propo']:6.3f}"
                f" TOF={out['tof']:6.2f}"
                f" u_atra={u_atra:6.3f}"
            )
        y_bis = out["bis"]
        y_tof = out["tof"]

    return pd.DataFrame(records)


def plot_fit_result(time_s: np.ndarray, bis_meas: np.ndarray, bis_pred: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(time_s, bis_meas, lw=1.6, color="tab:red", label="BIS measured")
    ax.plot(time_s, bis_pred, lw=1.3, color="tab:orange", linestyle="--", label="BIS fitted")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("BIS")
    ax.set_title("C50 fitting result (manual residual + least squares)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _to_builtin(value):
    """Convert numpy / pandas scalars into JSON-safe builtins."""
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    return value


def save_json(data: dict[str, object], out_path: Path) -> None:
    out_path.write_text(json.dumps(_to_builtin(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize_fit(bis_meas: np.ndarray, bis_pred: np.ndarray, ce_propo: np.ndarray) -> dict[str, float | int]:
    residual = np.asarray(bis_pred, dtype=float) - np.asarray(bis_meas, dtype=float)
    mask = np.isfinite(residual)
    residual = residual[mask]
    ce_propo = np.asarray(ce_propo, dtype=float)[mask]
    return {
        "n_samples": int(mask.sum()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "mean_residual": float(np.mean(residual)),
        "max_abs_residual": float(np.max(np.abs(residual))),
        "ce_min": float(np.min(ce_propo)),
        "ce_max": float(np.max(ce_propo)),
    }


def _find_settling_time(
    time_s: np.ndarray,
    measurement: np.ndarray,
    target: float,
    band: float,
    settle_window_s: float,
) -> Optional[float]:
    if len(time_s) == 0:
        return None
    dt = infer_sampling_time(time_s)
    hold_steps = max(1, int(np.ceil(settle_window_s / max(dt, 1e-9))))
    within_band = np.abs(np.asarray(measurement, dtype=float) - float(target)) <= float(band)
    if len(within_band) < hold_steps:
        return None
    window_hits = np.convolve(within_band.astype(int), np.ones(hold_steps, dtype=int), mode="valid")
    valid_idx = np.flatnonzero(window_hits == hold_steps)
    if len(valid_idx) == 0:
        return None
    return float(time_s[int(valid_idx[0])])


def summarize_tracking_response(
    time_s: np.ndarray,
    measurement: np.ndarray,
    target: float,
    band: float,
    settle_window_s: float,
) -> dict[str, Optional[float]]:
    measurement = np.asarray(measurement, dtype=float)
    error = measurement - float(target)
    abs_error = np.abs(error)
    dt = infer_sampling_time(np.asarray(time_s, dtype=float))
    return {
        "target": float(target),
        "final_value": float(measurement[-1]),
        "final_error": float(error[-1]),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(abs_error)),
        "iae": float(np.sum(abs_error) * dt),
        "max_above_target": float(np.max(measurement - float(target))),
        "max_below_target": float(np.max(float(target) - measurement)),
        "within_band_fraction": float(np.mean(abs_error <= float(band))),
        "settling_time_s": _find_settling_time(
            time_s=np.asarray(time_s, dtype=float),
            measurement=measurement,
            target=target,
            band=band,
            settle_window_s=settle_window_s,
        ),
    }


def summarize_actuator(
    command: np.ndarray,
    raw_command: np.ndarray,
    saturated: np.ndarray,
) -> dict[str, float]:
    command = np.asarray(command, dtype=float)
    raw_command = np.asarray(raw_command, dtype=float)
    saturated = np.asarray(saturated, dtype=bool)
    return {
        "min": float(np.min(command)),
        "max": float(np.max(command)),
        "mean": float(np.mean(command)),
        "raw_min": float(np.min(raw_command)),
        "raw_max": float(np.max(raw_command)),
        "saturation_fraction": float(np.mean(saturated)),
    }


def plot_fit_diagnostics(
    time_s: np.ndarray,
    ce_propo: np.ndarray,
    bis_meas: np.ndarray,
    bis_pred: np.ndarray,
    out_path: Path,
) -> None:
    residual = np.asarray(bis_pred, dtype=float) - np.asarray(bis_meas, dtype=float)
    order = np.argsort(ce_propo)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(time_s, bis_meas, color="tab:red", lw=1.5, label="measured")
    axes[0, 0].plot(time_s, bis_pred, color="tab:orange", lw=1.2, linestyle="--", label="fitted")
    axes[0, 0].set_title("BIS fit")
    axes[0, 0].set_ylabel("BIS")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(time_s, residual, color="tab:purple", lw=1.2)
    axes[0, 1].axhline(0.0, color="black", lw=0.9, linestyle="--")
    axes[0, 1].set_title("Fit residual")
    axes[0, 1].set_ylabel("BIS residual")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(time_s, ce_propo, color="tab:blue", lw=1.4)
    axes[1, 0].set_title("Schnider effect-site concentration")
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].set_ylabel("Ce propo")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].scatter(ce_propo, bis_meas, s=12, alpha=0.45, color="tab:red", label="measured")
    axes[1, 1].plot(ce_propo[order], bis_pred[order], color="tab:orange", lw=1.5, label="fitted curve")
    axes[1, 1].set_title("BIS vs Ce")
    axes[1, 1].set_xlabel("Ce propo")
    axes[1, 1].set_ylabel("BIS")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_closed_loop_overview(
    closed_loop_df: pd.DataFrame,
    out_path: Path,
    bis_target: float,
    bis_band: float,
    u_propo_min: float,
    u_propo_max: float,
    tof_target: Optional[float] = None,
    tof_band: Optional[float] = None,
    u_atra_min: Optional[float] = None,
    u_atra_max: Optional[float] = None,
) -> None:
    time_s = closed_loop_df["time"].to_numpy(dtype=float)
    enable_tof = tof_target is not None and "u_atra" in closed_loop_df.columns
    nrows = 5 if enable_tof else 3
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 3.1 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(time_s, closed_loop_df["bis"].to_numpy(dtype=float), color="tab:red", lw=1.5, label="BIS")
    ax.axhline(float(bis_target), color="black", lw=1.0, linestyle="--", label="target")
    ax.fill_between(
        time_s,
        float(bis_target) - float(bis_band),
        float(bis_target) + float(bis_band),
        color="tab:red",
        alpha=0.08,
        label=f"target +/- {bis_band:g}",
    )
    ax.set_ylabel("BIS")
    ax.set_title("Closed-loop outputs")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    row_idx = 1
    if enable_tof:
        ax = axes[row_idx]
        ax.plot(time_s, closed_loop_df["tof"].to_numpy(dtype=float), color="tab:green", lw=1.5, label="TOF")
        ax.axhline(float(tof_target), color="black", lw=1.0, linestyle="--", label="target")
        if tof_band is not None:
            ax.fill_between(
                time_s,
                float(tof_target) - float(tof_band),
                float(tof_target) + float(tof_band),
                color="tab:green",
                alpha=0.08,
                label=f"target +/- {tof_band:g}",
            )
        ax.set_ylabel("TOF")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        row_idx += 1

    ax = axes[row_idx]
    ax.plot(time_s, closed_loop_df["u_propo_raw"].to_numpy(dtype=float), color="0.6", lw=1.0, linestyle="--", label="u raw")
    ax.plot(time_s, closed_loop_df["u_propo"].to_numpy(dtype=float), color="tab:blue", lw=1.5, label="u sat")
    ax.axhline(float(u_propo_min), color="black", lw=0.8, linestyle=":")
    ax.axhline(float(u_propo_max), color="black", lw=0.8, linestyle=":")
    sat_mask = closed_loop_df["u_propo_saturated"].to_numpy(dtype=bool)
    if np.any(sat_mask):
        ax.scatter(
            time_s[sat_mask],
            closed_loop_df.loc[sat_mask, "u_propo"].to_numpy(dtype=float),
            s=14,
            color="tab:blue",
            alpha=0.8,
            label="saturated",
        )
    ax.set_ylabel("u propo")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    row_idx += 1

    if enable_tof:
        ax = axes[row_idx]
        ax.plot(time_s, closed_loop_df["u_atra_raw"].to_numpy(dtype=float), color="0.6", lw=1.0, linestyle="--", label="u raw")
        ax.plot(time_s, closed_loop_df["u_atra"].to_numpy(dtype=float), color="tab:olive", lw=1.5, label="u sat")
        ax.axhline(float(u_atra_min), color="black", lw=0.8, linestyle=":")
        ax.axhline(float(u_atra_max), color="black", lw=0.8, linestyle=":")
        sat_mask = closed_loop_df["u_atra_saturated"].to_numpy(dtype=bool)
        if np.any(sat_mask):
            ax.scatter(
                time_s[sat_mask],
                closed_loop_df.loc[sat_mask, "u_atra"].to_numpy(dtype=float),
                s=14,
                color="tab:olive",
                alpha=0.8,
                label="saturated",
            )
        ax.set_ylabel("u atra")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        row_idx += 1

    ax = axes[row_idx]
    ax.plot(time_s, closed_loop_df["ce_propo"].to_numpy(dtype=float), color="tab:blue", lw=1.4, label="Ce propo")
    if enable_tof:
        ax.plot(time_s, closed_loop_df["ce_atra"].to_numpy(dtype=float), color="tab:olive", lw=1.4, label="Ce atra")
    ax.set_ylabel("Ce")
    ax.set_xlabel("time (s)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pid_debug(closed_loop_df: pd.DataFrame, out_path: Path, enable_tof_control: bool) -> None:
    time_s = closed_loop_df["time"].to_numpy(dtype=float)
    nrows = 2 if enable_tof_control else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 3.6 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    def _plot_terms(ax, prefix: str, color: str, title: str) -> None:
        ax.plot(time_s, closed_loop_df[f"{prefix}_p_term"].to_numpy(dtype=float), lw=1.1, label="P", color=color)
        ax.plot(time_s, closed_loop_df[f"{prefix}_i_term"].to_numpy(dtype=float), lw=1.1, label="I", color="tab:orange")
        ax.plot(time_s, closed_loop_df[f"{prefix}_d_term"].to_numpy(dtype=float), lw=1.1, label="D", color="tab:green")
        ax.plot(time_s, closed_loop_df[f"{prefix}_raw"].to_numpy(dtype=float), lw=1.0, linestyle="--", label="raw", color="0.45")
        ax.plot(time_s, closed_loop_df[f"{prefix}_sat"].to_numpy(dtype=float), lw=1.4, label="sat", color="black")
        ax.set_title(title)
        ax.set_ylabel("command")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    _plot_terms(axes[0], prefix="u_propo", color="tab:blue", title="BIS PID internals")
    if enable_tof_control:
        _plot_terms(axes[1], prefix="u_atra", color="tab:olive", title="TOF PID internals")
        axes[1].set_xlabel("time (s)")
    else:
        axes[0].set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    """Unified entry: training, control computation, and PKPD closed-loop simulation."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, t_col, u_col, bis_col, tof_col = load_csv_timeseries(
        csv_path=Path(args.csv),
        time_col=args.time_col,
        u_col=args.u_col,
        bis_col=args.bis_col,
        tof_col=args.tof_col,
    )

    time_s = df[t_col].to_numpy(dtype=float)
    u_t = df[u_col].to_numpy(dtype=float)
    bis = df[bis_col].to_numpy(dtype=float) if bis_col else None
    tof = df[tof_col].to_numpy(dtype=float) if tof_col else None
    ts = float(args.ts) if args.ts is not None else infer_sampling_time(time_s)

    plot_dual_axis_u_y(
        time_s=time_s,
        u_t=u_t,
        bis=bis,
        tof=tof,
        out_path=output_dir / "raw_dual_axis.png",
        title_prefix="Input CSV",
    )

    patient_characteristic = [args.age, args.height, args.weight, args.sex]

    if bis is None:
        raise ValueError("BIS column is required for C50 fitting.")
    ce_propo = simulate_ce_propo_schnider(u_t, patient_characteristic, ts=ts)
    fit = fit_c50_least_squares(
        ce=ce_propo,
        bis_meas=bis,
        gamma=args.gamma_bis,
        e0=args.e0_bis,
        emax=args.emax_bis,
        c50_init=args.c50_init,
        c50_lb=args.c50_lb,
        c50_ub=args.c50_ub,
    )

    bis_pred = bis_model_from_ce(
        ce=ce_propo,
        c50=fit.c50,
        gamma=args.gamma_bis,
        e0=args.e0_bis,
        emax=args.emax_bis,
    )
    fit_df = pd.DataFrame(
        {
            "time": time_s,
            "u": u_t,
            "ce_propo_schnider": ce_propo,
            "bis_measured": bis,
            "bis_fitted": bis_pred,
            "bis_residual": bis_pred - bis,
        }
    )
    fit_df.to_csv(output_dir / "fit_result.csv", index=False)
    plot_fit_result(time_s=time_s, bis_meas=bis, bis_pred=bis_pred, out_path=output_dir / "fit_result.png")
    plot_fit_diagnostics(
        time_s=time_s,
        ce_propo=ce_propo,
        bis_meas=bis,
        bis_pred=bis_pred,
        out_path=output_dir / "fit_diagnostics.png",
    )
    fit_summary = summarize_fit(bis_meas=bis, bis_pred=bis_pred, ce_propo=ce_propo)

    patient = FittedVirtualPatient(
        patient_characteristic=patient_characteristic,
        c50_bis=fit.c50,
        gamma_bis=args.gamma_bis,
        e0_bis=args.e0_bis,
        emax_bis=args.emax_bis,
        tof_c50=args.tof_c50,
        tof_gamma=args.tof_gamma,
        ts=ts,
    )

    pid_bis = PIDController(
        kp=args.kp_bis,
        ki=args.ki_bis,
        kd=args.kd_bis,
        dt=ts,
        u_min=args.u_propo_min,
        u_max=args.u_propo_max,
        reverse_acting=True,
    )

    if args.enable_tof_control:
        pid_tof = PIDController(
            kp=args.kp_tof,
            ki=args.ki_tof,
            kd=args.kd_tof,
            dt=ts,
            u_min=args.u_atra_min,
            u_max=args.u_atra_max,
            reverse_acting=True,
        )
    else:
        pid_tof = None

    closed_loop_df = run_closed_loop_control(
        patient=patient,
        duration_s=args.control_duration,
        bis_target=args.bis_target,
        pid_bis=pid_bis,
        tof_target=args.tof_target if args.enable_tof_control else None,
        pid_tof=pid_tof,
        debug_print_every=args.debug_print_every,
    )
    closed_loop_df.to_csv(output_dir / "closed_loop.csv", index=False)
    plot_dual_axis_u_y(
        time_s=closed_loop_df["time"].to_numpy(dtype=float),
        u_t=closed_loop_df["u_propo"].to_numpy(dtype=float),
        bis=closed_loop_df["bis"].to_numpy(dtype=float),
        tof=closed_loop_df["tof"].to_numpy(dtype=float),
        out_path=output_dir / "closed_loop_dual_axis.png",
        title_prefix="Closed-loop simulation",
    )

    plot_closed_loop_overview(
        closed_loop_df=closed_loop_df,
        out_path=output_dir / "closed_loop_overview.png",
        bis_target=args.bis_target,
        bis_band=args.bis_band,
        u_propo_min=args.u_propo_min,
        u_propo_max=args.u_propo_max,
        tof_target=args.tof_target if args.enable_tof_control else None,
        tof_band=args.tof_band if args.enable_tof_control else None,
        u_atra_min=args.u_atra_min if args.enable_tof_control else None,
        u_atra_max=args.u_atra_max if args.enable_tof_control else None,
    )
    plot_pid_debug(
        closed_loop_df=closed_loop_df,
        out_path=output_dir / "closed_loop_pid_debug.png",
        enable_tof_control=bool(args.enable_tof_control),
    )

    closed_loop_summary = {
        "duration_s": float(closed_loop_df["time"].iloc[-1]) if not closed_loop_df.empty else 0.0,
        "n_steps": int(len(closed_loop_df)),
        "bis": summarize_tracking_response(
            time_s=closed_loop_df["time"].to_numpy(dtype=float),
            measurement=closed_loop_df["bis"].to_numpy(dtype=float),
            target=args.bis_target,
            band=args.bis_band,
            settle_window_s=args.settle_window,
        ),
        "u_propo": summarize_actuator(
            command=closed_loop_df["u_propo"].to_numpy(dtype=float),
            raw_command=closed_loop_df["u_propo_raw"].to_numpy(dtype=float),
            saturated=closed_loop_df["u_propo_saturated"].to_numpy(dtype=bool),
        ),
    }
    if args.enable_tof_control:
        closed_loop_summary["tof"] = summarize_tracking_response(
            time_s=closed_loop_df["time"].to_numpy(dtype=float),
            measurement=closed_loop_df["tof"].to_numpy(dtype=float),
            target=args.tof_target,
            band=args.tof_band,
            settle_window_s=args.settle_window,
        )
        closed_loop_summary["u_atra"] = summarize_actuator(
            command=closed_loop_df["u_atra"].to_numpy(dtype=float),
            raw_command=closed_loop_df["u_atra_raw"].to_numpy(dtype=float),
            saturated=closed_loop_df["u_atra_saturated"].to_numpy(dtype=bool),
        )

    summary = {
        "input_csv": str(Path(args.csv).resolve()),
        "resolved_columns": {"time": t_col, "u": u_col, "bis": bis_col, "tof": tof_col},
        "patient_characteristic": {
            "age": float(args.age),
            "height": float(args.height),
            "weight": float(args.weight),
            "sex": int(args.sex),
        },
        "sampling_time_s": float(ts),
        "fit": {
            "c50": float(fit.c50),
            "success": bool(fit.success),
            "cost": float(fit.cost),
            "nfev": int(fit.nfev),
            "metrics": fit_summary,
        },
        "closed_loop": closed_loop_summary,
    }
    save_json(summary, output_dir / "summary.json")

    return {
        "fit": fit,
        "fit_df": fit_df,
        "closed_loop_df": closed_loop_df,
        "output_dir": output_dir,
        "resolved_columns": {"time": t_col, "u": u_col, "bis": bis_col, "tof": tof_col},
        "ts": ts,
        "summary": summary,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PKPD C50 fitting + PID control pipeline.")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV.")
    parser.add_argument("--output-dir", type=str, default="outputs/pkpd_pid", help="Directory for outputs.")

    # Optional column mappings.
    parser.add_argument("--time-col", type=str, default=None, help="Time column name.")
    parser.add_argument("--u-col", type=str, default=None, help="Input column name u(t).")
    parser.add_argument("--bis-col", type=str, default=None, help="BIS column name.")
    parser.add_argument("--tof-col", type=str, default=None, help="TOF column name.")

    # Patient demographics.
    parser.add_argument("--age", type=float, required=True, help="Age (yr).")
    parser.add_argument("--height", type=float, required=True, help="Height (cm).")
    parser.add_argument("--weight", type=float, required=True, help="Weight (kg).")
    parser.add_argument("--sex", type=int, choices=[0, 1], required=True, help="Sex: 0 female, 1 male.")
    parser.add_argument("--ts", type=float, default=None, help="Sampling time in seconds. If omitted, infer from CSV.")

    # BIS fit parameters (only C50 is fitted).
    parser.add_argument("--gamma-bis", type=float, default=2.69, help="Fixed BIS gamma during C50 fitting.")
    parser.add_argument("--e0-bis", type=float, default=95.9, help="Fixed BIS E0 during C50 fitting.")
    parser.add_argument("--emax-bis", type=float, default=87.5, help="Fixed BIS Emax during C50 fitting.")
    parser.add_argument("--c50-init", type=float, default=4.5, help="Initial C50 guess.")
    parser.add_argument("--c50-lb", type=float, default=0.1, help="Lower bound for C50.")
    parser.add_argument("--c50-ub", type=float, default=20.0, help="Upper bound for C50.")

    # TOF model params.
    parser.add_argument("--tof-c50", type=float, default=0.625, help="TOF model C50.")
    parser.add_argument("--tof-gamma", type=float, default=4.25, help="TOF model gamma.")

    # Closed-loop targets.
    parser.add_argument("--control-duration", type=float, default=1800.0, help="Closed-loop duration in seconds.")
    parser.add_argument("--bis-target", type=float, default=50.0, help="BIS target.")
    parser.add_argument("--tof-target", type=float, default=10.0, help="TOF target (used only if TOF control enabled).")

    # BIS PID.
    parser.add_argument("--kp-bis", type=float, default=0.08, help="BIS PID Kp.")
    parser.add_argument("--ki-bis", type=float, default=0.002, help="BIS PID Ki.")
    parser.add_argument("--kd-bis", type=float, default=0.0, help="BIS PID Kd.")
    parser.add_argument("--u-propo-min", type=float, default=0.0, help="Propofol command minimum.")
    parser.add_argument("--u-propo-max", type=float, default=6.67, help="Propofol command maximum.")

    # TOF PID (optional).
    parser.add_argument("--enable-tof-control", action="store_true", help="Enable atracurium PID control for TOF.")
    parser.add_argument("--kp-tof", type=float, default=0.04, help="TOF PID Kp.")
    parser.add_argument("--ki-tof", type=float, default=0.001, help="TOF PID Ki.")
    parser.add_argument("--kd-tof", type=float, default=0.0, help="TOF PID Kd.")
    parser.add_argument("--u-atra-min", type=float, default=0.0, help="Atracurium command minimum.")
    parser.add_argument("--u-atra-max", type=float, default=20.0, help="Atracurium command maximum.")

    # Debug / diagnostics.
    parser.add_argument("--debug-print-every", type=int, default=0, help="Print one closed-loop debug line every N steps. 0 disables.")
    parser.add_argument("--bis-band", type=float, default=5.0, help="BIS target band used in summary and plots.")
    parser.add_argument("--tof-band", type=float, default=5.0, help="TOF target band used in summary and plots.")
    parser.add_argument("--settle-window", type=float, default=120.0, help="Hold window in seconds used for settling-time estimation.")
    return parser


def main_cli() -> None:
    """Advanced CLI with full parameter control."""
    parser = build_arg_parser()
    args = parser.parse_args()
    result = run_pipeline(args)
    fit = result["fit"]
    summary = result["summary"]
    print("===== Pipeline done =====")
    print(f"resolved columns: {result['resolved_columns']}")
    print(f"sampling time ts: {result['ts']:.3f} s")
    print(f"fitted C50: {fit.c50:.4f}")
    print(f"least_squares success: {fit.success}, cost: {fit.cost:.6f}, nfev: {fit.nfev}")
    print(f"fit RMSE: {summary['fit']['metrics']['rmse']:.4f}")
    print(f"closed-loop BIS settling time: {summary['closed_loop']['bis']['settling_time_s']}")
    print(f"outputs saved to: {result['output_dir']}")


def main() -> None:
    """Clean main: minimal setup, no long argument list.

    Usage:
      1) Edit the config block below and run directly.
      2) Or pass only CSV path as first argument:
         python scripts/pkpd_pid_pipeline.py /path/to/data.csv
    """
    csv_from_argv = sys.argv[1] if len(sys.argv) > 1 else "data/patient_timeseries.csv"

    # ===== Minimal user config block =====
    cfg = SimpleMainConfig(
        csv=csv_from_argv,
        age=40,
        height=170,
        weight=70,
        sex=1,  # 0: female, 1: male
        output_dir="outputs/pkpd_pid",
        control_duration=1800.0,
        bis_target=50.0,
        enable_tof_control=True,
        tof_target=10.0,
        debug_print_every=0,
    )
    # =====================================

    args = _namespace_from_simple_config(cfg)
    result = run_pipeline(args)
    fit = result["fit"]
    summary = result["summary"]
    print("===== Pipeline done (clean main) =====")
    print(f"resolved columns: {result['resolved_columns']}")
    print(f"sampling time ts: {result['ts']:.3f} s")
    print(f"fitted C50: {fit.c50:.4f}")
    print(f"least_squares success: {fit.success}, cost: {fit.cost:.6f}, nfev: {fit.nfev}")
    print(f"fit RMSE: {summary['fit']['metrics']['rmse']:.4f}")
    print(f"closed-loop BIS settling time: {summary['closed_loop']['bis']['settling_time_s']}")
    print(f"outputs saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
