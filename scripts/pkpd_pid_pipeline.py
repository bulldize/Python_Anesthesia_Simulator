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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def _import_pas_modules():
    """Import PAS modules with a source-tree fallback."""
    try:
        from python_anesthesia_simulator.pk_models import CompartmentModel, AtracuriumModel
        from python_anesthesia_simulator.pd_models import BIS_model, TOF_model
    except ImportError:
        project_root = Path(__file__).resolve().parents[1]
        src_path = project_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        from python_anesthesia_simulator.pk_models import CompartmentModel, AtracuriumModel
        from python_anesthesia_simulator.pd_models import BIS_model, TOF_model
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

    def step(self, setpoint: float, measurement: float) -> float:
        error = (measurement - setpoint) if self.reverse_acting else (setpoint - measurement)
        if not self.initialized:
            self.prev_error = error
            self.initialized = True

        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        u_raw = self.kp * error + self.ki * self.integral + self.kd * derivative
        u_sat = float(np.clip(u_raw, self.u_min, self.u_max))

        # Anti-windup: stop integrating when saturated in the same direction.
        if (u_raw > self.u_max and error > 0) or (u_raw < self.u_min and error < 0):
            self.integral -= error * self.dt

        self.prev_error = error
        return u_sat


def run_closed_loop_control(
    patient: FittedVirtualPatient,
    duration_s: float,
    bis_target: float,
    pid_bis: PIDController,
    tof_target: Optional[float] = None,
    pid_tof: Optional[PIDController] = None,
) -> pd.DataFrame:
    """Run PID closed loop and return trajectory dataframe."""
    n_steps = int(np.ceil(duration_s / patient.ts))
    records: list[dict[str, float]] = []

    y_bis = float(patient.last["bis"])
    y_tof = float(patient.last["tof"])
    for _ in range(n_steps):
        u_propo = pid_bis.step(setpoint=bis_target, measurement=y_bis)
        if pid_tof is not None and tof_target is not None:
            u_atra = pid_tof.step(setpoint=tof_target, measurement=y_tof)
        else:
            u_atra = 0.0

        out = patient.step(u_propo=u_propo, u_atra=u_atra)
        records.append(out)
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
        }
    )
    fit_df.to_csv(output_dir / "fit_result.csv", index=False)
    plot_fit_result(time_s=time_s, bis_meas=bis, bis_pred=bis_pred, out_path=output_dir / "fit_result.png")

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

    return {
        "fit": fit,
        "fit_df": fit_df,
        "closed_loop_df": closed_loop_df,
        "output_dir": output_dir,
        "resolved_columns": {"time": t_col, "u": u_col, "bis": bis_col, "tof": tof_col},
        "ts": ts,
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
    return parser


def main_cli() -> None:
    """Advanced CLI with full parameter control."""
    parser = build_arg_parser()
    args = parser.parse_args()
    result = run_pipeline(args)
    fit = result["fit"]
    print("===== Pipeline done =====")
    print(f"resolved columns: {result['resolved_columns']}")
    print(f"sampling time ts: {result['ts']:.3f} s")
    print(f"fitted C50: {fit.c50:.4f}")
    print(f"least_squares success: {fit.success}, cost: {fit.cost:.6f}, nfev: {fit.nfev}")
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
    )
    # =====================================

    args = _namespace_from_simple_config(cfg)
    result = run_pipeline(args)
    fit = result["fit"]
    print("===== Pipeline done (clean main) =====")
    print(f"resolved columns: {result['resolved_columns']}")
    print(f"sampling time ts: {result['ts']:.3f} s")
    print(f"fitted C50: {fit.c50:.4f}")
    print(f"least_squares success: {fit.success}, cost: {fit.cost:.6f}, nfev: {fit.nfev}")
    print(f"outputs saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
