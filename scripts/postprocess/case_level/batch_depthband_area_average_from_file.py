#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def weighted_mean(ts_by_node: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    x = np.asarray(ts_by_node, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("ts_by_node must be a 2D array with shape (nt, n_nodes)")
    if x.shape[1] != w.shape[0]:
        raise ValueError("weight length does not match node dimension")

    denom = np.sum(w)
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("sum of weights must be positive")

    return (x @ w) / denom


def read_bands_csv(path: Path) -> list[tuple[float, float]]:
    df = pd.read_csv(path)
    if "zmin_km" not in df.columns or "zmax_km" not in df.columns:
        raise ValueError("bands file must have columns zmin_km and zmax_km")

    bands: list[tuple[float, float]] = []
    for _, row in df.iterrows():
        z1 = float(row["zmin_km"])
        z2 = float(row["zmax_km"])
        bands.append((min(z1, z2), max(z1, z2)))

    if not bands:
        raise ValueError("no bands found in bands file")
    return bands


def compute_backward_rate(series: np.ndarray, time: np.ndarray, first_value: float) -> np.ndarray:
    s = np.asarray(series, dtype=np.float64).reshape(-1)
    t = np.asarray(time, dtype=np.float64).reshape(-1)

    if s.shape[0] != t.shape[0]:
        raise ValueError("series and time must have the same length")

    nt = s.shape[0]
    rate = np.empty(nt, dtype=np.float64)
    if nt == 0:
        return rate
    if nt == 1:
        rate[0] = first_value
        return rate

    dt = np.diff(t)
    if np.any(~np.isfinite(dt)) or np.any(dt <= 0.0):
        raise ValueError("time array must be finite and strictly increasing")

    rate[0] = first_value
    rate[1:] = np.diff(s) / dt
    return rate


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def load_config(config_path: str | None) -> dict[str, object]:
    if not config_path:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required when --config is used") from exc

    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise RuntimeError(f"Config file must contain a YAML mapping: {path}")
    return config


def config_path(config: dict[str, object], key: str) -> Path | None:
    value = config.get(key)
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value)).expanduser().resolve()


def resolve_local_postprocess_dir(
    path_arg: str | None,
    case_name: str | None,
    config: dict[str, object],
    leaf: str,
    allow_repo_default: bool,
    required_message: str,
) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    local_output_dir = config_path(config, "local_output_dir")
    parts = ["postprocess"]
    if case_name:
        parts.append(case_name)
    parts.append(leaf)
    if local_output_dir:
        return local_output_dir.joinpath(*parts)
    if allow_repo_default:
        return repo_root().joinpath("outputs", *parts)
    raise RuntimeError(required_message)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compute depth band area averages on the fault and define pressure rate "
            "from the band mean pressure using a backward difference in time."
        )
    )
    ap.add_argument("--config", default=None, help="Optional YAML config with local path placeholders")
    ap.add_argument("--case_name", default=None, help="Optional case name used for local_output_dir defaults")
    ap.add_argument("--bands_file", required=True, help="CSV file with columns zmin_km and zmax_km")
    ap.add_argument("--fault_dir", default=None)
    ap.add_argument("--dieterich_dir", default=None)
    ap.add_argument("--chi_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--pressure_unit", choices=["Pa", "MPa"], default="MPa")
    ap.add_argument("--time_axis", choices=["index", "month"], default="month")
    ap.add_argument(
        "--first_rate_value",
        type=float,
        default=0.0,
        help=(
            "Value assigned to pressure_rate at the first timestep, where a backward "
            "difference is undefined. Default is 0.0."
        ),
    )
    args = ap.parse_args()

    config = load_config(args.config)

    bands_file = Path(args.bands_file).expanduser().resolve()
    fault_dir = resolve_local_postprocess_dir(
        args.fault_dir,
        args.case_name,
        config,
        "resample/fault",
        False,
        "--fault_dir is required, or pass --config with local_output_dir.",
    )
    dieterich_dir = resolve_local_postprocess_dir(
        args.dieterich_dir,
        args.case_name,
        config,
        "dieterich",
        False,
        "--dieterich_dir is required, or pass --config with local_output_dir.",
    )
    chi_dir = resolve_local_postprocess_dir(
        args.chi_dir,
        args.case_name,
        config,
        "chi",
        False,
        "--chi_dir is required, or pass --config with local_output_dir.",
    )
    out_dir = resolve_local_postprocess_dir(
        args.out_dir,
        args.case_name,
        config,
        "band_stats",
        True,
        "--out_dir is required, or pass --config with local_output_dir.",
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    bands = read_bands_csv(bands_file)

    time = np.load(dieterich_dir / "time.npy").astype(np.float64).reshape(-1)
    depth_km = np.load(dieterich_dir / "depth_km.npy").astype(np.float64).reshape(-1)
    weights = np.load(dieterich_dir / "weights.npy").astype(np.float64).reshape(-1)

    pressure = np.load(fault_dir / "pressure.npy", mmap_mode="r")
    R = np.load(dieterich_dir / "R.npy", mmap_mode="r")
    chi = np.load(chi_dir / "chi.npy", mmap_mode="r")
    chiR = np.load(chi_dir / "chiR.npy", mmap_mode="r")

    nt = time.shape[0]
    n_nodes = depth_km.shape[0]

    for name, arr in [("pressure", pressure), ("R", R), ("chi", chi), ("chiR", chiR)]:
        if arr.shape != (nt, n_nodes):
            raise RuntimeError(f"{name} shape mismatch. got {arr.shape} expected {(nt, n_nodes)}")

    if args.time_axis == "month":
        x = time.copy()
        x_label = "time in months"
    else:
        x = np.arange(nt, dtype=float)
        x_label = "timestep index"

    p_scale = 1.0
    p_label = "pressure in Pa"
    dp_label = "pressure rate in Pa per month"
    if args.pressure_unit == "MPa":
        p_scale = 1.0e6
        p_label = "pressure in MPa"
        dp_label = "pressure rate in MPa per month"

    series: dict[tuple[float, float], dict[str, np.ndarray]] = {}

    for zmin, zmax in bands:
        mask = (
            (depth_km >= zmin)
            & (depth_km <= zmax)
            & np.isfinite(depth_km)
            & np.isfinite(weights)
            & (weights > 0.0)
        )
        n_sel = int(np.sum(mask))
        if n_sel == 0:
            print(f"[WARN] no nodes in band {zmin} to {zmax} km. skipped")
            continue

        w_sel = weights[mask].copy()

        p_band = weighted_mean(np.asarray(pressure[:, mask]), w_sel)
        dpdt_band = compute_backward_rate(p_band, time, args.first_rate_value)
        R_band = weighted_mean(np.asarray(R[:, mask]), w_sel)
        chi_band = weighted_mean(np.asarray(chi[:, mask]), w_sel)
        chiR_band = weighted_mean(np.asarray(chiR[:, mask]), w_sel)

        tag = f"z_{zmin:.3f}_{zmax:.3f}_km"

        df = pd.DataFrame(
            {
                "month_index": np.arange(nt, dtype=int),
                "time_month": time,
                "pressure_Pa": p_band,
                "pressure_rate_Pa_per_month": dpdt_band,
                "R": R_band,
                "chi": chi_band,
                "chiR": chiR_band,
                "zmin_km": zmin,
                "zmax_km": zmax,
                "n_nodes": n_sel,
                "sum_weights": float(np.sum(w_sel)),
                "pressure_rate_method": "backward_difference_of_area_mean_pressure",
            }
        )
        csv_path = out_dir / f"area_average_{tag}.csv"
        df.to_csv(csv_path, index=False)

        np.save(out_dir / f"pressure_{tag}.npy", p_band.astype(np.float64))
        np.save(out_dir / f"pressure_rate_{tag}.npy", dpdt_band.astype(np.float64))
        np.save(out_dir / f"R_{tag}.npy", R_band.astype(np.float64))
        np.save(out_dir / f"chi_{tag}.npy", chi_band.astype(np.float64))
        np.save(out_dir / f"chiR_{tag}.npy", chiR_band.astype(np.float64))

        plt.figure()
        plt.plot(x, p_band / p_scale)
        plt.xlabel(x_label)
        plt.ylabel(p_label)
        plt.title(f"Area mean pressure  {zmin} to {zmax} km")
        plt.tight_layout()
        plt.savefig(out_dir / f"pressure_{tag}.png", dpi=200)
        plt.close()

        plt.figure()
        plt.plot(x, dpdt_band / p_scale)
        plt.xlabel(x_label)
        plt.ylabel(dp_label)
        plt.title(f"Area mean pressure rate  {zmin} to {zmax} km")
        plt.tight_layout()
        plt.savefig(out_dir / f"pressure_rate_{tag}.png", dpi=200)
        plt.close()

        plt.figure()
        plt.plot(x, R_band)
        plt.xlabel(x_label)
        plt.ylabel("R")
        plt.title(f"Area mean R  {zmin} to {zmax} km")
        plt.tight_layout()
        plt.savefig(out_dir / f"R_{tag}.png", dpi=200)
        plt.close()

        plt.figure()
        plt.plot(x, chi_band)
        plt.xlabel(x_label)
        plt.ylabel("chi")
        plt.title(f"Area mean chi  {zmin} to {zmax} km")
        plt.tight_layout()
        plt.savefig(out_dir / f"chi_{tag}.png", dpi=200)
        plt.close()

        plt.figure()
        plt.plot(x, chiR_band)
        plt.xlabel(x_label)
        plt.ylabel("chiR")
        plt.title(f"Area mean chiR  {zmin} to {zmax} km")
        plt.tight_layout()
        plt.savefig(out_dir / f"chiR_{tag}.png", dpi=200)
        plt.close()

        series[(zmin, zmax)] = {
            "p": p_band / p_scale,
            "dp": dpdt_band / p_scale,
            "R": R_band,
            "chi": chi_band,
            "chiR": chiR_band,
        }

        print(f"[OK] band {zmin} to {zmax} km  nodes {n_sel}  wrote {csv_path.name}")

    if not series:
        raise RuntimeError("no band produced outputs")

    def overlay(key: str, ylabel: str, fname: str) -> None:
        plt.figure()
        for (zmin, zmax), d in series.items():
            plt.plot(x, d[key], label=f"{zmin} to {zmax} km")
        plt.xlabel(x_label)
        plt.ylabel(ylabel)
        plt.title(f"Area mean comparison by depth bands  {key}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=200)
        plt.close()

    overlay("p", p_label, "overlay_pressure.png")
    overlay("dp", dp_label, "overlay_pressure_rate.png")
    overlay("R", "R", "overlay_R.png")
    overlay("chi", "chi", "overlay_chi.png")
    overlay("chiR", "chiR", "overlay_chiR.png")

    print("")
    print("[DONE] outputs in", out_dir)


if __name__ == "__main__":
    main()
