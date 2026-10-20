#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import argparse

from paraview.simple import PVDReader, XMLUnstructuredGridReader, ResampleWithDataset, UpdatePipeline
from paraview import servermanager
from vtk.util.numpy_support import vtk_to_numpy


def fetch_points_and_point_array(proxy, name: str):
    data = servermanager.Fetch(proxy)
    pts = vtk_to_numpy(data.GetPoints().GetData()).astype(np.float64, copy=False)
    pd = data.GetPointData()
    arr = pd.GetArray(name)
    if arr is None:
        keys = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
        raise RuntimeError(f"Point array not found: {name}. Available: {keys}")
    vals = vtk_to_numpy(arr).astype(np.float64, copy=False).reshape(-1)
    return pts, vals


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


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


def resolve_output_root(out_root_arg: str | None, case_name: str | None, config: dict[str, object]) -> Path:
    if out_root_arg:
        return Path(out_root_arg).expanduser().resolve()
    local_output_dir = config_path(config, "local_output_dir")
    parts = ["postprocess"]
    if case_name:
        parts.append(case_name)
    parts.append("resample")
    if local_output_dir:
        return local_output_dir.joinpath(*parts)
    return repo_root().joinpath("outputs", *parts)


def resample_series(merged_pvd: Path, target_vtu: Path, out_dir: Path, head_name: str, rho: float, g: float):
    ensure_dir(out_dir)

    src = PVDReader(FileName=str(merged_pvd))
    tgt = XMLUnstructuredGridReader(FileName=[str(target_vtu)])
    rs = ResampleWithDataset(SourceDataArrays=src, DestinationMesh=tgt)

    timesteps = list(getattr(src, "TimestepValues", []))
    if not timesteps:
        raise RuntimeError("No timesteps in merged pvd")

    UpdatePipeline(time=timesteps[0], proxy=rs)
    pts0, h0 = fetch_points_and_point_array(rs, head_name)
    npts = pts0.shape[0]
    nt = len(timesteps)

    np.save(out_dir / "points.npy", pts0)
    np.save(out_dir / "time.npy", np.asarray(timesteps, dtype=np.float64))

    head_mm = np.lib.format.open_memmap(str(out_dir / "hydraulic_head.npy"), mode="w+", dtype=np.float32, shape=(nt, npts))
    pres_mm = np.lib.format.open_memmap(str(out_dir / "pressure.npy"), mode="w+", dtype=np.float32, shape=(nt, npts))

    scale = float(rho) * float(g)

    for it, t in enumerate(timesteps):
        UpdatePipeline(time=t, proxy=rs)
        pts, h = fetch_points_and_point_array(rs, head_name)

        if pts.shape[0] != npts:
            raise RuntimeError("Point count changed across timesteps")

        head_mm[it, :] = h.astype(np.float32, copy=False)
        pres_mm[it, :] = (scale * h).astype(np.float32, copy=False)

        if (it + 1) % 20 == 0 or it == 0 or it == nt - 1:
            print(f"[INFO] {out_dir.name} {it+1}/{nt} time {t}")

    head_mm.flush()
    pres_mm.flush()
    print(f"[DONE] wrote {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Optional YAML config with local path placeholders")
    ap.add_argument("--merged_pvd", required=True)
    ap.add_argument("--fault_vtu", required=True)
    ap.add_argument("--leduc_vtu", required=True)
    ap.add_argument("--case_name", default=None, help="Optional case name used for default output layout")
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--head_name", default="hydraulic_head")
    ap.add_argument("--rho", type=float, default=1000.0)
    ap.add_argument("--g", type=float, default=9.80665)
    args = ap.parse_args()

    config = load_config(args.config)

    merged_pvd = Path(args.merged_pvd).resolve()
    fault_vtu = Path(args.fault_vtu).resolve()
    leduc_vtu = Path(args.leduc_vtu).resolve()
    out_root = resolve_output_root(args.out_root, args.case_name, config)
    ensure_dir(out_root)

    resample_series(merged_pvd, fault_vtu, out_root / "fault", args.head_name, args.rho, args.g)
    resample_series(merged_pvd, leduc_vtu, out_root / "leduc_top", args.head_name, args.rho, args.g)

if __name__ == "__main__":
    main()
