#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import xml.etree.ElementTree as ET


PVTU_RE = re.compile(r"^result_ts_(\d+)\.pvtu$")


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


def resolve_result_dir(result_dir_arg: str | None, case_name: str | None, config: dict[str, object]) -> Path:
    if result_dir_arg:
        return Path(result_dir_arg).expanduser().resolve()
    raw_root = config_path(config, "ogs_results_raw_dir")
    if raw_root and case_name:
        return raw_root / case_name
    if raw_root:
        return raw_root
    raise RuntimeError(
        "--result_dir is required, or pass --config with ogs_results_raw_dir "
        "and optionally --case_name."
    )


def resolve_merged_dir(merged_dir_arg: str | None, case_name: str | None, config: dict[str, object]) -> Path:
    if merged_dir_arg:
        return Path(merged_dir_arg).expanduser().resolve()
    local_output_dir = config_path(config, "local_output_dir")
    parts = ["postprocess"]
    if case_name:
        parts.append(case_name)
    parts.append("merged")
    if local_output_dir:
        return local_output_dir.joinpath(*parts)
    return repo_root().joinpath("outputs", *parts)


def find_pvtu2vtu(explicit_path: str | None, config: dict[str, object]) -> str:
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"pvtu2vtu not found at {explicit_path}")
        return str(p)

    config_executable = config_path(config, "pvtu2vtu")
    if config_executable and config_executable.exists():
        return str(config_executable)

    env = os.environ.get("PVTU2VTU", "").strip()
    if env:
        p = Path(env)
        if p.exists():
            return str(p)

    which = shutil.which("pvtu2vtu")
    if which:
        return which

    raise FileNotFoundError(
        "pvtu2vtu not found. Set --pvtu2vtu, set PVTU2VTU, or set pvtu2vtu in --config."
    )


def parse_time_map_from_pvd(pvd_path: Path) -> dict[int, float]:
    time_map: dict[int, float] = {}
    if not pvd_path.exists():
        return time_map

    tree = ET.parse(pvd_path)
    root = tree.getroot()

    for ds in root.findall(".//DataSet"):
        file_attr = ds.attrib.get("file", "")
        time_attr = ds.attrib.get("timestep", None)
        if time_attr is None:
            continue

        m = PVTU_RE.match(Path(file_attr).name)
        if not m:
            continue

        ts = int(m.group(1))
        try:
            t = float(time_attr)
        except ValueError:
            continue
        time_map[ts] = t

    return time_map


def merge_one(pvtu2vtu: str, pvtu_path: Path, out_path: Path) -> tuple[int, bool, str]:
    ts_m = PVTU_RE.match(pvtu_path.name)
    if not ts_m:
        return (-1, False, f"Unexpected pvtu name {pvtu_path.name}")

    ts = int(ts_m.group(1))
    cmd = [pvtu2vtu, "-i", str(pvtu_path), "-o", str(out_path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as e:
        return (ts, False, f"Failed to run pvtu2vtu. {e}")

    if r.returncode != 0:
        msg = (r.stdout or "") + "\n" + (r.stderr or "")
        return (ts, False, msg.strip())

    return (ts, True, "")


def write_merged_pvd(merged_dir: Path, merged_files: list[tuple[int, Path]], time_map: dict[int, float]) -> Path:
    pvd_path = merged_dir / "merged_result.pvd"
    root = ET.Element("VTKFile", attrib={"type": "Collection", "version": "0.1", "byte_order": "LittleEndian"})
    col = ET.SubElement(root, "Collection")

    for ts, fpath in merged_files:
        t = time_map.get(ts, float(ts))
        rel = fpath.name
        ET.SubElement(col, "DataSet", attrib={"timestep": f"{t}", "group": "", "part": "0", "file": rel})

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(pvd_path, encoding="utf-8", xml_declaration=True)
    return pvd_path


def delete_originals(result_dir: Path, timesteps: list[int], dry_run: bool) -> None:
    for ts in timesteps:
        pvtu = result_dir / f"result_ts_{ts}.pvtu"

        patterns = [
            f"result_ts_{ts}_*.vtu",
            f"result_ts_{ts}.vtu",
        ]

        to_delete = []
        if pvtu.exists():
            to_delete.append(pvtu)

        for pat in patterns:
            to_delete.extend(result_dir.glob(pat))

        uniq = []
        seen = set()
        for p in to_delete:
            if p not in seen:
                seen.add(p)
                uniq.append(p)

        if not uniq:
            continue

        print(f"[DELETE] timestep {ts}")
        for p in uniq:
            if dry_run:
                print(f"  would delete {p}")
            else:
                try:
                    p.unlink()
                    print(f"  deleted {p}")
                except Exception as e:
                    print(f"  failed to delete {p}. {e}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Optional YAML config with local path placeholders")
    parser.add_argument("--result_dir", default=None, help="Input result directory that contains result_ts_*.pvtu")
    parser.add_argument("--case_name", default=None, help="Optional case directory name under ogs_results_raw_dir")
    parser.add_argument("--merged_dir", default=None, help="Output directory for merged vtus")
    parser.add_argument("--pvtu2vtu", default=None, help="Path to pvtu2vtu. If omitted, it will be auto detected")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers. Default is min(8, CPU count)")
    parser.add_argument("--dry_run", action="store_true", help="Do not run pvtu2vtu and do not delete files")
    parser.add_argument("--keep_original", action="store_true", help="Do not delete pvtu and rank vtus after merge")
    args = parser.parse_args()

    config = load_config(args.config)

    result_dir = resolve_result_dir(args.result_dir, args.case_name, config)
    merged_dir = resolve_merged_dir(args.merged_dir, args.case_name, config)
    merged_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print("[INFO] result_dir", result_dir)
    print("[INFO] merged_dir", merged_dir)
    print("")

    pvtu2vtu = find_pvtu2vtu(args.pvtu2vtu, config)
    print("[INFO] pvtu2vtu", pvtu2vtu)
    print("")

    pvtu_files = []
    for p in result_dir.glob("result_ts_*.pvtu"):
        m = PVTU_RE.match(p.name)
        if m:
            pvtu_files.append((int(m.group(1)), p))

    pvtu_files.sort(key=lambda x: x[0])

    if not pvtu_files:
        print("[ERROR] No result_ts_*.pvtu found in result_dir")
        return 2

    print(f"[INFO] Found {len(pvtu_files)} pvtu files")
    print(f"[INFO] First timestep {pvtu_files[0][0]}  Last timestep {pvtu_files[-1][0]}")
    print("")

    time_map = parse_time_map_from_pvd(result_dir / "result.pvd")

    workers = args.workers
    if workers is None:
        cpu = os.cpu_count() or 1
        workers = min(8, cpu)
    workers = max(1, workers)

    merged = []
    failed = []

    if args.dry_run:
        print("[DRY RUN] No pvtu2vtu execution. No deletion.")
        for ts, pvtu in pvtu_files:
            out = merged_dir / f"merged_result_ts_{ts}.vtu"
            print(f"  would merge {pvtu.name} -> {out.name}")
            merged.append((ts, out))
    else:
        print(f"[INFO] Merging with {workers} workers")
        print("")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = []
            for ts, pvtu in pvtu_files:
                out = merged_dir / f"merged_result_ts_{ts}.vtu"
                futs.append(ex.submit(merge_one, pvtu2vtu, pvtu, out))

            for fut in as_completed(futs):
                ts, ok, msg = fut.result()
                if ok:
                    merged.append((ts, merged_dir / f"merged_result_ts_{ts}.vtu"))
                    print(f"[OK] merged timestep {ts}")
                else:
                    failed.append((ts, msg))
                    print(f"[NG] timestep {ts}")
                    print(msg)
                    print("")

    merged.sort(key=lambda x: x[0])

    if failed:
        print("")
        print("[ERROR] Some timesteps failed. No deletion will be performed.")
        print(f"[ERROR] Failed count {len(failed)}")
        print("")
        return 3

    pvd_path = write_merged_pvd(merged_dir, merged, time_map)
    print("")
    print("[INFO] Wrote merged pvd", pvd_path)
    print("")

    if not args.keep_original and not args.dry_run:
        timesteps = [ts for ts, _ in pvtu_files]
        delete_originals(result_dir, timesteps, dry_run=False)
        print("")
        print("[INFO] Deleted original pvtu and rank vtus")
        print("")

    print("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
