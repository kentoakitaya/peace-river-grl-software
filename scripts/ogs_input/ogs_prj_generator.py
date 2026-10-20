#!/usr/bin/env python3
"""
generate_ogs_prj_petsc_dirs_v13_prepare.py

Goals
  Generate an OpenGeoSys project file (ogs_project.prj) from a JSON input file.

Notes
- In JSON, each well entry requires `vtu_file`.
- Injection schedules can be provided either as pre-generated `coords_bin` and
  `values_bin` files, or as a source `flow_file` text file that is converted to
  binaries during staging.
- If `coords_bin` and `values_bin` are present for every well, the original
  schedule text files are not required.
- well_id is not required.
- Optionally partition meshes for PETSc MPI workflows.
- This version does not run OGS. Use sbatch with srun for execution.
- Provide safer directory handling and support a dedicated mesh_dir and output_dir.

Key ideas
  - aux_bin_dir is treated as a generated intermediate directory for curve binaries.
  - mesh_dir is treated as a run-time directory for partitioned or staged meshes.
  - output_dir can be used with ogs -o so that .pvd and .pvtu paths are consistent.

This script is designed to be used in a case/run directory.
"""
from __future__ import annotations

import argparse
import json
import ast
import os
import glob
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import meshio


# -----------------------------
# Small helpers
# -----------------------------
def _parse_bool(value) -> Optional[bool]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s == "":
        return None
    if s in ["1", "true", "t", "yes", "y", "on"]:
        return True
    if s in ["0", "false", "f", "no", "n", "off"]:
        return False
    return None


def _shell_quote(s) -> str:
    if s is None:
        return ""
    s = str(s)
    if re.search(r"\s", s):
        return '"' + s.replace('"', r"\"") + '"'
    return s


def _resolve_int_setting(cli_value, general_settings, key, default) -> int:
    if cli_value is not None:
        return int(cli_value)
    raw = str(general_settings.get(key, "")).strip()
    if raw == "":
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _pick_setting(general_settings, key, fallback):
    val = str(general_settings.get(key, "")).strip()
    return val if val != "" else fallback


def _norm_slash(p: str) -> str:
    return str(p).replace("\\", "/")


def _resolve_relative_path(base_dir: str, path_value: str) -> str:
    path_str = str(path_value or "").strip()
    if path_str == "" or os.path.isabs(path_str):
        return path_value
    return _norm_slash(os.path.normpath(os.path.join(base_dir, path_str)))


def repo_root() -> str:
    return str(Path(__file__).resolve().parents[3])


def load_config(config_path: Optional[str]) -> Dict[str, object]:
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


def config_path(config: Dict[str, object], key: str) -> Optional[Path]:
    value = config.get(key)
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value)).expanduser().resolve()


def local_output_path(config: Dict[str, object], *parts: str) -> Path:
    local_output_dir = config_path(config, "local_output_dir")
    root = local_output_dir if local_output_dir is not None else Path(repo_root()) / "outputs"
    return root.joinpath(*parts)


def old_workspace_path(config: Dict[str, object], *parts: str) -> Optional[Path]:
    old_workspace_root = config_path(config, "old_workspace_root")
    if old_workspace_root is None:
        return None
    return old_workspace_root.joinpath(*parts)


def _basename_join(dir_path: str, original_path: str) -> str:
    if (dir_path is None) or str(dir_path).strip() == "":
        return original_path
    if (original_path is None) or str(original_path).strip() == "":
        return original_path
    return _norm_slash(os.path.join(str(dir_path), os.path.basename(str(original_path))))


def apply_source_dir_overrides(
    *,
    mesh_src_dir: Optional[str],
    injection_src_dir: Optional[str],
    injection_bin_src_dir: Optional[str],
    general_settings: Dict[str, object],
    mesh_files: List[str],
    dirichlet_condition: List[Dict[str, str]],
    wells: List[Dict[str, str]],
    generate_injection_bins: bool,
) -> Tuple[List[str], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Override file references in base_json so that:
      - all VTU mesh files are read from mesh_src_dir/<basename>
      - injection schedule sources are read from either:
          - injection_bin_src_dir/<basename> for pre-generated curve binaries, if provided, and/or
          - injection_src_dir/<basename> for original schedule text files, if provided.

    This keeps case_base.json portable. Paths are controlled by CLI, typically from run_case.sh.

    Notes
      - This function does not create symlinks or copies. It only rewrites paths and prechecks existence.
      - Existence checks depend on generate_injection_bins:
          - If a well has coords_bin and values_bin, they are checked (preferred).
          - Otherwise, flow_file is checked only when generate_injection_bins is True.
    """
    msd = str(mesh_src_dir).strip() if mesh_src_dir else ""
    isd = str(injection_src_dir).strip() if injection_src_dir else ""
    ibsd = str(injection_bin_src_dir).strip() if injection_bin_src_dir else ""

    new_mesh_files = [_basename_join(msd, p) for p in mesh_files]

    new_dc: List[Dict[str, str]] = []
    for dc in dirichlet_condition or []:
        new_dc.append(
            {
                "vtu_file": _basename_join(msd, dc.get("vtu_file", "")),
                "field_name": dc.get("field_name", ""),
            }
        )

    new_wells: List[Dict[str, str]] = []
    for w in wells or []:
        w2 = dict(w) if isinstance(w, dict) else {}
        w2["vtu_file"] = _basename_join(msd, w2.get("vtu_file", ""))

        # Original schedule text (optional)
        if "flow_file" in w2:
            w2["flow_file"] = _basename_join(isd, w2.get("flow_file", ""))

        # Pre-generated curve binaries (optional)
        if "coords_bin" in w2:
            w2["coords_bin"] = _basename_join(ibsd or isd, w2.get("coords_bin", ""))
        if "values_bin" in w2:
            w2["values_bin"] = _basename_join(ibsd or isd, w2.get("values_bin", ""))

        new_wells.append(w2)

    # Optional override for global_mesh_file if present
    if msd != "" and "global_mesh_file" in general_settings:
        gm = str(general_settings.get("global_mesh_file", "")).strip()
        if gm != "":
            general_settings["global_mesh_file"] = _basename_join(msd, gm)

    # Precheck existence for overridden sources
    missing: List[str] = []

    def _check_exists(p: str) -> None:
        p = str(p or "").strip()
        if p == "":
            return
        if not os.path.exists(p):
            missing.append(p)

    if msd != "":
        for p in new_mesh_files:
            _check_exists(p)
        for dc in new_dc:
            _check_exists(dc.get("vtu_file", ""))
        for w in new_wells:
            _check_exists(w.get("vtu_file", ""))

    # Injection sources
    for w in new_wells:
        coords = str(w.get("coords_bin", "")).strip()
        values = str(w.get("values_bin", "")).strip()

        if coords != "" or values != "":
            # If either is specified, require both and check both.
            if coords == "" or values == "":
                missing.append(coords if coords == "" else values)
                continue
            _check_exists(coords)
            _check_exists(values)
            continue

        # No bins given. Only require flow_file when we plan to generate bins.
        if generate_injection_bins:
            flow = str(w.get("flow_file", "")).strip()
            if flow == "":
                missing.append(flow)
            else:
                _check_exists(flow)

    if missing:
        preview = "\n".join([p for p in missing if p][:20])
        raise ValueError(
            "Some required source files are missing after applying --mesh_src_dir / --injection_src_dir / --injection_bin_src_dir.\n"
            f"First missing entries\n{preview}"
        )

    return new_mesh_files, new_dc, new_wells


def _safe_mkdir(path: str) -> None:
    if path and path.strip() != "":
        os.makedirs(path, exist_ok=True)


def _output_parent_dir_from_prefix(prefix: str) -> Optional[str]:
    """
    Used only for optional clean-up in keep or clean mode.
    We explicitly reject risky paths.
    """
    p = str(prefix).strip()
    if p == "":
        return None
    d = os.path.dirname(p)
    if d == "":
        return None
    dn = os.path.normpath(d)
    if dn in [".", "/"]:
        return None
    if dn.startswith(".."):
        return None
    if os.path.isabs(dn):
        return None
    return d


def _prepare_dir_clean(mode: str, path: Optional[str], label: str) -> None:
    if path is None:
        return
    if mode != "clean":
        return
    # Extra safety gate: refuse to delete if it contains parent traversal.
    norm = os.path.normpath(path)
    if norm.startswith("..") or os.path.isabs(norm) or norm in [".", "/"]:
        raise ValueError(f"Refusing to clean unsafe {label} path: {path}")
    if os.path.isdir(norm):
        shutil.rmtree(norm)
    os.makedirs(norm, exist_ok=True)


def _mesh_name_from_token(token: str) -> str:
    s = str(token).strip()
    if s == "":
        return ""
    base = os.path.basename(s)
    name, _ = os.path.splitext(base)
    return name


def _ensure_meshname_in_prefix(prefix: str) -> str:
    if "{:meshname}" in prefix:
        return prefix
    d = os.path.dirname(prefix)
    b = os.path.basename(prefix)
    new_b = "{:meshname}_" + b
    return os.path.join(d, new_b) if d else new_b


def _collect_defined_mesh_names(mesh_files, global_mesh_file, dirichlet_condition):
    names = set()
    for p in mesh_files:
        names.add(_mesh_name_from_token(p))
    if dirichlet_condition:
        for dc in dirichlet_condition:
            names.add(_mesh_name_from_token(dc.get("vtu_file", "")))
    names.add(_mesh_name_from_token(global_mesh_file))
    return {n for n in names if n}


# -----------------------------
# Input parsing
# -----------------------------

def read_input_json(base_json: str, params_json: str | None):
    base_json_path = os.path.abspath(base_json)
    base_json_dir = os.path.dirname(base_json_path)

    with open(base_json_path, "r", encoding="utf-8") as f:
        base = json.load(f)

    general_settings = dict(base.get("general", {}))
    wells = list(base.get("wells", []))
    time_settings = dict(base.get("time", {}))
    media_list = list(base.get("media", []))
    mesh_files_base = [
        _resolve_relative_path(base_json_dir, p) if isinstance(p, str) else p
        for p in list(base.get("meshes", []))
    ]

    # Load other sections that may reference meshes
    initial_condition = dict(base.get("initial_condition", {}))
    dirichlet_condition = list(base.get("dirichlet_condition", []))
    output_meshes = list(base.get("output_meshes", []))
    process_settings = dict(base.get("process", {}))
    output_settings = dict(base.get("output", {}))

    # Collect referenced meshes from boundary conditions, wells, and output meshes,
    # to avoid duplication in JSON while still listing all meshes required by OGS.
    referenced = []
    for bc in dirichlet_condition:
        if isinstance(bc, dict) and "vtu_file" in bc:
            bc["vtu_file"] = _resolve_relative_path(base_json_dir, bc["vtu_file"])
            referenced.append(bc["vtu_file"])
    for well in wells:
        if isinstance(well, dict):
            if "vtu_file" in well:
                well["vtu_file"] = _resolve_relative_path(base_json_dir, well["vtu_file"])
                referenced.append(well["vtu_file"])
            if "flow_file" in well:
                well["flow_file"] = _resolve_relative_path(base_json_dir, well["flow_file"])
            if "coords_bin" in well:
                well["coords_bin"] = _resolve_relative_path(base_json_dir, well["coords_bin"])
            if "values_bin" in well:
                well["values_bin"] = _resolve_relative_path(base_json_dir, well["values_bin"])
    for i, om in enumerate(output_meshes):
        if isinstance(om, str):
            output_meshes[i] = _resolve_relative_path(base_json_dir, om)
            referenced.append(output_meshes[i])
        elif isinstance(om, dict) and "vtu_file" in om:
            om["vtu_file"] = _resolve_relative_path(base_json_dir, om["vtu_file"])
            referenced.append(om["vtu_file"])

    # Merge while preserving order and removing duplicates.
    mesh_files = []
    seen = set()
    for p in list(mesh_files_base) + referenced:
        if not p:
            continue
        if p in seen:
            continue
        seen.add(p)
        mesh_files.append(p)
    solver_settings = dict(base.get("solvers", {}))

    if "global_mesh_file" in general_settings:
        general_settings["global_mesh_file"] = _resolve_relative_path(
            base_json_dir, general_settings["global_mesh_file"]
        )

    if "output_xml" not in general_settings and "output_project_file" in general_settings:
        general_settings["output_xml"] = general_settings["output_project_file"]

    if params_json:
        with open(params_json, "r", encoding="utf-8") as f:
            params = json.load(f)

        updates = params.get("media_updates", {}) or {}
        idx = {m["name"]: i for i, m in enumerate(media_list) if isinstance(m, dict) and "name" in m}

        for name, uv in updates.items():
            if name not in idx:
                raise ValueError(f"Unknown medium name in media_updates: {name}")
            i = idx[name]
            if isinstance(uv, dict):
                if "perm" in uv:
                    media_list[i]["perm"] = float(uv["perm"])
                if "storage" in uv:
                    media_list[i]["storage"] = float(uv["storage"])

    return (
        general_settings,
        wells,
        time_settings,
        media_list,
        mesh_files,
        initial_condition,
        dirichlet_condition,
        process_settings,
        output_settings,
        output_meshes,
        solver_settings,
    )


def read_input_file(input_txt: str):
    with open(input_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    wells = []
    time_settings: Dict[str, object] = {}
    general_settings: Dict[str, str] = {}
    media_list = []
    meshes = []
    initial_condition = {}
    dirichlet_condition = []
    process_settings: Dict[str, str] = {}
    output_settings: Dict[str, str] = {}
    solver_settings: Dict[str, str] = {}
    output_meshes = []
    section = None

    for line in lines:
        line = line.strip()
        if line.startswith("["):
            section = line.strip("[]")
            continue
        if (not line) or line.startswith("#"):
            continue

        if section == "GENERAL":
            key, value = line.split("=", 1)
            general_settings[key.strip()] = value.strip()

        elif section == "WELLS":
            parts = re.split(r"[\s,\t]+", line)
            wells.append(
                {"well_id": parts[0].strip(), "vtu_file": parts[1].strip(), "flow_file": parts[2].strip()}
            )

        elif section == "TIME":
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            try:
                time_settings[key] = ast.literal_eval(value)
            except Exception:
                time_settings[key] = value

        elif section == "MEDIA":
            parts = re.split(r"[\s,\t]+", line)
            media_list.append(
                {
                    "name": parts[0],
                    "id": int(parts[1]),
                    "perm": float(parts[2]),
                    "poro": float(parts[3]),
                    "storage": float(parts[4]),
                }
            )

        elif section == "MESHES":
            meshes.append(line)

        elif section == "INITIAL_CONDITION":
            parts = re.split(r"[\s,\t]+", line)
            if len(parts) == 2:
                initial_condition["vtu_file"] = parts[0].strip()
                initial_condition["field_name"] = parts[1].strip()

        elif section == "DIRICHLET_CONDITION":
            parts = re.split(r"[\s,\t]+", line)
            dirichlet_condition.append({"vtu_file": parts[0].strip(), "field_name": parts[1].strip()})

        elif section == "PROCESS":
            key, value = line.split("=", 1)
            process_settings[key.strip()] = value.strip()

        elif section == "OUTPUT":
            if "=" in line:
                key, value = line.split("=", 1)
                output_settings[key.strip()] = value.strip()
            else:
                output_meshes.append(line)

        elif section == "SOLVERS":
            key, value = line.split("=", 1)
            solver_settings[key.strip()] = value.strip()

    return (
        general_settings,
        wells,
        time_settings,
        media_list,
        meshes,
        initial_condition,
        dirichlet_condition,
        process_settings,
        output_settings,
        output_meshes,
        solver_settings,
    )


# -----------------------------
# Mesh staging and rewriting
# -----------------------------
def stage_meshes_to_dir(mesh_files: List[str], dirichlet_condition: List[Dict[str, str]], mesh_dir: str, method: str, nproc: int | None = None) -> None:
    """
    Create mesh_dir and place meshes there.
    method
      link  create symlinks if possible, fallback to copy
      copy  always copy
      none  do nothing
    """
    method = str(method).strip().lower()
    if method == "none":
        return
    if method not in ["link", "copy"]:
        method = "link"

    _safe_mkdir(mesh_dir)

    def stage_one(src: str) -> None:
        if not src:
            return
        if not os.path.exists(src):
            return
        dst = os.path.join(mesh_dir, os.path.basename(src))

        if os.path.lexists(dst):
            try:
                if os.path.isdir(dst) and not os.path.islink(dst):
                    shutil.rmtree(dst)
                else:
                    os.unlink(dst)
            except Exception:
                pass

        if method == "copy":
            shutil.copy2(src, dst)
            return

        try:
            os.symlink(os.path.abspath(src), dst)
        except Exception:
            shutil.copy2(src, dst)


    nproc_eff: Optional[int] = None
    try:
        if nproc is not None:
            nproc_eff = int(nproc)
    except Exception:
        nproc_eff = None

    def stage_partitioned_bins_for_vtu(vtu_path: str) -> None:
        # For MPI runs (nproc >= 2), OGS expects additional partitioned binary files
        # alongside the referenced VTU. Stage them into mesh_dir as well.
        if nproc_eff is None or nproc_eff < 2:
            return
        if not vtu_path:
            return
        stem = os.path.splitext(os.path.basename(vtu_path))[0]
        src_dir = os.path.dirname(vtu_path) or "."
        pattern = os.path.join(src_dir, f"{stem}_partitioned_*{nproc_eff}.bin")
        for src in glob.glob(pattern):
            stage_one(src)

    for p in mesh_files:
        stage_one(p)
        stage_partitioned_bins_for_vtu(p)
    for dc in dirichlet_condition or []:
        vp = dc.get("vtu_file", "")
        stage_one(vp)
        stage_partitioned_bins_for_vtu(vp)


def rewrite_mesh_paths_to_dir(mesh_files: List[str], dirichlet_condition: List[Dict[str, str]], mesh_dir: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Rewrite mesh file references to mesh_dir/<basename>.
    This is useful when you run from a run directory and want all runtime meshes under run/mesh.
    """
    if (mesh_dir is None) or str(mesh_dir).strip() == "":
        return mesh_files, dirichlet_condition

    new_meshes = []
    for p in mesh_files:
        if not p:
            continue
        new_meshes.append(_norm_slash(os.path.join(mesh_dir, os.path.basename(p))))

    new_dc = []
    for dc in dirichlet_condition or []:
        p = dc.get("vtu_file", "")
        new_dc.append({"vtu_file": _norm_slash(os.path.join(mesh_dir, os.path.basename(p))), "field_name": dc.get("field_name", "")})

    return new_meshes, new_dc


# -----------------------------
# Aux (curve binary) staging and rewriting
# -----------------------------
def stage_aux_bins_to_dir(wells: List[Dict[str, str]], aux_bin_dir: str, method: str) -> None:
    """
    Stage pre-generated curve binaries into aux_bin_dir.
    method
      link  create symlinks if possible, fallback to copy
      copy  always copy
      none  do nothing
    """
    method = str(method).strip().lower()
    if method == "none":
        return
    if method not in ["link", "copy"]:
        method = "link"

    _safe_mkdir(aux_bin_dir)

    def stage_one(src: str) -> None:
        if not src:
            return
        if not os.path.exists(src):
            return
        dst = os.path.join(aux_bin_dir, os.path.basename(src))

        if os.path.lexists(dst):
            try:
                if os.path.isdir(dst) and not os.path.islink(dst):
                    shutil.rmtree(dst)
                else:
                    os.unlink(dst)
            except Exception:
                pass

        if method == "copy":
            shutil.copy2(src, dst)
            return

        try:
            os.symlink(os.path.abspath(src), dst)
        except Exception:
            shutil.copy2(src, dst)

    for w in wells or []:
        stage_one(str(w.get("coords_bin", "")).strip())
        stage_one(str(w.get("values_bin", "")).strip())


def rewrite_aux_bin_paths_to_dir(wells: List[Dict[str, str]], aux_bin_dir: str) -> List[Dict[str, str]]:
    """
    Rewrite wells coords_bin and values_bin to aux_bin_dir/<basename>.
    This is used so that the prj references are stable relative paths under run/.
    """
    if (aux_bin_dir is None) or str(aux_bin_dir).strip() == "":
        return wells

    new_wells: List[Dict[str, str]] = []
    for w in wells or []:
        w2 = dict(w)
        cb = str(w2.get("coords_bin", "")).strip()
        vb = str(w2.get("values_bin", "")).strip()
        if cb != "":
            w2["coords_bin"] = _norm_slash(os.path.join(aux_bin_dir, os.path.basename(cb)))
        if vb != "":
            w2["values_bin"] = _norm_slash(os.path.join(aux_bin_dir, os.path.basename(vb)))
        new_wells.append(w2)
    return new_wells


# -----------------------------
# Curve generation
# -----------------------------
def write_curve_bins(flow_file: str, aux_bin_dir: str, well_id: str) -> Tuple[str, str]:
    """
    Convert a two-column text file into two float64 little-endian binary files.
    Returns relative references (slash-normalized) for OGS.
    """
    df_curve = pd.read_csv(
        flow_file,
        sep=r"\s+|,|\t",
        engine="python",
        comment="#",
        header=None,
        names=["t", "value"],
    )
    times = df_curve["t"].values.astype(np.float64)
    values = df_curve["value"].values.astype(np.float64)

    coords_fname = f"{well_id}_coords.bin"
    values_fname = f"{well_id}_values.bin"
    coords_bin = os.path.join(aux_bin_dir, coords_fname)
    values_bin = os.path.join(aux_bin_dir, values_fname)

    _safe_mkdir(aux_bin_dir)
    times.astype("<f8").tofile(coords_bin)
    values.astype("<f8").tofile(values_bin)

    coords_ref = _norm_slash(os.path.join(aux_bin_dir, coords_fname))
    values_ref = _norm_slash(os.path.join(aux_bin_dir, values_fname))
    return coords_ref, values_ref


# -----------------------------
# Partition helper
# -----------------------------
def collect_partition_targets(mesh_files, global_mesh_file):
    targets = []
    seen = set()
    for p in mesh_files:
        if not p:
            continue
        if os.path.normpath(p) == os.path.normpath(global_mesh_file):
            continue
        if p not in seen:
            seen.add(p)
            targets.append(p)
    return targets


def build_partition_commands(global_mesh_file, mesh_files, nproc, mesh_dir, partmesh_bin, mpmetis_bin):
    mesh_dir = mesh_dir or "mesh"
    nproc = int(nproc)

    global_mesh_file = str(global_mesh_file)
    mesh_name = os.path.splitext(os.path.basename(global_mesh_file))[0]

    targets = collect_partition_targets(mesh_files, global_mesh_file)
    target_basenames = [os.path.basename(p) for p in targets]

    cmds = []
    cmds.append(f"mkdir -p {_shell_quote(mesh_dir)}")
    cmds.append(f"{_shell_quote(partmesh_bin)} -i {_shell_quote(global_mesh_file)} -n {nproc} -s -o {_shell_quote(mesh_dir)}")
    cmds.append(f"{_shell_quote(mpmetis_bin)} -gtype=nodal {_shell_quote(os.path.join(mesh_dir, mesh_name + '.mesh'))} {nproc}")

    if target_basenames:
        lines = []
        lines.append(f"cd {_shell_quote(mesh_dir)}")
        lines.append(f"{_shell_quote(partmesh_bin)} -i {_shell_quote(mesh_name + '.vtu')} -n {nproc} -x {_shell_quote(mesh_name)} -- \\")
        for k, bn in enumerate(target_basenames):
            tail = " \\" if k < len(target_basenames) - 1 else ""
            lines.append(f"  {_shell_quote(bn)}{tail}")
        lines.append("cd ..")
        cmds.append("\n".join(lines))

    return cmds


def build_mpirun_command(ogs_bin, nproc, prj_file, output_dir: Optional[str] = None):
    nproc = int(nproc)
    if output_dir and str(output_dir).strip() != "":
        return f"mpirun -np {nproc} {_shell_quote(ogs_bin)} {_shell_quote(prj_file)} -o {_shell_quote(output_dir)}"
    return f"mpirun -np {nproc} {_shell_quote(ogs_bin)} {_shell_quote(prj_file)}"


# -----------------------------
# Main project generation
# -----------------------------
def generate_ogs_prj(
    output_xml: str,
    global_mesh_file: str,
    mesh_files: List[str],
    wells: List[Dict[str, str]],
    dirichlet_condition: List[Dict[str, str]],
    time_settings: Dict[str, object],
    media_list: List[Dict[str, object]],
    process_settings: Dict[str, str],
    output_settings: Dict[str, str],
    output_meshes: List[str],
    solver_settings: Dict[str, str],
    aux_bin_dir: str,
    generate_injection_bins: bool,
) -> None:
    root = ET.Element("OpenGeoSysProject")

    # processes
    root.append(ET.Comment(" Processes "))
    processes_tag = ET.SubElement(root, "processes")
    process_tag = ET.SubElement(processes_tag, "process")
    ET.SubElement(process_tag, "name").text = process_settings.get("name", "LiquidFlow")
    ET.SubElement(process_tag, "type").text = process_settings.get("type", "LIQUID_FLOW")
    ET.SubElement(process_tag, "integration_order").text = process_settings.get("integration_order", "2")
    proc_var_name = process_settings.get("process_variable", "hydraulic_head")
    proc_vars = ET.SubElement(process_tag, "process_variables")
    ET.SubElement(proc_vars, "process_variable").text = proc_var_name
    secondary_vars = ET.SubElement(process_tag, "secondary_variables")
    ET.SubElement(secondary_vars, "secondary_variable", internal_name="darcy_velocity", output_name="darcy_velocity")
    ET.SubElement(process_tag, "specific_body_force").text = process_settings.get("specific_body_force", "0 0 -9.81")

    # meshes
    root.append(ET.Comment(" Mesh files "))
    meshes_tag = ET.SubElement(root, "meshes")
    ordered = []
    seen = set()

    def add(path):
        if path and path not in seen:
            seen.add(path)
            ordered.append(path)

    for mesh_file in mesh_files:
        add(mesh_file)
    if dirichlet_condition:
        for dc in dirichlet_condition:
            add(dc.get("vtu_file"))

    for path in ordered:
        ET.SubElement(meshes_tag, "mesh").text = _norm_slash(path)

    # time loop and output
    root.append(ET.Comment(" Time loop "))
    time_loop = ET.SubElement(root, "time_loop")
    processes = ET.SubElement(time_loop, "processes")
    tag = process_settings.get("name", "LiquidFlow")
    process = ET.SubElement(processes, "process", attrib={"ref": tag})

    if str(time_settings.get("solver")) == "nonlinear":
        ET.SubElement(process, "nonlinear_solver").text = solver_settings.get("nonlinear_solver_name", "basic_picard")
    else:
        ET.SubElement(process, "linear_solver").text = solver_settings.get("linear_solver_name", "general_linear_solver")

    convergence_criterion = ET.SubElement(process, "convergence_criterion")
    ET.SubElement(convergence_criterion, "type").text = "DeltaX"
    ET.SubElement(convergence_criterion, "norm_type").text = "NORM2"
    ET.SubElement(convergence_criterion, "abstol").text = "1.e-10"

    time_discretization = ET.SubElement(process, "time_discretization")
    ET.SubElement(time_discretization, "type").text = "BackwardEuler"

    time_stepping = ET.SubElement(process, "time_stepping")
    ET.SubElement(time_stepping, "type").text = str(time_settings.get("type", "SingleStep"))
    ET.SubElement(time_stepping, "t_initial").text = str(time_settings.get("t_initial", 0.0))
    ET.SubElement(time_stepping, "t_end").text = str(time_settings.get("t_end", 1.0))
    timesteps = ET.SubElement(time_stepping, "timesteps")
    pair = ET.SubElement(timesteps, "pair")
    ET.SubElement(pair, "repeat").text = str(time_settings.get("timesteps_repeat", 1))
    ET.SubElement(pair, "delta_t").text = str(time_settings.get("timesteps_delta_t", 1.0))

    time_loop.append(ET.Comment(" Output "))
    output = ET.SubElement(time_loop, "output")
    ET.SubElement(output, "type").text = output_settings.get("type", "VTK")

    output_prefix = output_settings.get("prefix", "result")

    # If prefix contains a directory, OGS may write a .pvd that references that directory
    # even when .pvd itself is placed elsewhere in some MPI setups.
    # To avoid manual editing of .pvd, prefer using ogs -o <output_dir> and keep prefix base-only.
    if os.path.dirname(output_prefix) != "":
        output_prefix = os.path.basename(output_prefix)

    selected_mesh_names = []
    if output_meshes:
        defined = _collect_defined_mesh_names(mesh_files, global_mesh_file, dirichlet_condition)
        for tok in output_meshes:
            name = _mesh_name_from_token(tok)
            if name == "":
                continue
            if name not in defined:
                raise ValueError(
                    f"[OUTPUT] mesh '{tok}' does not match any mesh in [MESHES] or DIRICHLET_CONDITION"
                )
            selected_mesh_names.append(name)

    if selected_mesh_names:
        output_prefix = _ensure_meshname_in_prefix(output_prefix)

    ET.SubElement(output, "prefix").text = output_prefix

    if selected_mesh_names:
        meshes_tag2 = ET.SubElement(output, "meshes")
        for mn in selected_mesh_names:
            ET.SubElement(meshes_tag2, "mesh").text = mn

    timesteps2 = ET.SubElement(output, "timesteps")
    pair2 = ET.SubElement(timesteps2, "pair")
    ET.SubElement(pair2, "repeat").text = str(output_settings.get("repeat", 1))
    ET.SubElement(pair2, "each_steps").text = str(output_settings.get("each_steps", 1))
    variables = ET.SubElement(output, "variables")
    ET.SubElement(variables, "variable").text = process_settings.get("process_variable", "hydraulic_head")
    ET.SubElement(output, "suffix").text = "_ts_{:timestep}"

    # media
    root.append(ET.Comment(" Domain property "))
    media_tag = ET.SubElement(root, "media")
    for m in media_list:
        media_tag.append(ET.Comment(str(m["name"])))
        medium = ET.SubElement(media_tag, "medium", id=str(m["id"]))
        phase = ET.SubElement(ET.SubElement(medium, "phases"), "phase")
        ET.SubElement(phase, "type").text = "AqueousLiquid"
        props = ET.SubElement(phase, "properties")
        for name, val in [("viscosity", 1), ("density", 1)]:
            p = ET.SubElement(props, "property")
            ET.SubElement(p, "name").text = name
            ET.SubElement(p, "type").text = "Constant"
            ET.SubElement(p, "value").text = str(val)
        mprops = ET.SubElement(medium, "properties")
        for name, val in [
            ("permeability", m["perm"]),
            ("porosity", m["poro"]),
            ("reference_temperature", 293.15),
            ("storage", m["storage"]),
        ]:
            p = ET.SubElement(mprops, "property")
            ET.SubElement(p, "name").text = name
            ET.SubElement(p, "type").text = "Constant"
            ET.SubElement(p, "value").text = str(val)

    # curves and parameters for injection
    written_curves = set()
    if wells:
        root.append(ET.Comment(" Curves "))
        curves_tag = ET.SubElement(root, "curves")

        for well in wells:
            well_mesh_id = os.path.splitext(os.path.basename(well["vtu_file"]))[0]

            coords_ref = str(well.get("coords_bin", "")).strip()
            values_ref = str(well.get("values_bin", "")).strip()

            if coords_ref != "" or values_ref != "":
                if coords_ref == "" or values_ref == "":
                    raise ValueError(
                        f"Well '{well_mesh_id}' must provide both coords_bin and values_bin (or neither)."
                    )
                if not os.path.exists(coords_ref):
                    raise ValueError(f"Missing coords_bin for well '{well_mesh_id}': {coords_ref}")
                if not os.path.exists(values_ref):
                    raise ValueError(f"Missing values_bin for well '{well_mesh_id}': {values_ref}")

            else:
                if not generate_injection_bins:
                    raise ValueError(
                        f"Injection bins are required but not provided for well '{well_mesh_id}'. "
                        "Provide coords_bin and values_bin in JSON or enable generation from flow_file."
                    )
                flow_file = str(well.get("flow_file", "")).strip()
                if flow_file == "" or (not os.path.exists(flow_file)):
                    raise ValueError(
                        f"Missing flow_file for well '{well_mesh_id}' needed to generate injection bins: {flow_file}"
                    )
                coords_ref, values_ref = write_curve_bins(flow_file, aux_bin_dir, well_mesh_id)

            if well_mesh_id not in written_curves:
                curve = ET.SubElement(curves_tag, "curve")
                ET.SubElement(curve, "name").text = f"curve_{well_mesh_id}"
                ET.SubElement(curve, "read_from_file").text = "true"
                ET.SubElement(curve, "coords").text = coords_ref
                ET.SubElement(curve, "values").text = values_ref
                written_curves.add(well_mesh_id)

    root.append(ET.Comment(" Parameters "))
    parameters = ET.SubElement(root, "parameters")

    if wells:
        for well in wells:
            well_mesh_id = os.path.splitext(os.path.basename(well["vtu_file"]))[0]
            parameter = ET.SubElement(parameters, "parameter")
            ET.SubElement(parameter, "name").text = f"injection_schedule_{well_mesh_id}"
            ET.SubElement(parameter, "type").text = "Function"
            ET.SubElement(parameter, "expression").text = f"curve_{well_mesh_id}(t)"

    ic_param = ET.SubElement(parameters, "parameter")
    ET.SubElement(ic_param, "name").text = "ic_hydraulic_head"
    ET.SubElement(ic_param, "type").text = "MeshNode"
    ET.SubElement(ic_param, "mesh").text = os.path.splitext(os.path.basename(global_mesh_file))[0]
    ET.SubElement(ic_param, "field_name").text = "hydraulic_head"

    if dirichlet_condition:
        i = 0
        for dc in dirichlet_condition:
            i += 1
            parameter = ET.SubElement(parameters, "parameter")
            ET.SubElement(parameter, "name").text = f"dirichlet_condition_{i}"
            ET.SubElement(parameter, "type").text = "MeshNode"
            ET.SubElement(parameter, "mesh").text = os.path.splitext(os.path.basename(dc["vtu_file"]))[0]
            ET.SubElement(parameter, "field_name").text = dc["field_name"]

    # process variables
    root.append(ET.Comment(" Process variables "))
    proc_var_tag = ET.SubElement(ET.SubElement(root, "process_variables"), "process_variable")
    ET.SubElement(proc_var_tag, "name").text = "hydraulic_head"
    ET.SubElement(proc_var_tag, "components").text = "1"
    ET.SubElement(proc_var_tag, "order").text = "1"
    ET.SubElement(proc_var_tag, "initial_condition").text = "ic_hydraulic_head"

    bcs = ET.SubElement(proc_var_tag, "boundary_conditions")
    if dirichlet_condition:
        i = 0
        for dc in dirichlet_condition:
            i += 1
            bc = ET.SubElement(bcs, "boundary_condition")
            ET.SubElement(bc, "mesh").text = os.path.splitext(os.path.basename(dc["vtu_file"]))[0]
            ET.SubElement(bc, "type").text = "Dirichlet"
            ET.SubElement(bc, "parameter").text = f"dirichlet_condition_{i}"

    if wells:
        source_terms = ET.SubElement(proc_var_tag, "source_terms")
        for well in wells:
            well_mesh_id = os.path.splitext(os.path.basename(well["vtu_file"]))[0]
            well_tag = ET.SubElement(source_terms, "source_term")
            ET.SubElement(well_tag, "type").text = "Nodal"
            ET.SubElement(well_tag, "mesh").text = well_mesh_id
            ET.SubElement(well_tag, "parameter").text = f"injection_schedule_{well_mesh_id}"

    # solvers
    root.append(ET.Comment(" Solver "))
    nl = ET.SubElement(root, "nonlinear_solvers")
    nls = ET.SubElement(nl, "nonlinear_solver")
    ET.SubElement(nls, "name").text = solver_settings.get("nonlinear_solver_name", "basic_picard")
    ET.SubElement(nls, "type").text = solver_settings.get("nonlinear_solver_type", "Picard")
    ET.SubElement(nls, "max_iter").text = solver_settings.get("nonlinear_solver_max_iter", "10")
    ET.SubElement(nls, "linear_solver").text = solver_settings.get("nonlinear_solver_linear", "general_linear_solver")

    backend = str(solver_settings.get("linear_solver_backend", "petsc")).strip().lower()
    ls = ET.SubElement(root, "linear_solvers")
    lsolver = ET.SubElement(ls, "linear_solver")
    ET.SubElement(lsolver, "name").text = solver_settings.get("linear_solver_name", "general_linear_solver")

    if backend == "petsc":
        petsc = ET.SubElement(lsolver, "petsc")
        ET.SubElement(petsc, "prefix").text = solver_settings.get("linear_solver_petsc_prefix", "lf")
        ET.SubElement(petsc, "parameters").text = solver_settings.get(
            "linear_solver_petsc_parameters",
            "-lf_ksp_type cg -lf_pc_type bjacobi -lf_ksp_rtol 1e-16 -lf_ksp_max_it 10000",
        )
    elif backend == "lis":
        ET.SubElement(lsolver, "lis").text = solver_settings.get(
            "linear_solver_lis",
            "-i cg -p jacobi -tol 1e-20 -maxiter 10000",
        )
    elif backend == "eigen":
        eigen = ET.SubElement(lsolver, "eigen")
        ET.SubElement(eigen, "solver_type").text = solver_settings.get("linear_solver_eigen_solver_type", "CG")
        ET.SubElement(eigen, "precon_type").text = solver_settings.get("linear_solver_eigen_precon_type", "DIAGONAL")
        ET.SubElement(eigen, "max_iteration_step").text = solver_settings.get("linear_solver_eigen_max_iteration_step", "10000")
        ET.SubElement(eigen, "error_tolerance").text = solver_settings.get("linear_solver_eigen_error_tolerance", "1e-20")
    else:
        raise ValueError("Unknown linear_solver_backend. Use petsc, lis, or eigen.")

    root.append(ET.Comment(""))
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    print(f"[INFO] Project file written: {output_xml}")


# -----------------------------
# CLI overrides from GENERAL
# -----------------------------
def apply_general_overrides_to_args(args, general_settings, argv=None):
    """
    If user explicitly passed an option, keep it.
    Otherwise if [GENERAL] provides it, use it.
    """
    if argv is None:
        argv = sys.argv[1:]

    def cli_has(flag):
        return flag in argv

    def set_if_missing(attr, key, cast=None):
        if getattr(args, attr, None) is None and key in general_settings:
            val = general_settings.get(key, None)
            if val is None:
                return
            val = str(val).strip()
            if val == "":
                return
            setattr(args, attr, cast(val) if cast is not None else val)

    set_if_missing("nproc", "nproc", int)
    set_if_missing("omp_num_threads", "omp_num_threads", int)
    set_if_missing("mesh_dir", "mesh_dir", str)
    set_if_missing("aux_bin_dir", "aux_bin_dir", str)
    set_if_missing("output_dir", "output_dir", str)
    # ogs_bin is kept for version checks only. Actual execution should be done by sbatch with srun.
    set_if_missing("ogs_bin", "ogs_bin", str)
    set_if_missing("partmesh_bin", "partmesh_bin", str)
    set_if_missing("mpmetis_bin", "mpmetis_bin", str)
    set_if_missing("stage_meshes", "stage_meshes", str)

    if (not cli_has("--do_partition")) and ("do_partition" in general_settings):
        vb = _parse_bool(general_settings.get("do_partition"))
        if vb is not None:
            args.do_partition = vb

    if (not cli_has("--print_workflow")) and ("print_workflow" in general_settings):
        vb = _parse_bool(general_settings.get("print_workflow"))
        if vb is not None:
            args.print_workflow = vb

    if (not cli_has("--skip_ogs_precheck")) and ("skip_ogs_precheck" in general_settings):
        vb = _parse_bool(general_settings.get("skip_ogs_precheck"))
        if vb is not None:
            args.skip_ogs_precheck = vb

    return args



def main():
    parser = argparse.ArgumentParser(
        description="Prepare OGS run directory. Generate prj, stage meshes, generate aux binaries, and optionally partition meshes."
    )
    parser.add_argument("--config", default=None, help="Optional local YAML config file.")
    parser.add_argument("--base_json", default="case_base.json", help="Path to base case JSON.")
    parser.add_argument("--params_json", default="", help="Path to params JSON. Optional.")
    parser.add_argument("--output_xml", default=None, help="Output OGS project file path.")
    parser.add_argument("--do_partition", action="store_true", help="Run partition commands. If not set, only staging is done.")
    parser.add_argument("--print_workflow", action="store_true", help="Print partition commands.")
    parser.add_argument("--nproc", type=int, default=None, help="Number of MPI ranks used for partitioning.")
    parser.add_argument("--mesh_src_dir", default=None, help="Directory containing all mesh VTU files referenced by base_json. Overrides mesh file paths to <mesh_src_dir>/<basename>.")
    parser.add_argument("--injection_src_dir", default=None, help="Directory containing all injection schedule text files referenced by base_json wells. Overrides flow_file paths to <injection_src_dir>/<basename>.")
    parser.add_argument("--injection_bin_src_dir", default=None, help="Directory containing all pre-generated injection curve binary files referenced by base_json wells (coords_bin/values_bin). Overrides bin paths to <injection_bin_src_dir>/<basename>.")
    parser.add_argument("--no_generate_injection_bins", action="store_true", help="Do not generate injection curve binaries from flow_file. Require coords_bin and values_bin to be provided (and staged) for every well.")
    parser.add_argument("--mesh_dir", default=None, help="Runtime mesh directory. Required.")
    parser.add_argument("--aux_bin_dir", default=None, help="Directory to write curve binary files. Required.")
    parser.add_argument("--stage_meshes", default=None, help="Stage meshes into mesh_dir. Values: link, copy.")
    parser.add_argument("--partmesh_bin", default=None, help="Path to partmesh executable.")
    parser.add_argument("--mpmetis_bin", default=None, help="Path to mpmetis executable.")

    args = parser.parse_args()
    config = load_config(args.config)

    (
        general_settings,
        wells,
        time_settings,
        media_list,
        mesh_files,
        initial_condition,
        dirichlet_condition,
        process_settings,
        output_settings,
        output_meshes,
        solver_settings,
    ) = read_input_json(args.base_json, (args.params_json.strip() or None))

    if args.output_xml:
        general_settings["output_xml"] = args.output_xml
    elif "output_xml" not in general_settings and "output_project_file" not in general_settings:
        general_settings["output_xml"] = str(local_output_path(config, "ogs_input", "ogs_project.prj"))

    if args.mesh_src_dir is None:
        mesh_src_candidate = old_workspace_path(config, "ogs", "input", "mesh", "vtu")
        if mesh_src_candidate is not None and mesh_src_candidate.exists():
            args.mesh_src_dir = str(mesh_src_candidate)

    if args.mesh_dir is None and str(general_settings.get("mesh_dir", "")).strip() == "":
        args.mesh_dir = str(local_output_path(config, "ogs_input", "mesh"))

    if args.aux_bin_dir is None and str(general_settings.get("aux_bin_dir", "")).strip() == "":
        args.aux_bin_dir = str(local_output_path(config, "ogs_input", "aux"))

    # Optional path overrides for shared assets.
    # mesh_src_dir should contain ALL mesh VTU files (domain + boundary + well meshes) in one directory.
    # injection_src_dir should contain ALL well schedule text files.
    mesh_files, dirichlet_condition, wells = apply_source_dir_overrides(
        mesh_src_dir=args.mesh_src_dir,
        injection_src_dir=args.injection_src_dir,
        injection_bin_src_dir=args.injection_bin_src_dir,
        general_settings=general_settings,
        mesh_files=mesh_files,
        dirichlet_condition=dirichlet_condition,
        wells=wells,
        generate_injection_bins=(not args.no_generate_injection_bins),
    )

    aux_bin_dir = (args.aux_bin_dir or str(general_settings.get("aux_bin_dir", "")).strip()).strip()
    mesh_dir = (args.mesh_dir or str(general_settings.get("mesh_dir", "")).strip()).strip()
    stage_meshes_mode = (args.stage_meshes or str(general_settings.get("stage_meshes", "link")).strip()).strip().lower()

    if aux_bin_dir == "":
        raise ValueError("aux_bin_dir is required. Set it in GENERAL or pass --aux_bin_dir.")
    if mesh_dir == "":
        raise ValueError("mesh_dir is required. Set it in GENERAL or pass --mesh_dir.")

    if stage_meshes_mode not in ["link", "copy"]:
        stage_meshes_mode = "link"

    nproc = _resolve_int_setting(args.nproc, general_settings, "nproc", 2)

    partmesh_bin = args.partmesh_bin or _pick_setting(general_settings, "partmesh_bin", "partmesh")
    mpmetis_bin = args.mpmetis_bin or _pick_setting(general_settings, "mpmetis_bin", "mpmetis")

    # Enforce safer output prefix so that pvd references are local to output_dir
    prefix_raw = str(output_settings.get("prefix", "")).strip()
    if prefix_raw == "":
        output_settings["prefix"] = "result"
    else:
        output_settings["prefix"] = os.path.basename(prefix_raw)

    # Always stage meshes into mesh_dir, then rewrite prj references to mesh_dir/<basename>
    stage_meshes_to_dir(mesh_files, dirichlet_condition, mesh_dir, stage_meshes_mode, nproc=nproc)
    mesh_files_eff, dirichlet_condition_eff = rewrite_mesh_paths_to_dir(mesh_files, dirichlet_condition, mesh_dir)

    # Injection curve binaries:
    # If coords_bin/values_bin are provided (pre-generated), stage them into aux_bin_dir and rewrite references to aux/<basename>.
    # Otherwise, generate them from flow_file into aux_bin_dir if generation is enabled.
    generate_injection_bins = (not args.no_generate_injection_bins)

    any_bins_specified = False
    for w in wells or []:
        if str(w.get("coords_bin", "")).strip() != "" or str(w.get("values_bin", "")).strip() != "":
            any_bins_specified = True
            break

    if any_bins_specified:
        stage_aux_bins_to_dir(wells, aux_bin_dir, stage_meshes_mode)
        wells_eff = rewrite_aux_bin_paths_to_dir(wells, aux_bin_dir)
    else:
        wells_eff = wells

    # Generate project file
    output_xml = str(general_settings.get("output_xml", "ogs_project.prj")).strip() or "ogs_project.prj"
    global_mesh_file = str(general_settings.get("global_mesh_file", "")).strip() or mesh_files_eff[0]

    generate_ogs_prj(
        output_xml=output_xml,
        global_mesh_file=global_mesh_file,
        mesh_files=mesh_files_eff,
        wells=wells_eff,
        dirichlet_condition=dirichlet_condition_eff,
        time_settings=time_settings,
        media_list=media_list,
        process_settings=process_settings,
        output_settings=output_settings,
        output_meshes=output_meshes,
        solver_settings=solver_settings,
        aux_bin_dir=aux_bin_dir,
        generate_injection_bins=generate_injection_bins,
    )

    # Partition commands
    cmds = build_partition_commands(
        global_mesh_file=global_mesh_file,
        mesh_files=mesh_files_eff,
        nproc=nproc,
        mesh_dir=mesh_dir,
        partmesh_bin=partmesh_bin,
        mpmetis_bin=mpmetis_bin,
    )

    if args.print_workflow:
        print()
        print("Partition commands")
        for c in cmds:
            print(c)
        print()

    if args.do_partition:
        for c in cmds:
            subprocess.run(c, shell=True, check=True, executable="/bin/bash")

    print("Prepared run directory")
    print(f"  prj: {output_xml}")
    print(f"  mesh_dir: {mesh_dir}")
    print(f"  aux_bin_dir: {aux_bin_dir}")


if __name__ == "__main__":
    main()
