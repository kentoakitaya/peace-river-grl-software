import argparse
from pathlib import Path

import gmsh
import numpy as np
import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_config(config_path: str | None) -> dict[str, object]:
    if not config_path:
        return {}
    with open(Path(config_path).expanduser().resolve(), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise RuntimeError("--config must contain a YAML mapping.")
    return config


def config_path(config: dict[str, object], key: str) -> Path | None:
    value = config.get(key)
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value)).expanduser().resolve()


def resolve_input_msh(input_msh_arg: str | None, config: dict[str, object]) -> Path:
    if input_msh_arg:
        return Path(input_msh_arg).expanduser().resolve()

    old_workspace_root = config_path(config, "old_workspace_root")
    if old_workspace_root is not None:
        candidate = old_workspace_root / "ogs" / "input" / "mesh" / "msh" / "mesh.msh"
        if candidate.exists():
            return candidate

    raise RuntimeError(
        "--input_msh is required, or pass --config with old_workspace_root for migration-helper lookup."
    )


def resolve_output_msh(output_msh_arg: str | None, config: dict[str, object]) -> Path:
    if output_msh_arg:
        return Path(output_msh_arg).expanduser().resolve()

    local_output_dir = config_path(config, "local_output_dir")
    if local_output_dir is not None:
        return local_output_dir / "mesh" / "mesh_rescaled.msh"

    return repo_root() / "outputs" / "mesh" / "mesh_rescaled.msh"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rescale a generated Peace River mesh for the OGS workflow."
    )
    parser.add_argument("--config", default=None, help="Optional local YAML config file.")
    parser.add_argument("--input_msh", default=None, help="Input generated mesh file.")
    parser.add_argument("--output_msh", default=None, help="Output rescaled mesh file.")
    parser.add_argument("--magnification", type=float, default=5.0, help="Vertical rescaling factor.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)

    input_file = resolve_input_msh(args.input_msh, config)
    output_file = resolve_output_msh(args.output_msh, config)

    if not input_file.exists():
        raise FileNotFoundError(f"Input mesh file not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.open(str(input_file))

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords).reshape(-1, 3)
        coords[:, 2] /= args.magnification

        for tag, coord in zip(node_tags, coords):
            gmsh.model.mesh.setNode(tag, coord.tolist(), [])

        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(output_file))
    finally:
        gmsh.finalize()

    print()
    print()
    print(f"*** Saved the rescaled mesh file -> '{output_file}' ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
