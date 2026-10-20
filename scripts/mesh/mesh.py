import argparse

import gmsh
import pandas as pd
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter
from collections import defaultdict
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

Magnification = 5

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
MESH_OUTPUT_DIR = REPO_ROOT / "outputs" / "mesh"
MESH_FILE_PATH = MESH_OUTPUT_DIR / "mesh.msh"
VOLUME_TAGS_PATH = MESH_OUTPUT_DIR / "volume_tags.txt"
CONFIG_PATH = SCRIPT_DIR / "config.txt"
PHYSICAL_GROUP_PATH = SCRIPT_DIR / "physical_group.txt"

temp_box_top_z = 1000
temp_box_top_z = temp_box_top_z * Magnification
bottom_z = -10000
bottom_z = bottom_z * Magnification

smoothing_intensity=3

def read_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_local_config(config_path):
    if not config_path:
        return {}
    with open(Path(config_path).expanduser().resolve(), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError("--config must contain a YAML mapping.")
    return data


def config_path(config, key):
    value = config.get(key)
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value)).expanduser().resolve()


def old_workspace_path(config, *parts):
    old_workspace_root = config_path(config, "old_workspace_root")
    if old_workspace_root is None:
        return None
    return old_workspace_root.joinpath(*parts)


def local_output_path(config, *parts):
    local_output_dir = config_path(config, "local_output_dir")
    root = local_output_dir if local_output_dir is not None else REPO_ROOT / "outputs"
    return root.joinpath(*parts)


def resolve_existing_path(path_value):
    raw_path = Path(path_value)
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.extend([Path.cwd() / raw_path, SCRIPT_DIR / raw_path, REPO_ROOT / raw_path])

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def resolve_input_dir(mesh_config, input_dir_arg, local_config):
    if input_dir_arg:
        resolved = Path(input_dir_arg).expanduser().resolve()
        if resolved.is_dir():
            return resolved
        raise FileNotFoundError(f"Input directory not found: {resolved}")

    configured_dir = mesh_config.get("output_dir")
    if configured_dir:
        resolved = resolve_existing_path(configured_dir)
        if resolved is not None and resolved.is_dir():
            return resolved

    for candidate in [
        old_workspace_path(local_config, "ogs", "input", "mesh", "mesh_inputs"),
        old_workspace_path(local_config, "ogs", "input", "mesh", "output"),
    ]:
        if candidate is not None and candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Input directory not found. Pass --input_dir, set output_dir in --mesh_config, "
        "or pass --config with old_workspace_root for migration-helper lookup."
    )

def create_grid(config):
    Lx, Ly = config['Lx'], config['Ly']
    dx, dy = config['dx'], config['dy']
    nx, ny = int(Lx // dx), int(Ly // dy)
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    return np.meshgrid(x, y), nx, ny

def apply_thickness(df, array, magnification, min_thickness):
    for _, row in df.iterrows():
        ix, iy = int(row["ix"]), int(row["iy"])
        value = max(row["z"] * magnification, min_thickness * magnification)
        array[iy, ix] = value


def create_bspline_surface(xx, yy, zz, nx, ny):
    points = [[gmsh.model.occ.addPoint(xx[iy, ix], yy[iy, ix], zz[iy, ix])
               for ix in range(nx)] for iy in range(ny)]
    flat = [points[iy][ix] for ix in range(nx) for iy in range(ny)]
    return gmsh.model.occ.addBSplineSurface(flat, numPointsU=ny)


def create_mask_box_from_boundary(boundary_path, dx, dy, x0, y0, zmin):
    with open(boundary_path, "r") as f:
        lines = f.readlines()
    
    all_boundary_lists = []
    current_loop = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if current_loop:
                all_boundary_lists.append(current_loop)
                current_loop = []
        else:
            current_loop.append(tuple(map(int, stripped.split(","))))
    
    if current_loop:
        all_boundary_lists.append(current_loop)

    if not all_boundary_lists:
        raise ValueError("No valid boundary loops found in the input file.")

    loop_ids = []

    for boundary_list in all_boundary_lists:
        edge_pts = []
        for ix, iy in boundary_list:
            x = x0 + ix * dx
            y = y0 + iy * dy
            edge_pts.append(gmsh.model.occ.addPoint(x, y, temp_box_top_z))

        curves = [gmsh.model.occ.addLine(edge_pts[i], edge_pts[i + 1]) for i in range(len(edge_pts) - 1)]
        curves.append(gmsh.model.occ.addLine(edge_pts[-1], edge_pts[0]))

        loop_id = gmsh.model.occ.addCurveLoop(curves)
        loop_ids.append(loop_id)

    # 最初のループが outer、残りは inner
    outer_loop = loop_ids[0]
    inner_loops = loop_ids[1:]
    surface = gmsh.model.occ.addPlaneSurface([outer_loop] + inner_loops)
    extruded = gmsh.model.occ.extrude([(2, surface)], 0, 0, zmin - 1)

    gmsh.model.occ.synchronize()
    return extruded


def create_3d_fault(config, center_x, center_y, length, thickness, depth_top, depth_bottom, strike_degree, dip_degree):
    strike_rad=np.radians(config['strike_deg'])
    xc = (center_x-config['x_corner_pt'])*np.cos(strike_rad)-(center_y-config['y_corner_pt'])*np.sin(strike_rad)
    yc = (center_x-config['x_corner_pt'])*np.sin(strike_rad)+(center_y-config['y_corner_pt'])*np.cos(strike_rad)
    L = length
    T = thickness
    s = np.radians(strike_degree-config['strike_deg'])
    dip = np.radians(dip_degree)

    p1 = gmsh.model.occ.addPoint(xc - L/2*np.sin(s) - T/2*np.cos(s), yc - L/2*np.cos(s) + T/2*np.sin(s), depth_top*Magnification)
    p2 = gmsh.model.occ.addPoint(xc - L/2*np.sin(s) + T/2*np.cos(s), yc - L/2*np.cos(s) - T/2*np.sin(s), depth_top*Magnification)
    p3 = gmsh.model.occ.addPoint(xc + L/2*np.sin(s) + T/2*np.cos(s), yc + L/2*np.cos(s) - T/2*np.sin(s), depth_top*Magnification)
    p4 = gmsh.model.occ.addPoint(xc + L/2*np.sin(s) - T/2*np.cos(s), yc + L/2*np.cos(s) + T/2*np.sin(s), depth_top*Magnification)

    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p4)
    l4 = gmsh.model.occ.addLine(p4, p1)
    wire = gmsh.model.occ.addWire([l1, l2, l3, l4])
    face = gmsh.model.occ.addPlaneSurface([wire])

    depth = depth_top - depth_bottom
    tan_dip = np.tan(dip)
    offset = depth / tan_dip
    dx = offset * np.cos(s)
    dy = -offset * np.sin(s)

    fault_zone = gmsh.model.occ.extrude([(2, face)], dx, dy, (depth_bottom-depth_top)*Magnification)
    volume = [e for e in fault_zone if e[0] == 3][0]  # dim = 3
    gmsh.model.occ.synchronize()
    return volume


def create_3d_fault_gouge(config, center_x, center_y, length, thickness, depth_top, depth_bottom, strike_degree, dip_degree):
    strike_rad=np.radians(config['strike_deg'])
    xc = (center_x-config['x_corner_pt'])*np.cos(strike_rad)-(center_y-config['y_corner_pt'])*np.sin(strike_rad)
    yc = (center_x-config['x_corner_pt'])*np.sin(strike_rad)+(center_y-config['y_corner_pt'])*np.cos(strike_rad)
    L = length
    T = thickness
    s = np.radians(strike_degree-config['strike_deg'])
    dip = np.radians(dip_degree)

    # Gouge boundary
    p1_x=xc-L/2*np.sin(s)-T/2*np.cos(s); p1_y=yc-L/2*np.cos(s)+T/2*np.sin(s)
    p2_x=xc-L/2*np.sin(s)+T/2*np.cos(s); p2_y=yc-L/2*np.cos(s)-T/2*np.sin(s)
    p3_x=xc+L/2*np.sin(s)+T/2*np.cos(s); p3_y=yc+L/2*np.cos(s)-T/2*np.sin(s)
    p4_x=xc+L/2*np.sin(s)-T/2*np.cos(s); p4_y=yc+L/2*np.cos(s)+T/2*np.sin(s)
    p1 = gmsh.model.occ.addPoint(p1_x, p1_y, depth_top*Magnification)
    p2 = gmsh.model.occ.addPoint(p2_x, p2_y, depth_top*Magnification)
    p3 = gmsh.model.occ.addPoint(p3_x, p3_y, depth_top*Magnification)
    p4 = gmsh.model.occ.addPoint(p4_x, p4_y, depth_top*Magnification)

    # Gouge center
    p12_center_x=(p1_x+p2_x)/2; p12_center_y=(p1_y+p2_y)/2
    p34_center_x=(p3_x+p4_x)/2; p34_center_y=(p3_y+p4_y)/2
    p12_center = gmsh.model.occ.addPoint(p12_center_x, p12_center_y, depth_top*Magnification)
    p34_center = gmsh.model.occ.addPoint(p34_center_x, p34_center_y, depth_top*Magnification)

    l1 = gmsh.model.occ.addLine(p1        , p12_center)
    l2 = gmsh.model.occ.addLine(p12_center, p34_center)
    l3 = gmsh.model.occ.addLine(p34_center, p4        )
    l4 = gmsh.model.occ.addLine(p4        , p1        )
    wire1 = gmsh.model.occ.addWire([l1, l2, l3, l4])
    face1 = gmsh.model.occ.addPlaneSurface([wire1])

    l5 = gmsh.model.occ.addLine(p12_center, p2        )
    l6 = gmsh.model.occ.addLine(p2        , p3        )
    l7 = gmsh.model.occ.addLine(p3        , p34_center)
    wire2 = gmsh.model.occ.addWire([l5, l6, l7, l2])
    face2 = gmsh.model.occ.addPlaneSurface([wire2])

    depth = depth_top - depth_bottom
    tan_dip = np.tan(dip)
    offset = depth / tan_dip
    dx = offset * np.cos(s)
    dy = -offset * np.sin(s)

    fault_zone1 = gmsh.model.occ.extrude([(2, face1)], dx, dy, (depth_bottom-depth_top)*Magnification)
    fault_zone2 = gmsh.model.occ.extrude([(2, face2)], dx, dy, (depth_bottom-depth_top)*Magnification)
    gmsh.model.occ.synchronize()
    fault_zone = fault_zone1 + fault_zone2
    volume = [e for e in fault_zone if e[0] == 3]  # dim = 3
    return volume[0]


def create_well(config,x_input,y_input,z_coord):
    # Create gmsh point
    well=gmsh.model.occ.addPoint(x_input,y_input,z_coord)
    return well


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build the Peace River Gmsh mesh from local geometry and mesh input files."
    )
    parser.add_argument("--config", default=None, help="Optional local YAML config file.")
    parser.add_argument("--mesh_config", default=None, help="Mesh generator config file.")
    parser.add_argument("--physical_group", default=None, help="Physical group table.")
    parser.add_argument("--input_dir", default=None, help="Directory containing local mesh source CSV and boundary files.")
    parser.add_argument("--out_dir", default=None, help="Directory for generated mesh products.")
    parser.add_argument("--mesh_filename", default="mesh.msh", help="Output mesh filename.")
    parser.add_argument("--volume_tags_filename", default="volume_tags.txt", help="Output volume-tag filename.")
    parser.add_argument("--no_gui", action="store_true", help="Do not open the Gmsh GUI after mesh generation.")
    return parser


def main():
    global CONFIG_PATH, PHYSICAL_GROUP_PATH, MESH_OUTPUT_DIR, MESH_FILE_PATH, VOLUME_TAGS_PATH

    args = build_parser().parse_args()
    local_config = load_local_config(args.config)

    if args.mesh_config:
        CONFIG_PATH = Path(args.mesh_config).expanduser().resolve()
    else:
        candidate = old_workspace_path(local_config, "ogs", "input", "mesh", "scripts", "mesh_generator", "config.txt")
        if candidate is not None and candidate.exists():
            CONFIG_PATH = candidate

    if args.physical_group:
        PHYSICAL_GROUP_PATH = Path(args.physical_group).expanduser().resolve()
    else:
        candidate = old_workspace_path(
            local_config,
            "ogs",
            "input",
            "mesh",
            "scripts",
            "mesh_generator",
            "physical_group.txt",
        )
        if candidate is not None and candidate.exists():
            PHYSICAL_GROUP_PATH = candidate

    if args.out_dir:
        MESH_OUTPUT_DIR = Path(args.out_dir).expanduser().resolve()
    else:
        MESH_OUTPUT_DIR = local_output_path(local_config, "mesh")

    MESH_FILE_PATH = MESH_OUTPUT_DIR / args.mesh_filename
    VOLUME_TAGS_PATH = MESH_OUTPUT_DIR / args.volume_tags_filename

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Mesh config file not found: {CONFIG_PATH}")
    if not PHYSICAL_GROUP_PATH.exists():
        raise FileNotFoundError(f"Physical group file not found: {PHYSICAL_GROUP_PATH}")

    print()
    print("Mesh generation started...")
    print()
    config = read_config(CONFIG_PATH)
    input_dir = resolve_input_dir(config, args.input_dir, local_config)
    MESH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (xx, yy), nx, ny = create_grid(config)
 
    gmsh.initialize()
    
    gmsh.option.setNumber("Geometry.Tolerance", 1e-4)
    gmsh.option.setNumber("Geometry.AutoCoherence", 1)

    gmsh.model.add("PeaceRiver_domain")

    # Load data
    print("  Data loading...")
    print(f"  Input directory: {input_dir}")
    precambrian_df             = pd.read_csv(input_dir / "Precambrian_top.csv")
    elkpoint_df                = pd.read_csv(input_dir / "ElkPoint.csv")
    beaverhilllake_df          = pd.read_csv(input_dir / "BeaverhillLake.csv")
    leduc_df                   = pd.read_csv(input_dir / "Leduc.csv")
    woodbend_df                = pd.read_csv(input_dir / "Woodbend_excluding_Leduc.csv")
    winterburn_df              = pd.read_csv(input_dir / "Winterburn.csv")
    wabamun_df                 = pd.read_csv(input_dir / "Wabamun.csv")
    banff_exshaw_df            = pd.read_csv(input_dir / "Banff_Exshaw.csv")
    permian_stoddart_rundle_df = pd.read_csv(input_dir / "Permian_Stoddart_Rundle.csv")
    land_surface_df            = pd.read_csv(input_dir / "Surface_elevation.csv")
    print("  ...finished.")
    print()


    print("  Thickness calculation...")
    precambrian_top                   = np.full_like(xx, np.nan)
    elkpoint_thickness                = np.full_like(xx, np.nan)
    beaverhilllake_thickness          = np.full_like(xx, np.nan)
    leduc_thickness                   = np.full_like(xx, np.nan)
    woodbend_thickness                = np.full_like(xx, np.nan)
    winterburn_thickness              = np.full_like(xx, np.nan)
    wabamun_thickness                 = np.full_like(xx, np.nan)
    banff_exshaw_thickness            = np.full_like(xx, np.nan)
    permian_stoddart_rundle_thickness = np.full_like(xx, np.nan)
    apply_thickness(elkpoint_df               , elkpoint_thickness               , Magnification, min_thickness=40)
    apply_thickness(beaverhilllake_df         , beaverhilllake_thickness         , Magnification, min_thickness=40)
    apply_thickness(leduc_df                  , leduc_thickness                  , Magnification, min_thickness=40)
    apply_thickness(woodbend_df               , woodbend_thickness               , Magnification, min_thickness=30)
    apply_thickness(winterburn_df             , winterburn_thickness             , Magnification, min_thickness=30)
    apply_thickness(wabamun_df                , wabamun_thickness                , Magnification, min_thickness=30)
    apply_thickness(banff_exshaw_df           , banff_exshaw_thickness           , Magnification, min_thickness=30)
    apply_thickness(permian_stoddart_rundle_df, permian_stoddart_rundle_thickness, Magnification, min_thickness=30)
    print("  ...finished.")
    print()


    print("  Layer surface top generation...")
    def smooth_surface(surface):
        filled = np.where(np.isnan(surface), np.nanmean(surface), surface)
        smoothed = gaussian_filter(filled, sigma=smoothing_intensity)
        return np.where(np.isnan(surface), np.nan, smoothed)
    for _, row in precambrian_df.iterrows():
        precambrian_top[int(row["iy"]), int(row["ix"])] = row["z"] * Magnification
    precambrian_top             = smooth_surface(precambrian_top)
    elkpoint_top                = precambrian_top    + elkpoint_thickness
    elkpoint_top                = smooth_surface(elkpoint_top)
    beaverhilllake_top          = elkpoint_top       + beaverhilllake_thickness
    beaverhilllake_top          = smooth_surface(beaverhilllake_top)
    leduc_top                   = beaverhilllake_top + leduc_thickness
    leduc_top                   = smooth_surface(leduc_top)
    woodbend_top                = leduc_top          + woodbend_thickness
    woodbend_top                = smooth_surface(woodbend_top)
    winterburn_top              = woodbend_top       + winterburn_thickness
    winterburn_top              = smooth_surface(winterburn_top)
    wabamun_top                 = winterburn_top     + wabamun_thickness
    wabamun_top                 = smooth_surface(wabamun_top)
    banff_exshaw_top            = wabamun_top        + banff_exshaw_thickness
    banff_exshaw_top            = smooth_surface(banff_exshaw_top)
    permian_stoddart_rundle_top = banff_exshaw_top   + permian_stoddart_rundle_thickness

    print("  ...finished.")
    print()


    print("  Land surface generation...")
    land_surface_z = np.full_like(xx, np.nan)
    for _, row in land_surface_df.iterrows():
        ix, iy = int(row["ix"]), int(row["iy"])
        land_surface_z[iy, ix] = row["z"] * Magnification
    print("  ...finished.")
    print()


    # Grid info
    dx = xx[0, 1] - xx[0, 0]
    dy = yy[1, 0] - yy[0, 0]
    x0, y0 = xx[0, 0], yy[0, 0]
    min_z = np.nanmin(precambrian_top) # Lowest elevation in precambrian_top


    # === ElkPoint volume ===
    def elkpoint():
        print("  Processing ElkPoint...")
        elkpoint_surf    = (2,create_bspline_surface(xx, yy, elkpoint_top, nx, ny))
        precambrian_surf = (2,create_bspline_surface(xx, yy, precambrian_top, nx, ny))
        gmsh.model.occ.synchronize()
        # Generate masked temporary volume
        create_mask_box_from_boundary(input_dir / "ElkPoint_boundary_nodes.txt", dx, dy, x0, y0, min_z-temp_box_top_z-1000)
        # Extract the target layer volume
        all_volumes=gmsh.model.getEntities(dim=3)
        entities_to_fragment=[precambrian_surf]+[elkpoint_surf]
        out,_=gmsh.model.occ.fragment(all_volumes,entities_to_fragment)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.remove([(3,1),(3,3)], recursive=True)
        gmsh.model.occ.remove([(2,36)], recursive=True)
        precambrian_surf=[(2,35)]
        gmsh.model.occ.synchronize()

        # Clean up
        all_volumes=gmsh.model.getEntities(dim=3)
        entities_to_leave=all_volumes+precambrian_surf
        remove_list=[ent for ent in gmsh.model.getEntities() if ent not in entities_to_leave]
        gmsh.model.removeEntities(remove_list,recursive=True)
        print("  ...finished.")
        print()
        return precambrian_surf


    # === BeaverhillLake volume ===
    def beaverhilllake(precambrian_surf):
        print("  Processing BeaverhillLake...")
        beaverhilllake_surf = (2,create_bspline_surface(xx, yy, beaverhilllake_top, nx, ny))
        gmsh.model.occ.synchronize()
        # Generate masked temporary volume
        create_mask_box_from_boundary(input_dir / "BeaverhillLake_boundary_nodes.txt", dx, dy, x0, y0, min_z-temp_box_top_z-1000)
        # Extract the target layer volume
        all_volumes=gmsh.model.getEntities(dim=3)
        entities_to_fragment=precambrian_surf+[beaverhilllake_surf]
        out,_=gmsh.model.occ.fragment(all_volumes,entities_to_fragment)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.remove([(3,3),(3,5)], recursive=True)
        gmsh.model.occ.remove([(2,74)], recursive=True)
        precambrian_surf=[(2,73)]
        gmsh.model.occ.synchronize()

        # Clean up
        entities_to_leave=gmsh.model.getEntities(dim=3)+precambrian_surf
        remove_list=[ent for ent in gmsh.model.getEntities() if ent not in entities_to_leave]
        gmsh.model.removeEntities(remove_list,recursive=True)
        print("  ...finished.")
        print()
        return precambrian_surf


    # === Leduc volume ===
    def leduc(precambrian_surf):
        print("  Processing Leduc...")
        leduc_surf = (2,create_bspline_surface(xx, yy, leduc_top, nx, ny))
        gmsh.model.occ.synchronize()
        # Generate masked temporary volume
        create_mask_box_from_boundary(input_dir / "Leduc_boundary_nodes.txt", dx, dy, x0, y0, min_z-temp_box_top_z-1000)
        # Extract the target layer volume
        all_volumes=gmsh.model.getEntities(dim=3)
        entities_to_fragment=precambrian_surf+[leduc_surf]
        out,_=gmsh.model.occ.fragment(all_volumes,entities_to_fragment)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.remove([(3,5),(3,7)], recursive=True)
        gmsh.model.occ.remove([(2,125),(2,126)], recursive=True)
        precambrian_surf=[(2,123),(2,124)]
        gmsh.model.occ.synchronize()

        # Clean up
        entities_to_leave=gmsh.model.getEntities(dim=3)+precambrian_surf
        remove_list=[ent for ent in gmsh.model.getEntities() if ent not in entities_to_leave]
        gmsh.model.removeEntities(remove_list,recursive=True)
        print("  ...finished.")
        print()
        return precambrian_surf


    # === Woodbend(except for Leduc) volume ===
    def woodbend(precambrian_surf):
        print("  Processing Woodbend(except for Leduc)...")
        woodbend_surf = (2,create_bspline_surface(xx, yy, woodbend_top, nx, ny))
        gmsh.model.occ.synchronize()
        # Generate masked temporary volume
        masked_box = create_mask_box_from_boundary(input_dir / "Woodbend_excluding_Leduc_boundary_nodes.txt", dx, dy, x0, y0, min_z-temp_box_top_z-1000)
        # Extract the target layer volume
        all_volumes=gmsh.model.getEntities(dim=3)
        entities_to_fragment=precambrian_surf+[woodbend_surf]
        out,_=gmsh.model.occ.fragment(all_volumes,entities_to_fragment)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.remove([(3,7),(3,9)], recursive=True)
        gmsh.model.occ.remove([(2,238)], recursive=True)
        precambrian_surf=[(2,123)]
        gmsh.model.occ.synchronize()

        # Clean up
        entities_to_leave=gmsh.model.getEntities(dim=3)+precambrian_surf
        remove_list=[ent for ent in gmsh.model.getEntities() if ent not in entities_to_leave]
        gmsh.model.removeEntities(remove_list,recursive=True)
        print("  ...finished.")
        print()
        return precambrian_surf


    # === Winterburn and Wabamun volume ===
    def winterburn_wabamun(precambrian_surf):
        print("  Processing Winterburn and Wabamun...")
        wabamun_surf    = (2,create_bspline_surface(xx, yy, wabamun_top   , nx, ny))
        winterburn_surf = (2,create_bspline_surface(xx, yy, winterburn_top, nx, ny))
        gmsh.model.occ.synchronize()
        # Generate masked temporary volume
        create_mask_box_from_boundary(input_dir / "Winterburn_boundary_nodes.txt", dx, dy, x0, y0, min_z-temp_box_top_z-1000)
        # Extract the target layer volume
        all_volumes=gmsh.model.getEntities(dim=3)
        entities_to_fragment=precambrian_surf+[winterburn_surf]+[wabamun_surf]
        out,_=gmsh.model.occ.fragment(all_volumes,entities_to_fragment)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.remove([(3,9),(3,12)], recursive=True)
        gmsh.model.occ.synchronize()

        # Clean up
        entities_to_leave=gmsh.model.getEntities(dim=3)
        remove_list=[ent for ent in gmsh.model.getEntities() if ent not in entities_to_leave]
        gmsh.model.removeEntities(remove_list,recursive=True)
        print("  ...finished.")
        print()
        return


    # === Banff_Exshaw volume ===
    def banff_exshaw():
        print("  Processing Banff_Exshaw...")
        banff_exshaw_surf = (2,create_bspline_surface(xx, yy, banff_exshaw_top, nx, ny))
        gmsh.model.occ.synchronize()
        # Generate masked temporary volume
        create_mask_box_from_boundary(input_dir / "Banff_Exshaw_boundary_nodes.txt", dx, dy, x0, y0, min_z-temp_box_top_z-1000)
        # Extract the target layer volume
        all_volumes=gmsh.model.getEntities(dim=3)
        entities_to_fragment=[banff_exshaw_surf]
        out,_=gmsh.model.occ.fragment(all_volumes,entities_to_fragment)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.remove([(3,17),(3,19)], recursive=True)
        gmsh.model.occ.synchronize()

        # Clean up
        entities_to_leave=gmsh.model.getEntities(dim=3)
        remove_list=[ent for ent in gmsh.model.getEntities() if ent not in entities_to_leave]
        gmsh.model.removeEntities(remove_list,recursive=True)
        print("  ...finished.")
        print()
        return


    # === Permian_Stoddart_Rundle volume ===
    def permian_stoddart_rundle():
        print("  Processing Permian_Stoddart_Rundle...")
        permian_stoddart_rundle_surf = (2,create_bspline_surface(xx, yy, permian_stoddart_rundle_top, nx, ny))
        gmsh.model.occ.synchronize()
        # Generate masked temporary volume
        create_mask_box_from_boundary(input_dir / "Permian_Stoddart_Rundle_boundary_nodes.txt", dx, dy, x0, y0, min_z-temp_box_top_z-1000)
        # Extract the target layer volume
        all_volumes=gmsh.model.getEntities(dim=3)
        entities_to_fragment=[permian_stoddart_rundle_surf]
        out,_=gmsh.model.occ.fragment(all_volumes,entities_to_fragment)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.remove([(3,30),(3,32)], recursive=True)
        # gmsh.model.occ.remove([(3,31)], recursive=True)
        gmsh.model.occ.synchronize()

        # Clean up
        entities_to_leave=gmsh.model.getEntities(dim=3)
        remove_list=[ent for ent in gmsh.model.getEntities() if ent not in entities_to_leave]
        gmsh.model.removeEntities(remove_list,recursive=True)
        print("  ...finished.")
        print()
        return


    # === Precambrian basement ===
    def precambrian():
        print("  Processing Precambrian basement...")
        # Temp volume
        domain_box = gmsh.model.occ.addBox(x0, y0, bottom_z, xx.max() - x0, yy.max() - y0, 2*np.abs(bottom_z))
        gmsh.model.occ.synchronize()
        # Fragment
        all_volumes=gmsh.model.getEntities(dim=3)
        entities_to_fragment=all_volumes
        out, _ = gmsh.model.occ.fragment([(3, domain_box)], entities_to_fragment)
        gmsh.model.occ.synchronize()
        # Delete unnecesarry volumes
        gmsh.model.occ.remove([(3,33)], recursive=True)
        gmsh.model.occ.synchronize()

        # Clean up
        entities_to_leave=gmsh.model.getEntities(dim=3)
        remove_list=[ent for ent in gmsh.model.getEntities() if ent not in entities_to_leave]
        gmsh.model.removeEntities(remove_list,recursive=True)
        print("  ...finished.")
        print()
        return
    

    # === Precambrian basement ===
    def temp_precambrian(precambrian_surf):
        print("  Processing temp Precambrian basement...")

        # Land surface
        land_surface_surf = (2, create_bspline_surface(xx, yy, land_surface_z, nx, ny))
        gmsh.model.occ.synchronize()
        # Temp volume
        gmsh.model.occ.addBox(x0, y0, bottom_z, xx.max() - x0, yy.max() - y0, 2*np.abs(bottom_z))
        gmsh.model.occ.synchronize()
        # Fragment
        all_volumes=gmsh.model.getEntities(dim=3)
        entities_to_fragment=precambrian_surf+[land_surface_surf]
        out, _ = gmsh.model.occ.fragment(all_volumes, entities_to_fragment)
        gmsh.model.occ.synchronize()
        # Delete unnecesarry volumes
        fragmented_volumes = [ent for ent in out if ent[0] == 3]
        z=[]
        for dim,tag in fragmented_volumes:
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            z.append((tag,com[2]))
        z.sort(key=lambda x: x[1], reverse=True)
        gmsh.model.occ.remove([(3,z[0][0])], recursive=True)
        gmsh.model.occ.synchronize()

        # Clean up
        entities_to_leave=gmsh.model.getEntities(dim=3)
        remove_list=[ent for ent in gmsh.model.getEntities() if ent not in entities_to_leave]
        gmsh.model.removeEntities(remove_list,recursive=True)

        print("  ...finished.")
        print()
        return
    

    def add_3d_faults():
        print("  Processing faults...")

        # Common settings
        FGZ_thickness=100; FDZ_thickness=300; depth_top=6000; depth_bottom=-7000

        # Fault1
        print("    Fault 1")
        xc=392541; yc=6214304-1000; length=24780; strike=290; dip=60
        f1_gz=create_3d_fault_gouge(config,xc,yc,length                ,FGZ_thickness,depth_top,depth_bottom,strike,dip)
        f1_dz=create_3d_fault      (config,xc,yc,length+FDZ_thickness,FDZ_thickness,depth_top,depth_bottom,strike,dip)

        # Fault2
        print("    Fault 2")
        xc=392003; yc=6212326-1000; length=21660; strike=288; dip=60
        f2_gz=create_3d_fault_gouge(config,xc,yc,length                ,FGZ_thickness,depth_top,depth_bottom,strike,dip)
        f2_dz=create_3d_fault      (config,xc,yc,length+FDZ_thickness,FDZ_thickness,depth_top,depth_bottom,strike,dip)

        # Fault3
        print("    Fault 3")
        xc=378551+2000; yc=6206059-2000; length=6860; strike=235.3; dip=60
        f3_gz=create_3d_fault_gouge(config,xc,yc,length                ,FGZ_thickness,depth_top,depth_bottom,strike,dip)
        f3_dz=create_3d_fault      (config,xc,yc,length+FDZ_thickness,FDZ_thickness,depth_top,depth_bottom,strike,dip)

        # Fault4
        print("    Fault 4")
        xc=385625-1500; yc=6201477+3000; length=21380; strike=106.9; dip=60
        f4_gz=create_3d_fault_gouge(config,xc,yc,length                ,FGZ_thickness,depth_top,depth_bottom,strike,dip)
        f4_dz=create_3d_fault      (config,xc,yc,length+FDZ_thickness,FDZ_thickness,depth_top,depth_bottom,strike,dip)

        # Fault5
        print("    Fault 5")
        xc=388258+2000; yc=6204800-2000; length=14820; strike=250.6; dip=60
        f5_gz=create_3d_fault_gouge(config,xc,yc,length                ,FGZ_thickness,depth_top,depth_bottom,strike,dip)
        f5_dz=create_3d_fault      (config,xc,yc,length+FDZ_thickness,FDZ_thickness,depth_top,depth_bottom,strike,dip)

        # Fault6
        print("    Fault 6")
        xc=391226-7000; yc=6194604+3000; length=8990; strike=107.1; dip=60
        f6_gz=create_3d_fault_gouge(config,xc,yc,length                ,FGZ_thickness,depth_top,depth_bottom,strike,dip)
        f6_dz=create_3d_fault      (config,xc,yc,length+FDZ_thickness,FDZ_thickness,depth_top,depth_bottom,strike,dip)

        print("    fragmenting faults...")
        entities_to_fragment=\
             [f1_gz]+[f1_dz]\
            +[f2_gz]+[f2_dz]\
            +[f3_gz]+[f3_dz]\
            +[f4_gz]+[f4_dz]\
            +[f5_gz]+[f5_dz]\
            +[f6_gz]+[f6_dz]
        all_volumes=gmsh.model.getEntities(dim=3)
        out, _ = gmsh.model.occ.fragment(all_volumes, entities_to_fragment)
        gmsh.model.occ.synchronize()

        # Remove the unnecessary temp volumes
        all_volumes=gmsh.model.getEntities(dim=3)
        z=[]
        for dim,tag in all_volumes:
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            z.append((tag,com[2]))
        z.sort(key=lambda x: x[1], reverse=True)
        for i in range(48):
            gmsh.model.occ.remove([(3,z[i][0])], recursive=True)
        gmsh.model.occ.synchronize()

        # Clean up
        entities_to_leave=gmsh.model.getEntities(dim=3)
        remove_list=[ent for ent in gmsh.model.getEntities() if ent not in entities_to_leave]
        gmsh.model.removeEntities(remove_list,recursive=True)
        print("  ...finished.")
        print()
        return


    # === Land surface ===
    def land_surface():
        print("  Processing Land surface...")

        # Land surface
        land_surface_surf = (2, create_bspline_surface(xx, yy, land_surface_z, nx, ny))
        gmsh.model.occ.synchronize()
        # Temp volume
        domain_box = gmsh.model.occ.addBox(x0, y0, bottom_z, xx.max() - x0, yy.max() - y0, 2*np.abs(bottom_z))
        gmsh.model.occ.synchronize()
        # Fragment
        all_volumes=gmsh.model.getEntities(dim=3)
        entities_to_fragment=all_volumes+[land_surface_surf]
        out, _ = gmsh.model.occ.fragment([(3, domain_box)], entities_to_fragment)
        gmsh.model.occ.synchronize()
        # Delete unnecesarry volumes
        fragmented_volumes = [ent for ent in out if ent[0] == 3]
        z=[]
        for dim,tag in fragmented_volumes:
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            z.append((tag,com[2]))
        z.sort(key=lambda x: x[1], reverse=True)
        gmsh.model.occ.remove([(3,z[0][0])], recursive=True)
        gmsh.model.occ.synchronize()

        # Clean up
        entities_to_leave=gmsh.model.getEntities(dim=3)
        remove_list=[ent for ent in gmsh.model.getEntities() if ent not in entities_to_leave]
        gmsh.model.removeEntities(remove_list,recursive=True)

        print("  ...finished.")
        print()
        return


    # === Exctract local domain ===
    def internal_domain():
        print("  Processing internal domain...")
        # Generate masked temporary volume
        masked_box = create_mask_box_from_boundary(input_dir / "Domain_boundary_nodes.txt", dx, dy, x0, y0, bottom_z-temp_box_top_z*2)
        masked_box_volume = [v for v in masked_box if v[0] == 3]
        # Extract the target layer volume
        all_volumes=gmsh.model.getEntities(dim=3)
        entities_to_fragment=all_volumes
        out,_=gmsh.model.occ.fragment(masked_box_volume,entities_to_fragment)
        gmsh.model.occ.synchronize()

        # Delete outside volumes
        gmsh.model.occ.remove([(3,   8)], recursive=True)
        gmsh.model.occ.remove([(3,  10)], recursive=True)
        gmsh.model.occ.remove([(3,  12)], recursive=True)
        gmsh.model.occ.remove([(3,  14)], recursive=True)
        gmsh.model.occ.remove([(3,  16)], recursive=True)
        gmsh.model.occ.remove([(3,  17)], recursive=True)
        gmsh.model.occ.remove([(3,  18)], recursive=True)
        gmsh.model.occ.remove([(3,  19)], recursive=True)
        gmsh.model.occ.remove([(3,  21)], recursive=True)
        gmsh.model.occ.remove([(3,  23)], recursive=True)
        gmsh.model.occ.remove([(3,  25)], recursive=True)
        gmsh.model.occ.remove([(3,  27)], recursive=True)
        gmsh.model.occ.remove([(3,  29)], recursive=True)
        gmsh.model.occ.remove([(3, 184)], recursive=True)
        gmsh.model.occ.remove([(3, 199)], recursive=True)
        gmsh.model.occ.remove([(3, 200)], recursive=True)
        gmsh.model.occ.remove([(3, 201)], recursive=True)
        gmsh.model.occ.remove([(3, 202)], recursive=True)
        gmsh.model.occ.remove([(3, 203)], recursive=True)
        gmsh.model.occ.remove([(3, 204)], recursive=True)
        gmsh.model.occ.remove([(3, 205)], recursive=True)
        gmsh.model.occ.remove([(3, 206)], recursive=True)
        gmsh.model.occ.remove([(3, 207)], recursive=True)
        gmsh.model.occ.remove([(3, 208)], recursive=True)
        gmsh.model.occ.remove([(3, 209)], recursive=True)
        gmsh.model.occ.remove([(3, 210)], recursive=True)
        gmsh.model.occ.remove([(3, 211)], recursive=True)
        gmsh.model.occ.remove([(3, 212)], recursive=True)
        gmsh.model.occ.remove([(3, 213)], recursive=True)
        gmsh.model.occ.remove([(3, 214)], recursive=True)
        gmsh.model.occ.remove([(3, 215)], recursive=True)
        gmsh.model.occ.synchronize()

        # Clean up
        entities_to_leave=gmsh.model.getEntities(dim=3)
        remove_list=[ent for ent in gmsh.model.getEntities() if ent not in entities_to_leave]
        gmsh.model.removeEntities(remove_list,recursive=True)
        print("  ...finished.")
        print()
        return
    
    def add_wells():
        print("Processing wells...")

        well_definitions = [
            # Permian_Stoddart_Rundle
            {"name": "PSR_BAYTEX_HZ_SWD_SEAL_4_4_84_19"    , "x": 379176.9805, "y": 6232764.5594, "layer_name": "Permian_Stoddart_Rundle"},
            {"name": "PSR_PENN_WEST_PEACE_RIVER_4_24_84_19", "x": 384174.1043, "y": 6237545.0809, "layer_name": "Permian_Stoddart_Rundle"},
            {"name": "PSR_PENN_WEST_PEACE_RIVER_4_32_83_18", "x": 387215.3151, "y": 6231096.3258, "layer_name": "Permian_Stoddart_Rundle"},
            {"name": "PSR_CNUL_CADOTTE_5_19_85_18"         , "x": 386231.9980, "y": 6247788.6126, "layer_name": "Permian_Stoddart_Rundle"},
            {"name": "PSR_CNUL_CADOTTE_11_19_85_18"        , "x": 386584.2728, "y": 6247937.2561, "layer_name": "Permian_Stoddart_Rundle"},
            {"name": "PSR_BAYTEX_HARMONV_6_14_82_18_near"  , "x": 393887.8927, "y": 6216878.5160, "layer_name": "Permian_Stoddart_Rundle"},
            {"name": "PSR_BAYTEX_HARMONV_6_14_82_18_far"   , "x": 397887.8927, "y": 6222878.5160, "layer_name": "Permian_Stoddart_Rundle"},
            {"name": "PSR_PENN_WEST_HARMONV_12_26_82_20"   , "x": 373998.9354, "y": 6220866.6066, "layer_name": "Permian_Stoddart_Rundle"},
            # Wabamun
            {"name": "Wm_BAYTEX_CADOTTE_10_28_84_18"       , "x": 390230.4137, "y": 6240099.6064, "layer_name": "Wabamun"},
            # Leduc
            {"name": "Le_CCS_CADOTTE_2_20_85_19"           , "x": 379238.6331, "y": 6247374.1882, "layer_name": "Leduc"},
            {"name": "Le_CNUL_CADOTTE_16_23_85_19"         , "x": 384091.3078, "y": 6248595.7058, "layer_name": "Leduc"},
            {"name": "Le_CNUL_CADOTTE_14_25_85_19"         , "x": 385048.9190, "y": 6250142.5944, "layer_name": "Leduc"},
            {"name": "Le_BAYTEX_RE_SWD_SEAL_6_29_84_18"    , "x": 387960.2534, "y": 6239593.2876, "layer_name": "Leduc"},
            {"name": "Le_CNUL_CADOTTE_2_22_84_17"          , "x": 401249.2441, "y": 6237266.4136, "layer_name": "Leduc"},
            {"name": "Le_CNUL_CADOTTE_13_11_84_17"         , "x": 402078.2750, "y": 6235175.9772, "layer_name": "Leduc"},
            {"name": "Le_PENN_WEST_WALRUS_14_18_82_17_near", "x": 397175.4564, "y": 6217626.5334, "layer_name": "Leduc"},
            {"name": "Le_PENN_WEST_WALRUS_14_18_82_17_far" , "x": 398675.4564, "y": 6224626.5334, "layer_name": "Leduc"},
            # Observation well
            {"name": "Obs_well_western_cluster"    , "x": 366279, "y": 6240473, "layer_name": "Leduc"},
            {"name": "Obs_well_eastern_cluster"    , "x": 400460, "y": 6233508, "layer_name": "Leduc"}
        ]

        # 地層名と対応するトップマップ・厚さマップの対応辞書
        layer_top_map = {
            "Permian_Stoddart_Rundle": permian_stoddart_rundle_top,
            "Banff_Exshaw": banff_exshaw_top,
            "Wabamun": wabamun_top,
            "Winterburn": winterburn_top,
            "Woodbend": woodbend_top,
            "Leduc": leduc_top,
            "BeaverhillLake": beaverhilllake_top,
            "ElkPoint": elkpoint_top,
        }
        layer_thickness_map = {
            "Permian_Stoddart_Rundle": permian_stoddart_rundle_thickness,
            "Banff_Exshaw": banff_exshaw_thickness,
            "Wabamun": wabamun_thickness,
            "Winterburn": winterburn_thickness,
            "Woodbend": woodbend_thickness,
            "Leduc": leduc_thickness,
            "BeaverhillLake": beaverhilllake_thickness,
            "ElkPoint": elkpoint_thickness,
        }

        x_coords = xx[0, :]
        y_coords = yy[:, 0]

        interpolators = {}
        for name in layer_top_map.keys():
            top_interp = RegularGridInterpolator((y_coords, x_coords), layer_top_map[name])
            thickness_interp = RegularGridInterpolator((y_coords, x_coords), layer_thickness_map[name])
            interpolators[name] = (top_interp, thickness_interp)

        # Coordinate transformation
        x_std=config['x_corner_pt']
        y_std=config['y_corner_pt']
        strike_std=np.radians(config['strike_deg'])

        wells=[]
        for well_def in well_definitions:
            well_name = well_def["name"]
            x_input_raw = well_def["x"]
            y_input_raw = well_def["y"]
            layer_name = well_def["layer_name"]

            # Coordinate transformation
            x_input = (x_input_raw - x_std) * np.cos(strike_std) - (y_input_raw - y_std) * np.sin(strike_std)
            y_input = (x_input_raw - x_std) * np.sin(strike_std) + (y_input_raw - y_std) * np.cos(strike_std)

            if layer_name not in interpolators:
                raise ValueError(f"Layer name {layer_name} not found.")
            
            top_interp, thickness_interp = interpolators[layer_name]

            point = np.array([[y_input, x_input]])
            top_z = top_interp(point)[0]
            thickness = thickness_interp(point)[0]

            center_z = top_z - thickness / 2

            well = create_well(config, x_input, y_input, center_z)
            wells.append((well_name,well))
        gmsh.model.occ.synchronize()

        print("  ...finished.")
        print()
        return wells
    

    def add_physical_group(wells):
        import csv

        print("  Processing physical group...")

        # Permian_Stoddart_Rundle
        gmsh.model.mesh.embed(0,[wells[0][1]],3,186)
        gmsh.model.mesh.embed(0,[wells[1][1]],3,186)
        gmsh.model.mesh.embed(0,[wells[2][1]],3,186)
        gmsh.model.mesh.embed(0,[wells[3][1]],3,186)
        gmsh.model.mesh.embed(0,[wells[4][1]],3,186)
        gmsh.model.mesh.embed(0,[wells[5][1]],3,186)
        gmsh.model.mesh.embed(0,[wells[6][1]],3,186)
        # Wabamun
        gmsh.model.mesh.embed(0,[wells[7][1]],3,188)
        # Leduc
        gmsh.model.mesh.embed(0,[wells[ 8][1]],3,197)
        gmsh.model.mesh.embed(0,[wells[ 9][1]],3,197)
        gmsh.model.mesh.embed(0,[wells[10][1]],3,197)
        gmsh.model.mesh.embed(0,[wells[11][1]],3,197)
        gmsh.model.mesh.embed(0,[wells[12][1]],3,197)
        gmsh.model.mesh.embed(0,[wells[13][1]],3,197)
        gmsh.model.mesh.embed(0,[wells[14][1]],3,197)
        # Observation well
        gmsh.model.mesh.embed(0,[wells[15][1]],3,192)
        gmsh.model.mesh.embed(0,[wells[16][1]],3,197)

        gmsh.model.occ.synchronize()


        i=0
        for well_name, well_tag in wells:
            gmsh.model.addPhysicalGroup(0, [well_tag], tag=200+i, name=well_name)
            i=i+1

        # volume (dim=3) に物理グループを付与
        physical_group_map = defaultdict(list)
        with open(PHYSICAL_GROUP_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                tag = int(row["Tag"])
                name = row["Physical_group"]
                physical_group_map[name].append(tag)
        
        volume_tag = {
            "Overburden": 0,
            "Permian_Stoddart_Rundle": 1,
            "Banff_Exshaw": 2,
            "Wabamun": 3,
            "Winterburn": 4,
            "Woodbend": 5,
            "Leduc": 6,
            "Beaverhilllake": 7,
            "Precambrian_basement": 8,
            "F1_G": 9,
            "F2_G": 10,
            "F3_G": 11,
            "F4_G": 12,
            "F5_G": 13,
            "F6_G": 14,
            "F1_D": 15,
            "F2_D": 16,
            "F3_D": 17,
            "F4_D": 18,
            "F5_D": 19,
            "F6_D": 20
        }

        for name, tags in physical_group_map.items():
            # print(tags, name)
            gmsh.model.addPhysicalGroup(3, tags, tag=volume_tag[name], name=name)
            gmsh.model.setPhysicalName(3, volume_tag[name], name)

        # Land surface
        gmsh.model.addPhysicalGroup(2, [1319], tag=100, name="Land_surface")

        # Bottom surface
        gmsh.model.addPhysicalGroup(2, [1442], tag=101, name="Bottom")

        # Fault gouge plane
        tags = [1125,1033,862,951,765,596,489,1220]
        gmsh.model.addPhysicalGroup(2, tags, tag=110, name="F1_gouge_center_plane")
        tags = [484,565,728,818,921,1000,1092,1189]
        gmsh.model.addPhysicalGroup(2, tags, tag=111, name="F1_gouge_south_plane")
        tags = [494,568,731,821,924,1003,1095,1192]
        gmsh.model.addPhysicalGroup(2, tags, tag=112, name="F1_gouge_north_plane")
        tags = [1220]
        gmsh.model.addPhysicalGroup(2, tags, tag=113, name="F1_gouge_center_basement_plane")

        # Leduc top plane
        tags = [1368,1369,1370,1387]
        gmsh.model.addPhysicalGroup(2, tags, tag=114, name="Leduc_top")

        # Side surfaces
        tags = [1328,1334,1340,1346,1352,1376,1434,1441]
        gmsh.model.addPhysicalGroup(2, tags, tag=115, name="SS1")
        tags = [1327,1333,1339,1345,1351,1375,1378,1426,1427,1436,1448,1452,1453,1459]
        gmsh.model.addPhysicalGroup(2, tags, tag=116, name="SS2")
        tags = [1326,1332,1338,1344,1350,1374,1377,1423,1410,1411,1444,1450,1451]
        gmsh.model.addPhysicalGroup(2, tags, tag=117, name="SS3")
        tags = [1323,1329,1335,1341,1347,1371,1407,1443]
        gmsh.model.addPhysicalGroup(2, tags, tag=118, name="SS4")
        tags = [1324,1330,1336,1342,1348,1372,1408,1433,1440]
        gmsh.model.addPhysicalGroup(2, tags, tag=119, name="SS5")

        print("  ...finished.")
        print()
        return

    precambrian_surf = elkpoint()
    precambrian_surf = beaverhilllake(precambrian_surf)
    precambrian_surf = leduc(precambrian_surf)
    precambrian_surf = woodbend(precambrian_surf)
    winterburn_wabamun(precambrian_surf)
    banff_exshaw()
    permian_stoddart_rundle()
    precambrian()
    # temp_precambrian(precambrian_surf)

    add_3d_faults()

    land_surface()

    internal_domain()

    wells=add_wells()

    add_physical_group(wells)

    gmsh.model.occ.remove([(2, 276)], recursive=True)
    gmsh.model.occ.remove([(2, 410)], recursive=True)
    gmsh.model.occ.synchronize()

    # Set mesh size
    def set_mesh_size_by_volume(vol_tag, size):
        surfaces = gmsh.model.getBoundary([(3, vol_tag)], oriented=False, recursive=True)
        points = set()
        for s_dim, s_tag in surfaces:
            edges = gmsh.model.getBoundary([(s_dim, s_tag)], oriented=False, recursive=True)
            for e_dim, e_tag in edges:
                if e_dim == 0:
                    points.add(e_tag)
        for pt in points:
            gmsh.model.mesh.setSize([(0, pt)], size)
        print(f"Volume {vol_tag} → {len(points)} points set to mesh size {size}")
        return

    def apply_mesh_size_from_physical_group(path=PHYSICAL_GROUP_PATH):
        import csv
        print("  Applying mesh sizes by volume tags...")

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                tag = int(row["Tag"])
                size = float(row["Mesh_size"])
                set_mesh_size_by_volume(tag, size)
        return
    
    apply_mesh_size_from_physical_group()


    # Output tags to text file
    all_volumes=gmsh.model.getEntities(dim=3)
    volume_info=[]
    for dim,tag in all_volumes:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        mass_properties=gmsh.model.occ.getMass(dim,tag)
        volume_info.append({
            "tag": tag,
            "x": com[0],
            "y": com[1],
            "z": com[2],
            "volume": mass_properties
        })
    volume_info.sort(key=lambda v: v["z"], reverse=True)
    with open(VOLUME_TAGS_PATH, "w", encoding="utf-8") as f:
        f.write("Tag,X,Y,Z,Volume\n")
        for info in volume_info:
            f.write(f"{info['tag']}, {info['x']}, {info['y']}, {info['z']}, {info['volume']}\n")
    

    # gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(MESH_FILE_PATH))
    print(f"  Mesh written to: {MESH_FILE_PATH}")
    if not args.no_gui:
        gmsh.fltk.run()
    gmsh.finalize()

if __name__ == "__main__":
    main()
