# Workflow Guide

The commands below assume that the software and data packages are available as
sibling directories:

```text
<RELEASE_ROOT>/
  code_package/
  data_package/
```

Run the commands from `<RELEASE_ROOT>`. Generated files should be written to a
user-selected output directory.

## Package layout

The principal software directories are:

- `code_package/scripts/mesh/`: model-mesh preparation and rescaling.
- `code_package/scripts/ogs_input/`: hydraulic ensemble and OGS input
  preparation.
- `code_package/scripts/postprocess/`: pressure-series merging, resampling, and
  export.
- `code_package/scripts/analysis/`: pressure-contribution, paired-pressure, and
  depth-split analyses.
- `code_package/scripts/figures/`: pressure-dominance and Supporting
  Information figures.
- `code_package/config/`: an example configuration for inputs stored outside
  the packages.
- `code_package/environment/`: Conda environment specifications.

Copy `code_package/config/local_paths.example.yml` to a local path and fill in
only the entries required by the workflow being run. Licensed data, model
outputs, and private absolute paths should remain outside redistributed copies
of the packages.

## Public data inputs

The sibling data package provides three scientific CSV files:

- `data_package/params_design_reference.csv`: the 1,000-realization hydraulic
  ensemble design.
- `data_package/paired_pressure_differences.csv`: November 2022 paired pressure
  differences for 1,000 realizations and seven modeled source groups.
- `data_package/southern_cluster_monthly_counts.csv`: monthly earthquake counts
  for the southern Peace River cluster.

`paired_pressure_differences.csv` contains `case_id`, `source_group`, and
`paired_difference_kPa`. The figure scripts validate the case inventory,
source groups, finite values, and expected pressure-dominance class counts.

## SI Figure S3

SI Figure S3 joins the paired pressure differences with the hydraulic ensemble
design:

```bash
python code_package/scripts/figures/plot_well15_paired_pressure_parameter_grid.py \
  --paired-csv data_package/paired_pressure_differences.csv \
  --params-csv data_package/params_design_reference.csv \
  --output-dir <OUTPUT_DIR>/parameter_grid
```

These two CSV files contain all scientific inputs needed by this plotting
workflow.

## SI Figure S5

SI Figure S5 calculates the empirical cumulative distribution functions for
all seven modeled source groups directly from the paired-pressure CSV:

```bash
python code_package/scripts/figures/plot_paired_difference_ecdf.py \
  --paired-csv data_package/paired_pressure_differences.csv \
  --output <OUTPUT_DIR>/paired_difference_ecdf_pressure.pdf
```

## Main Figure 3

The paired classification and the Well 10 and Well 15 ECDF panels use
`data_package/paired_pressure_differences.csv`. Generating the full spatial
figure additionally requires locally available model pressure arrays and
spatial overlay data specified through the local configuration:

```bash
python code_package/scripts/figures/plot_well15_paired_sign_pressure_maps.py \
  --config <LOCAL_PATH_CONFIG.yml> \
  --paired-csv data_package/paired_pressure_differences.csv \
  --output-dir <OUTPUT_DIR>/figure3
```

The script uses the November 2022 pressure output for both the classification
and displayed conditional pressure fields.

## Other workflows requiring external or local inputs

### SI Figure S2

SI Figure S2 requires the licensed injection dataset supplied through a local
path:

```bash
python code_package/scripts/figures/plot_injection_timeseries_panels.py \
  --injection-csv <LICENSED_INJECTION_CSV> \
  --output <OUTPUT_DIR>/monthly_injection_stacked_panels.pdf
```

### SI Figure S4

SI Figure S4 requires the pressure-contribution summary produced from local
processed model outputs:

```bash
python code_package/scripts/figures/plot_component_timeseries_panels.py \
  --analysis-dir <LOCAL_CONTRIBUTION_ANALYSIS_DIR> \
  --output <OUTPUT_DIR>/component_timeseries_panels_pressure.pdf
```

The summary can be prepared from source-decomposed pressure time series with:

```bash
python code_package/scripts/analysis/analyze_fault_pressure_contributions.py \
  --input-dir <LOCAL_PRESSURE_TIMESERIES_DIR> \
  --output-dir <LOCAL_CONTRIBUTION_ANALYSIS_DIR>
```

### Paired-pressure and depth-split analyses

The full paired-pressure analysis uses locally available source-decomposed
pressure time series:

```bash
python code_package/scripts/analysis/analyze_paired_component_differences.py \
  --input-dir <LOCAL_PRESSURE_TIMESERIES_DIR> \
  --output-dir <OUTPUT_DIR>/paired_component_differences
```

The Well 15 depth-split analysis uses the public paired classification together
with locally available all-wells pressure arrays and mesh-derived masks and
weights:

```bash
python code_package/scripts/analysis/analyze_well15_dominance_depth_split.py \
  --raw-root <LOCAL_OGS_RESULTS_RAW_DIR> \
  --paired-csv data_package/paired_pressure_differences.csv \
  --output-dir <OUTPUT_DIR>/well15_dominance_depth_split
```

The package also includes an optional post-mainshock pressure-increment
analysis. Its required model outputs are supplied through local paths.

Main Figures 1 and 2 and SI Figure S1 combine model results with external,
licensed, or manually assembled spatial context.
