# Peace River GRL Software

This package contains the analysis and visualization software associated with
the study “Hydraulic uncertainty complicates estimating how much injection
wells individually contribute to fault pressure loading near Peace River,
Alberta.” It accompanies the separate Peace River Pressure-Dominance Data
package.

## Package contents

- `scripts/mesh/`: model-mesh preparation and rescaling.
- `scripts/ogs_input/`: hydraulic ensemble definition and OGS project-file
  preparation.
- `scripts/postprocess/`: pressure-series merging, resampling, and export.
- `scripts/analysis/`: pressure-contribution, paired-pressure, depth-split, and
  optional post-mainshock pressure-increment analyses.
- `scripts/figures/`: Main Figure 3 and Supporting Information figure scripts.
- `scripts/paired_pressure_data.py`: validation and transformation helpers for
  the public paired-pressure and parameter-design tables.
- `environment/`: Conda environment specifications.
- `config/`: an example local-path configuration.
- `workflows/`: commands and input requirements for the main workflows.

## Data inputs

The commands in `workflows/WORKFLOW_NOTE.md` assume that the software and data
packages are sibling directories:

```text
<RELEASE_ROOT>/
  code_package/
  data_package/
```

The data package provides these flat-file inputs:

- `data_package/params_design_reference.csv`: the 1,000-realization hydraulic
  parameter design.
- `data_package/paired_pressure_differences.csv`: paired pressure differences
  for 1,000 realizations and seven source groups.
- `data_package/southern_cluster_monthly_counts.csv`: monthly southern-cluster
  earthquake counts used in Figure 1.

Scripts that require model outputs, spatial overlays, or licensed injection
data accept local paths through command-line options or a local configuration
file. Their input requirements are described in the workflow guide.

## Main workflows

- **Hydraulic ensemble definition:** `scripts/ogs_input/generate_params.py`
  constructs the 1,000-case parameter design used for the model ensemble.
- **SI Figure S3:**
  `scripts/figures/plot_well15_paired_pressure_parameter_grid.py` uses
  `paired_pressure_differences.csv` and `params_design_reference.csv`.
- **SI Figure S5:** `scripts/figures/plot_paired_difference_ecdf.py` calculates
  the seven empirical cumulative distributions directly from
  `paired_pressure_differences.csv`.
- **Main Figure 3:**
  `scripts/figures/plot_well15_paired_sign_pressure_maps.py` uses
  `paired_pressure_differences.csv` for paired classification and the Well 10
  and Well 15 ECDF panels. Generating the spatial fields additionally requires
  local model pressure arrays and spatial overlay data.
- **Other Supporting Information workflows:** SI Figure S2 requires licensed
  injection data, and SI Figure S4 requires the pressure-contribution summary
  produced from local processed model outputs.

`scripts/analysis/analyze_post_mainshock_pressure_increment.py` provides an
optional post-mainshock pressure-increment analysis. The Well 10 and Well 15
ECDF panels use helpers in
`scripts/figures/plot_well10_well15_dominance_ecdf.py`.

## Environment

The `environment/` directory provides Conda specifications for the OGS and
mesh workflows. Copy `config/local_paths.example.yml` to a local path and set
only the entries needed for the workflow being run. Detailed commands are in
`workflows/WORKFLOW_NOTE.md`.

## Citation and license

Citation guidance is provided in `CITATION_NOTE.md`. Author-generated software
and documentation are provided under the Apache License 2.0; see `LICENSE` and
`LICENSE_NOTE.md`.
