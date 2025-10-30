# ROMAr

## Examples

---

### Setup

Before running the scripts, **edit the `./inputs/paths.json` file** to reflect your local environment:

```json
{
  "paths": {
    "library": "/u/zanardi3/Codes/ML/ROMAr/romar/",
    "saving": "/u/zanardi3/Workspace/cr_argon/roms/run02/",
    "database": "/u/zanardi3/Codes/ML/ROMAr/romar/examples/database/",
    "mpl_style": "/u/zanardi3/scratch/Workspace/styles/matplotlib/paper_1column.mplstyle"
  }
}
```

**Field Descriptions**:
- `library`: Path to the ROMAr source code (should contain the `romar/` package).
- `saving`: Output folder where results and input files will be written.
- `database`: Path to the database of kinetic or thermodynamic reference data (e.g., `./database/`).
- `mpl_style`: Optional matplotlib `.mplstyle` file used for consistent plot styling (`null` to disable).

**Important**:  
Each `./inputs/<stage>/allrun.sh` script is the main driver for that stage of the pipeline. You only need to **modify the path to `paths.json` file and the Conda environment name** inside each `allrun.sh`. No other manual edits are needed.

---

### Running the Pipeline

You can either run the full pipeline with **SLURM** using `./inputs/jobscript.sh`, or execute each stage manually on a local machine.

**Option 1: Run with SLURM (HPC)**

```bash
cd ./inputs/
sbatch jobscript.sh
```

> This script runs both `./inputs/gen_data/allrun.sh` and `./inputs/max_mom_2/allrun.sh` sequentially on a SLURM cluster.

**Option 2: Run Locally (Laptop or Workstation)**

```bash
# Phase 1: Generate training and test data
cd gen_data
bash allrun.sh
cd ..

# Phase 2: Build and evaluate the ROM
cd max_mom_2
bash allrun.sh
cd ..
```

---

### Notes

- The **data generation phase** (`./inputs/gen_data/`) builds the datasets for training and testing.
- The **ROM generation phase** (`./inputs/max_mom_2/`) uses up to **second-order moments** as observables.
- To extend to higher-order moments (e.g. 3rd, 4th...), you can:
  - Copy `./inputs/max_mom_2/` to `./inputs/max_mom_3/`, `./inputs/max_mom_4/`, etc.
  - Adjust the `./inputs/max_mom_<x>/input_files/` JSON configs accordingly.
