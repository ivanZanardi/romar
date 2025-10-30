# ⚙️ Reduced-Order Model (ROM) Pipeline

This project automates the full workflow for generating and evaluating reduced-order models for thermochemical nonequilibrium systems, using the [ROMAr library](https://github.com/ivanZanardi/romar).

---

## 🛠️ `paths.json` Configuration

Edit the `paths.json` file before running the scripts:

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

- `library`: Path to the `romar` Python source code
- `saving`: Where generated data and ROM results are stored
- `database`: Location of kinetic or thermodynamic input data
- `mpl_style`: Matplotlib style file used for plotting (optional)

---

## 🚀 Run Instructions

### ▶️ Local Execution (Laptop or Workstation)

```bash
# Phase 1: Generate training and test data
cd gen_data
bash allrun.sh
cd ..

# Phase 2: Build and evaluate ROM
cd max_mom_2
bash allrun.sh
cd ..
```

Each `allrun.sh` will:
- Activate the correct `conda` environment (must be named `sciml`)
- Extract script/data paths from `../paths.json`
- Run the necessary Python scripts in order
- Clean up temporary input folders after execution

---

### 🧠 SLURM Execution (HPC)

Submit the provided job script from the root folder:

```bash
sbatch jobscript.sh
```

**`jobscript.sh`**:
- Runs both `gen_data/allrun.sh` and `max_mom_2/allrun.sh`
- Logs output and errors per SLURM job ID

Example SLURM header (in `jobscript.sh`):

```bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --partition=chess
#SBATCH --job-name=roms_run01
#SBATCH --output=log.output_%j
#SBATCH --error=log.error_%j
```

---

## 📘 Notes

- The **data generation phase** (`gen_data/`) builds all training/test datasets.
- The **ROM building phase** (`max_mom_2/`) assumes **second-order moments** as observables.
- You can easily extend the structure to higher moments by copying `max_mom_2/` into folders like `max_mom_3/`, etc., and modifying the JSON input files.
