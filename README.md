# RONEK

**Reduced Order modeling for Non-Equilibrium Kinetics**

---

RONEK is a Python library designed to perform model reduction for non-equilibrium kinetics, based on balanced truncation techniques. It currently supports mixtures consisting of two species: one atom and one molecule, with a single chemical component. For instance, it can manage mixtures like $\mathcal{S}=\left[\text{N},\text{N}2\right]$ or $\mathcal{S}=\left[\text{O},\text{O}2\right]$.

It leverages PyTorch for GPU acceleration. This implementation excels in performance, making it well-suited for high-dimensional problems.

## Installation

To install RONEK, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/ivanZanardi/ronek.git
cd ronek
```

2. Create a Conda environment:

```bash
conda env create -f conda/env.yml
conda activate ronek
```

3. Install the package:

```bash
pip install ronek
```

4. Activate the Conda environment whenever using RONEK:

```bash
conda activate ronek
```

## Citation

If you use this code or find this work useful in your research, please cite us:

```bibtex
@article{Zanardi_Ronek_2024,
  author    = {Zanardi, Ivan and Padovan, Alberto and Bodony, Daniel Joseph and Panesi, Marco},
  title     = {Petrov-Galerkin model reduction for thermochemical nonequilibrium gas mixtures},
  year      = {2024}
}
```

## Explore

Check out the [examples](https://github.com/ivanZanardi/ronek/tree/main/examples) provided in the repository to see RONEK in action.

## License

RONEK is distributed under the [Apache-2.0 license](https://github.com/ivanZanardi/ronek/blob/main/LICENSE). You are welcome to utilize, modify, and contribute to this project in accordance with the terms outlined in the license.
