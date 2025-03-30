# ROMAr

**Reduced order modeling for Argon plasma**

---

ROMAr is a Python library designed to implement model reduction techniques for collisional-radiative Argon plasma, leveraging the CoBRAS method.

#### References:

- Physical model:

```bibtex
@article{Kapper_2011_Argon,
    author = {Kapper, M. G. and Cambier, J.-L.},
    title = {Ionizing shocks in argon. Part I: Collisional-radiative model and steady-state structure},
    journal = {Journal of Applied Physics},
    volume = {109},
    number = {11},
    pages = {113308},
    year = {2011},
    month = {06},
    issn = {0021-8979},
    doi = {10.1063/1.3585688}
}
```

- Model reduction:

```bibtex
@article{Otto_2023_CoBRAS,
  author = {Otto, Samuel E. and Padovan, Alberto and Rowley, Clarence W.},
  title = {Model Reduction for Nonlinear Systems by Balanced Truncation of State and Gradient Covariance},
  journal = {SIAM Journal on Scientific Computing},
  volume = {45},
  number = {5},
  pages = {A2325-A2355},
  year = {2023},
  doi = {10.1137/22M1513228}
}
```

## Installation

To install ROMAr, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/ivanZanardi/romar.git
cd romar
```

2. Create a Conda environment:

```bash
conda env create -f conda/env.yml
conda activate romar
```
> **Note:** If you encounter issues installing `pyharm`, download it manually from its [GitHub repository](https://github.com/ivanZanardi/pyharm.git) and install it locally.

1. Install the package:

```bash
pip install romar
```

4. Activate the Conda environment whenever using ROMAr:

```bash
conda activate romar
```

## Citation

If you use this code or find this work useful in your research, please cite us:

```bibtex
@misc{Zanardi_2024_ROMAr,
  title={Petrov-Galerkin model reduction for thermochemical nonequilibrium gas mixtures}, 
  author={Ivan Zanardi and Alberto Padovan and Daniel J. Bodony and Marco Panesi},
  year={2025},
  eprint={2411.01673},
  archivePrefix={arXiv},
  primaryClass={physics.comp-ph},
  url={https://arxiv.org/abs/2411.01673}, 
}
```

## Explore

Check out the [examples](https://github.com/ivanZanardi/romar/tree/main/examples) provided in the repository to see ROMAr in action.

## License

ROMAr is distributed under the [Apache-2.0 license](https://github.com/ivanZanardi/romar/blob/main/LICENSE). You are welcome to utilize, modify, and contribute to this project in accordance with the terms outlined in the license.
