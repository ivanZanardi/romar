# ROMAr

**Reduced order modeling for Argon plasma**

---

ROMAr is a Python library designed to implement model reduction techniques for collisional-radiative argon plasma [1], leveraging the CoBRAS method [2].

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
@misc{Zanardi_ROMAr_2025,
  title={Petrov-Galerkin model reduction for collisional-radiative argon plasma},
  author={Ivan Zanardi and Alessandro Meini and Alberto Padovan and Daniel J. Bodony and Marco Panesi},
  month={06},
  year={2025},
  eprint={2506.05483},
  archivePrefix={arXiv},
  primaryClass={physics.comp-ph},
  url={https://arxiv.org/abs/2506.05483},
  doi={10.48550/arXiv.2506.05483},
  author+an={1=highlight}
}
```

## Explore

Check out the [examples](https://github.com/ivanZanardi/romar/tree/main/examples) provided in the repository to see ROMAr in action.

## License

ROMAr is distributed under the [MIT License](https://github.com/ivanZanardi/romar/blob/main/LICENSE). You are welcome to utilize, modify, and contribute to this project in accordance with the terms outlined in the license.

## References

1. Kapper, M. G., Cambier, J.-L. (2011). *Ionizing shocks in argon. Part I: Collisional-radiative model and steady-state structure*. Journal of Applied Physics, **109**(11), 113308. https://doi.org/10.1063/1.3585688

2. Otto, S. E., Padovan, A., Rowley, C. W. (2023). *Model reduction for nonlinear systems by balanced truncation of state and gradient covariance*. SIAM Journal on Scientific Computing, **45**(5), A2325–A2355. https://doi.org/10.1137/22M1513228
