# Sequential cognitive states: code

This repository holds the code that produces every computed figure and number of
the paper

> M. Virginia Bolelli, Luca Greco, Dario Prandi, *Modeling sequential cognitive
> states via population-level cortical dynamics*.

It covers the heteroclinic cycle of Figure 2.1(b), the trained neural fields of
Section 5 and the cyclic-inhibition example of Figure S1.1 of the Supplementary
Material.

## Requirements

Python 3 with the packages of `requirements.txt` (the pinned versions are the
ones used for the paper), plus a TeX distribution with `latexmk` and `pgfplots`
for the two figures drawn with pgfplots.

```sh
pip install -r requirements.txt
```

## Usage

Run everything from the repository root.

```sh
make all        # figures/ and numbers.tex from the committed results/
make verify     # rerun the checks of Section 5 on the committed weights
make train      # retrain both networks of Section 5, three minutes each
make data       # recompute the pgfplots tables, about fifteen minutes
make replot     # redraw every figure from results/
make install PAPER=<path>   # copy figures and numbers.tex into <path>
```

Pass another interpreter with `make PYTHON=<command>`. Every figure script
separates computation from plotting: it caches its results in `results/` and
accepts `--replot` to redraw the figure from that cache in a few seconds.

| Paper item               | Command                 | Output                               |
| ------------------------ | ----------------------- | ------------------------------------ |
| Figure 2.1(b)            | `heteroclinic_cycle.py` | `figures/heteroclinic-cycle.pdf`     |
| Figure 5.1(a)            | `train_lv.py`           | `figures/comparison_symmetric.png`   |
| Figure 5.1(b)            | `train_lv.py`           | `figures/comparison_asymmetric.png`  |
| Figure 5.2(a)            | `train_lv.py`           | `figures/populations_symmetric.png`  |
| Figure 5.2(b)            | `train_lv.py`           | `figures/populations_asymmetric.png` |
| Section 5, periods       | `train_lv.py`           | `results/lv-training.json`           |
| Section 5, checks        | `verify_lv.py`          | `results/lv-verify.json`             |
| Figure S1.1              | `ring_cycle.py`         | `figures/ring-cycle.pdf`             |
| Section S1, threshold c* | `ring_cycle.py`         | `results/ring-results.json`          |
| Every number quoted      | `make_numbers.py`       | `numbers.tex`                        |

Each script runs as `python3 <script>` from the repository root. "Section 5,
periods" covers the periods, the return errors, the projection errors and the
displayed windows. "Section 5, checks" covers everything that `verify_lv.py`
computes, listed below.

## Files

- `lvnet.py`: the Lotka-Volterra target, the block neural field, the training
  loop, the integrator and the period estimate of Section 5.
- `train_lv.py`: trains both networks with a fixed seed, integrates them and
  draws Figures 5.1 and 5.2. The weights, training histories and trajectories
  are cached in `results/lv-training-*.npz`.
- `verify_lv.py`: from the cached weights, the field and Jacobian errors on K
  and near the heteroclinic cycle, the equilibria near the saddles, the sign of
  the normal components on the coordinate planes, a sweep of 30 initial
  conditions and the Floquet multipliers of the periodic orbit.
- `ring_cycle.py`, `heteroclinic_cycle.py`: the tables of the two pgfplots
  figures, whose sources are in `tex/`.
- `make_numbers.py`: writes `numbers.tex`, one LaTeX macro per number, from the
  JSON files in `results/`. It computes nothing itself.

## Reproducibility

The figures and numbers come from the committed weights in `results/`. Training
is deterministic given the seed. On another machine, or with other versions of
NumPy or of the BLAS library, retraining gives networks that are close to the
committed ones but not bit-identical. Since the periodic orbit depends on the
trained field, derived quantities can then move noticeably: between two machines
we observed periods differing in the second digit. `make verify` and `make all`
never retrain.

## License

MIT, see `LICENSE`.
