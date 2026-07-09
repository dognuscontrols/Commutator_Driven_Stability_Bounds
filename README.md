# Three-Mode Sweep

[![arXiv](https://img.shields.io/badge/arXiv-2607.05829-b31b1b.svg)](https://arxiv.org/abs/2607.05829)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FLCSYS.2026.3708744-blue.svg)](https://doi.org/10.1109/LCSYS.2026.3708744)
[![Julia](https://img.shields.io/badge/Julia-1.9%2B-9558B2.svg)](https://julialang.org)

Supporting numerics for:

> D. Mallik and N. Chopra, "Commutator-Driven Stability Bounds for Periodic Switching," *IEEE Control Systems Letters*, 2026. DOI: [10.1109/LCSYS.2026.3708744](https://doi.org/10.1109/LCSYS.2026.3708744). Preprint: [arXiv:2607.05829](https://arxiv.org/abs/2607.05829).

This script implements the sampled certificate computation (Algorithm 1) and reproduces the three-mode example of Section IV. The mode matrices are fixed to the paper's

```
A1 = [-1 4 0; 0 -1 0; 0 0 -1]
A2 = [-1 0 0; -4 -1 0; 0 0 -1]
A3(r) = [-1 1.6 0; 0 -1 r; 0 0 -1]
```

and only the coupling `r` is swept over `0.0:0.2:2.0`. This activates the single Euclidean commutator edge `||[A3(r), A1]||_2 = 4r` while `||[A2, A1]||_2 = 16` and `||[A3, A2]||_2 = 6.4` stay fixed. For each `r`, the script:

- estimates a sampled Floquet dwell-time threshold via bisection on the monodromy spectral radius,
- synthesizes a simplex-affine Lyapunov metric as a semidefinite feasibility problem via JuMP + COSMO, maximizing the decay rate `eta` by bisection,
- verifies the certificate a posteriori by eigenvalue checks on the sampled grid,
- computes the sampled certified switching period `tau*_G` (via the prefix condition `kappa*_G(s) < 1` for all `s` up to `tau`) and the conservatism ratio against the sampled Floquet threshold.

Certificate acceptance deliberately requires `MOI.OPTIMAL` from the solver — stricter than accepting any feasible point — and every accepted certificate is additionally re-verified by direct eigenvalue checks.

The constant common-Lyapunov metric discussed in the paper (which fails at every grid point for every `r`) is implemented but disabled by default; see [Options](#options) to enable it.

## Requirements

- Julia 1.9+
- [JuMP](https://github.com/jump-dev/JuMP.jl)
- [COSMO](https://github.com/oxfordcontrol/COSMO.jl)

## Setup

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Usage

Run the full sweep from the command line:

```bash
julia --project=. three_mode_sweep.jl
```

Or from the REPL with custom options:

```julia
include("three_mode_sweep.jl")

result = run_three_mode_sweep(opts = Options(gridN = 31, r_values = collect(0.0:0.2:2.0)))
```

## Outputs

A summary is printed to stdout. Unless `write_csv = false`, the script also writes two CSV files to Julia's current working directory (`pwd()`) — no separate export step is needed — and logs the absolute path of each after writing:

- `three_mode_sweep_summary.csv` — full per-`r` results (edge norms, `eta*`, `tau*`, feasible grid counts, Floquet diagnostics),
- `three_mode_table1.csv` — the rows of Table I in the paper (`r = 0.0:0.4:2.0`, columns `||[A3, A1]||_2`, `eta*`, `tau*`, and the conservatism ratio).

Both filenames are gitignored: they're disposable, regenerated on every run, and never committed.

The one committed reference file is **`results/sweep_reference.csv`** — the exact `three_mode_sweep_summary.csv` output from the run underlying the paper's numbers. To check reproduction, run the script and diff a fresh `three_mode_sweep_summary.csv` against it (small last-digit differences may occur across COSMO versions, since COSMO is a first-order solver). Table I corresponds to the rows `r = 0.0, 0.4, 0.8, 1.2, 1.6, 2.0` and the columns `edge13_unscaled`, `eta_star`, `tau_cert`, and `conservatism_ratio`.

## Options

All parameters live in the `Options` struct (see its docstring), including grid resolution (`gridN`, `alpha_min`), the sweep and table rows (`r_values`, `table_r_values`), SDP conditioning (`epsP`), bisection tolerances and caps for `eta`, `tau`, and the Floquet threshold, output file names, and metric toggles (`run_constant_metric`, `run_affine_metric`).

## License

This code is released under the MIT License. See [LICENSE](LICENSE).
