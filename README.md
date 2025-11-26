# Galaxy Angular Correlation (CUDA)

CUDA kernels for the 2‑point angular correlation function, comparing real and random galaxy catalogs.

## Overview
- Computes DD, DR, and RR histograms of pair angles, then evaluates the Landy–Szalay estimator `w(θ) = (DD - 2·DR + RR) / RR`.
- Multiple CUDA implementations: shared-memory optimized, no-cache baseline, and a manual/experimental variant.
- Organized layout with Makefile builds, sample data, and output folders for reproducible runs.

## Repository Layout
- `src/galaxy_shared_mem.cu` — shared-memory kernel (90° range, 1024 threads/block).
- `src/galaxy_no_cache.cu` — global-memory baseline (180° range).
- `src/galaxy_manual.cu` — experimental/manual tuning.
- `src/template.cu` — scaffold for new kernels.
- `data/real.dat`, `data/synthetic.dat` — sample catalogs; first line = count, subsequent lines: `ra dec` in arcminutes.
- `bin/` — compiled binaries.
- `outputs/` — place run artifacts and plots.
- `docs/` — long-form notes, profiling reports, figures.

## Requirements
- NVIDIA CUDA toolkit (`nvcc` on PATH).
- NVIDIA GPU; set `ARCH` in the Makefile to your compute capability (e.g., `sm_86`).
- Bash-compatible shell on Linux/macOS (adjust commands for other platforms).

## Build
```bash
make                  # builds all targets into bin/
# override defaults if needed:
# make ARCH=sm_86 NVCCFLAGS="-O3 -lineinfo"
```
Targets produced: `galaxy_shared_mem`, `galaxy_no_cache`, `galaxy_manual`.

## Run
```bash
./bin/galaxy_shared_mem data/real.dat data/synthetic.dat outputs/shared.out
./bin/galaxy_no_cache   data/real.dat data/synthetic.dat outputs/no_cache.out
./bin/galaxy_manual     data/real.dat data/synthetic.dat outputs/manual.out
```
- Inputs are arcminutes. In code: `ra_rad = ra_arcmin * (π / 10800)`, `dec_rad = dec_arcmin * (π / 10800)`.
- Outputs are per-bin counts; post-process to derive `w(θ)` or plot histograms.

## Data Format
```
N
ra_0 dec_0
ra_1 dec_1
...
```
- First line `N` = number of entries.
- Right Ascension and Declination are arcminutes; conversion to radians occurs in the code.

## Implementation Notes
- Bin width is 0.25° (`binsperdegree = 4`). `totaldegrees` controls the angular extent (90° in shared-memory variant, 180° in baseline).
- `galaxy_shared_mem.cu`: tiles coordinates into shared memory, accumulates per-block histograms, then atomically merges.
- `galaxy_no_cache.cu`: simple global-memory approach using unified memory for histograms.
- `galaxy_manual.cu`: sandbox for manual tuning or alternative strategies.

## Profiling & Best Practices
- Use `nvprof`, `nsys profile`, or `ncu` to inspect occupancy, shared-memory usage, and atomic contention.
- Tune `threadsperblock` for your GPU; ensure `ARCH` matches the device.
- For larger catalogs, consider hierarchical binning, pair subsampling, or widening bins to reduce atomic pressure.
- Keep outputs and plots under `outputs/`; store profiling notes in `docs/`.

## Contributing
- Add new kernels in `src/` and hook them into the Makefile pattern rule.
- Favor deterministic reductions when comparing variants.
- Submit changes under MIT License (see `LICENSE`).

## License
MIT License. See `LICENSE` for details.
