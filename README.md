# Galaxy Angular Correlation (CUDA)

CUDA implementations for the 2‑point angular correlation function comparing real and random galaxy catalogs.

## Repository Layout
- `src/galaxy_shared_mem.cu` – shared-memory kernel (90° range, 1024 threads/block).
- `src/galaxy_no_cache.cu` – global-memory baseline (180° range).
- `src/galaxy_manual.cu` – manual/experimental variant.
- `src/template.cu` – scaffold for new kernels.
- `data/real.dat`, `data/synthetic.dat` – sample catalogs (first line count, following lines `ra dec` in arcminutes).
- `bin/` – build outputs.
- `outputs/` – place run artifacts.
- `docs/` – add figures/notes here.

## Requirements
- CUDA toolkit (`nvcc` on PATH).
- NVIDIA GPU with compute capability matching your `ARCH` flag.
- Linux/macOS; adjust flags if building on other platforms.

## Build
```
make           # builds all targets into bin/
# optional overrides:
# make NVCCFLAGS="-O3 -lineinfo" ARCH=sm_86
```

Targets: `galaxy_shared_mem`, `galaxy_no_cache`, `galaxy_manual`.

## Run
```
./bin/galaxy_shared_mem data/real.dat data/synthetic.dat outputs/shared.out
./bin/galaxy_no_cache   data/real.dat data/synthetic.dat outputs/no_cache.out
./bin/galaxy_manual     data/real.dat data/synthetic.dat outputs/manual.out
```

Input is assumed in arcminutes. Inside the code, values are converted to radians:
```
ra_rad   = ra_arcmin   * (π / 10800)
dec_rad  = dec_arcmin  * (π / 10800)
```

## Science Quick Reference
- Angle between two positions:
  ```
  θ = arccos( sin(d1)·sin(d2) + cos(d1)·cos(d2)·cos(a1 - a2) )
  ```
- Correlation estimator per bin `i`:
  ```
  w_i(θ) = (DD_i - 2·DR_i + RR_i) / RR_i
  ```
  Values near 0 ⇒ random-like; significant deviation ⇒ clustered.

## Implementation Notes
- `galaxy_shared_mem.cu`: loads tiles of coordinates into shared memory and accumulates per-block histograms before atomically combining into global bins.
- `galaxy_no_cache.cu`: straightforward global-memory access using unified memory for histograms.
- `galaxy_manual.cu`: experimental variant for manual tuning.
- Bin width is 0.25° (`binsperdegree = 4`). The shared-memory version currently caps at 90°; adjust `totaldegrees` if you need the full 180° range.

## Profiling & Best Practices
- Use `nvprof`, `nsys profile`, or `ncu` to measure occupancy, shared-memory pressure, and atomic throughput.
- Tune `threadsperblock` to your GPU; ensure `ARCH` in the Makefile matches the device (e.g., `sm_86` for A100/GA100).
- Keep input files in pinned memory if you experiment with host-side preprocessing.
- If you expand to >100k galaxies, consider hierarchical binning or pair subsampling to keep histogram contention manageable.

## Contributing
- Add new kernels under `src/` and hook them into the `Makefile` pattern rule.
- Prefer deterministic output (avoid non-deterministic atomics when comparing variants).
- Store plots/results in `outputs/` and long-form notes in `docs/`.
