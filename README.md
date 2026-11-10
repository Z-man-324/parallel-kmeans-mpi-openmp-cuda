# DSA5208 Project 3 — Parallel K-means (MPI + OpenMP + CUDA)

## Contents and provenance
This package contains the C++/CUDA source, original PBS scripts from Atlas and Vanda, analysis scripts, summarized CSVs and selected original benchmark text outputs. PBS scripts are preserved as run on the respective clusters. Experimental results are measurements, not simulation. Debugging variants, PBS stdout/stderr logs, executable binaries, object files, SSH keys and passwords are excluded.

## Algorithm and command-line arguments
Implementations: serial K-means (`serial_general.cpp`), distributed MPI (`mpi_kmeans.cpp`), hybrid MPI/OpenMP (`hybrid_kmeans.cpp`), allocation-reuse hybrid (`hybrid_kmeans_optimized.cpp`), and MPI/CUDA (`mpi_cuda_kmeans.cpp` + `cuda_assignment.cu`). Shared deterministic synthetic data generation: `benchmark_data.hpp`; shared serial logic: `kmeans_common.hpp`.

Executables take positional arguments `N D K seed [mode]`. Reported experiments use `D=10`, `K=5`, `seed=42`, `mode=overlap`. Output reports iterations, inertia, runtime and centroids. MPI+OpenMP experiments use three threads per rank, one rank per three allocated CPU cores. GPU offload assigns points on the GPU, but cluster accumulation and centroid updates remain on the CPU.

## Building on Vanda
From the package root on a Vanda login node (compile only; submit computational workloads to PBS):

```bash
bash scripts/build_vanda.sh
bash scripts/stage_executables.sh
```

The original Vanda build script uses `module load gompi/2023a` and `module load CUDA/12.6.0`, GCC 12.3 and OpenMPI 4.1.5. `build_vanda.sh` creates `bin/` executables. Historical PBS scripts invoke `./tests/<program>`; `stage_executables.sh` copies the newly built executables into `tests/` to match those paths. The build script was previously run successfully on Vanda; the staging helper is a packaging convenience and has not been independently executed on the cluster. If only CPU programs are needed, omit the CUDA-specific build steps or compile them separately.

For Atlas, the historical environment uses the `serial` and `parallel` queues, and some PBS scripts load `openmpi/4.1.1` with Atlas-specific library paths. Do not use Vanda module names on Atlas. Build with the cluster's compatible `g++`/`mpicxx` and ensure `tests/` contains the matching executables before submitting historical PBS scripts. The Atlas executable environment was used for the original experiments; the Vanda build helper is not claimed to have been tested on Atlas.

## Running PBS experiments
Run commands from the project root after placing the package on the intended cluster and staging its executables. The PBS scripts assume `cd "$PBS_O_WORKDIR"`; do not run them directly as shell scripts.

```bash
qsub scripts/atlas/serial_size_scaling_final.pbs
qsub scripts/atlas/problem_size_scaling.pbs
qsub scripts/vanda/vanda_288_mpi_scatter.pbs
qsub scripts/vanda/vanda_288_hybrid_scatter.pbs
qsub scripts/vanda/vanda_weak_288_mpi.pbs
qsub scripts/vanda/vanda_weak_288_hybrid.pbs
qsub scripts/vanda/vanda_gpu_benchmark_auto.pbs
```

**Submit Atlas scripts only on Atlas and Vanda scripts only on Vanda.** Resource availability, partitions, PBS module versions, and queues may change. All script paths above are preserved historical configurations; they are not guarantees of future queue availability.

### Experimental design
- Atlas serial problem-size scaling: N = 100,000; 500,000; 1,000,000; 2,000,000. Five runs each; the N=1,000,000 series comes from an earlier serial baseline job.
- Atlas MPI/Hybrid: 12, 24, 48 CPU cores; strong and weak scaling; 12-core problem-size scaling.
- Vanda strong scaling: N = 2,000,000, 36/72/144/288 CPU cores, five repeats per implementation and configuration.
- Vanda weak scaling: N = 10,000 times CPU core count, same core counts and repeats.
- Vanda GPU: NVIDIA A40, one MPI rank, paired CPU vs GPU for N = 100,000; 500,000; 1,000,000; 2,000,000, five repeats each.

## Results and statistical interpretation
- `results/atlas/analysis/` and `results/vanda/analysis/` contain the exported summaries; `results/*/raw/` retain original program output files for audit.
- Vanda strong scaling speedup is **normalized to the 36-core runtime of the same implementation**, not to one-core serial runtime. Efficiency = relative speedup / (cores/36).
- Weak scaling normalizes runtime against the 36-core baseline. Iteration counts vary across problem sizes; apparent efficiencies above 100% must not be interpreted as proven superlinear hardware speedups.
- Reported runtime excludes initial data generation and initial MPI data distribution in the MPI implementations. GPU time includes repeated device allocation and host/device copies within iterations; this is not a pure CUDA-kernel timing.
- Five-run medians are used for headline comparisons. Outlying first runs at 288 cores remain in the raw data, and must not be silently discarded.

## Analysis scripts
`analyze_vanda_final.py`, `calculate_performance.py`, and `create_log_figures.py` are copies of the original scripts. They expect a project-root `results/vanda_*` directory layout; this archive organizes raw outputs under `results/vanda/raw/`. To re-run the original analysis unchanged, reconstruct its original directory structure (copy the `results/vanda/raw/vanda_*` directories into root `results/`) and run the scripts from the project root. Similarly, the Atlas analysis scripts expect the original `results/` structure. Existing CSVs are provided so that the published figures can be checked without rerunning analysis.

## Validation and reproducibility limitations
The four 288-core Vanda jobs exited with PBS status zero; all 10 paired strong/weak MPI–Hybrid runs matched iteration counts, inertia and centroids (maximum centroid difference 0.0). GPU benchmark validation reported 20/20 paired comparisons passing. Original output files are included. A zero PBS exit code alone does not prove numerical correctness.

This submission does not include profiler traces or a measured communication-only breakdown; proposed explanations for timing anomalies are hypotheses. PBS job scheduling and hardware allocation may vary on future reruns. Source files were checked for consistency across the two archived clusters; package scripts were syntax-checked offline, but the assembled submission package was not rerun end-to-end on either cluster.

## Academic integrity
The PDF report contains the AI-tool use declaration. The student is responsible for verifying all reported results and following the university's submission requirements.
