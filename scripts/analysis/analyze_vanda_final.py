from pathlib import Path
import csv
import re
import math
import statistics

ROOT = Path("results")
OUT = ROOT / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

records = []
warnings = []

def parse_file(path, experiment, implementation, cores, expected_n):
    text = path.read_text()

    def field(pattern):
        match = re.search(pattern, text)
        if not match:
            raise ValueError("Missing field: " + pattern)
        return match.group(1)

    n = int(field(r"N=(\d+)"))
    d = int(field(r"D=(\d+)"))
    k = int(field(r"K=(\d+)"))
    seed = int(field(r"Seed=(\d+)"))
    ranks = int(field(r"MPI ranks:\s*(\d+)"))
    iterations = int(field(r"Iterations:\s*(\d+)"))
    inertia = float(field(r"Inertia:\s*([0-9.eE+-]+)"))
    runtime = float(field(r"Runtime \(seconds\):\s*([0-9.eE+-]+)"))

    if (n, d, k, seed) != (expected_n, 10, 5, 42):
        raise ValueError("Unexpected benchmark parameters")

    if not math.isfinite(runtime) or runtime <= 0:
        raise ValueError("Invalid runtime")

    if not math.isfinite(inertia):
        raise ValueError("Invalid inertia")

    threads = 1
    if implementation == "hybrid":
        threads = int(field(
            r"Actual OpenMP threads per rank:\s*(\d+)"
        ))
        if threads != 3:
            raise ValueError("Unexpected OpenMP thread count")

    if cores is not None:
        expected_ranks = cores if implementation == "mpi" else cores // 3
        if ranks != expected_ranks:
            raise ValueError("Unexpected MPI rank count")

    centroid_rows = re.findall(
        r"^Centroid\s+(\d+):\s*(.+)$",
        text,
        re.MULTILINE
    )

    if len(centroid_rows) != k:
        raise ValueError("Incomplete centroids")

    for idx, values in centroid_rows:
        coords = [float(x) for x in values.split()]
        if len(coords) != d or not all(map(math.isfinite, coords)):
            raise ValueError("Invalid centroid coordinates")

    repeat = int(re.search(r"_(\d+)\.txt$", path.name).group(1))

    records.append({
        "experiment": experiment,
        "implementation": implementation,
        "cores": cores if cores is not None else "",
        "N": n,
        "D": d,
        "K": k,
        "seed": seed,
        "mpi_ranks": ranks,
        "threads_per_rank": threads,
        "repeat": repeat,
        "iterations": iterations,
        "inertia": inertia,
        "runtime_seconds": runtime,
        "source_file": str(path)
    })

def load(path, experiment, implementation, cores, n):
    if not path.is_dir():
        warnings.append("Missing directory: " + str(path))
        return

    for i in range(1, 6):
        file = path / f"run_{i}.txt"

        if not file.exists():
            warnings.append("Missing file: " + str(file))
            continue

        try:
            parse_file(file, experiment, implementation, cores, n)
        except Exception as e:
            warnings.append(f"{file}: {e}")

# CPU strong and weak scaling
for cores in (36, 72, 144, 288):
    for version in ("mpi", "hybrid"):

        if cores == 72 and version == "hybrid":
            strong_dir = ROOT / "vanda_strong_72cpu_scatter2" / version
        elif cores in (36, 72):

            strong_dir = ROOT / f"vanda_strong_{cores}cpu_auto" / version
        else:
            strong_dir = ROOT / f"vanda_strong_{cores}cpu_scatter" / version

        load(strong_dir, "strong", version, cores, 2000000)

        weak_dir = ROOT / f"vanda_weak_{cores}cpu" / version
        load(weak_dir, "weak", version, cores, cores * 10000)

# GPU benchmark
gpu_dir = ROOT / "vanda_gpu_benchmark_auto"

for n in (100000, 500000, 1000000, 2000000):
    for version in ("cpu", "gpu"):
        for i in range(1, 6):
            file = gpu_dir / f"N{n}_{version}_{i}.txt"

            if not file.exists():
                warnings.append("Missing file: " + str(file))
                continue

            try:
                parse_file(file, "gpu_benchmark", version, None, n)
            except Exception as e:
                warnings.append(f"{file}: {e}")

# Save raw records
raw_path = OUT / "vanda_all_runs.csv"

with raw_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "experiment", "implementation", "cores",
        "N", "D", "K", "seed",
        "mpi_ranks", "threads_per_rank",
        "repeat", "iterations", "inertia",
        "runtime_seconds", "source_file"
    ])
    writer.writeheader()
    writer.writerows(records)

# Aggregate by experiment, implementation, cores and N
groups = {}

for row in records:
    key = (
        row["experiment"],
        row["implementation"],
        row["cores"],
        row["N"]
    )
    groups.setdefault(key, []).append(row)

summary = []

for key, rows in sorted(groups.items(), key=lambda x: str(x[0])):
    times = [r["runtime_seconds"] for r in rows]

    summary.append({
        "experiment": key[0],
        "implementation": key[1],
        "cores": key[2],
        "N": key[3],
        "valid_runs": len(times),
        "median_seconds": statistics.median(times),
        "mean_seconds": statistics.mean(times),
        "std_seconds": statistics.stdev(times) if len(times) > 1 else 0,
        "min_seconds": min(times),
        "max_seconds": max(times),
        "iterations": ",".join(
            str(r["iterations"]) for r in rows
        )
    })

summary_path = OUT / "vanda_summary.csv"

with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "experiment", "implementation", "cores",
        "N", "valid_runs", "median_seconds",
        "mean_seconds", "std_seconds",
        "min_seconds", "max_seconds", "iterations"
    ])
    writer.writeheader()
    writer.writerows(summary)

print("=== VANDA FINAL DATA ANALYSIS ===")
print("Raw records:", len(records))
print("Summary groups:", len(summary))
print("Warnings:", len(warnings))

for row in summary:
    print(
        row["experiment"],
        row["implementation"],
        "cores=" + str(row["cores"]),
        "N=" + str(row["N"]),
        "runs=" + str(row["valid_runs"]),
        "median=" + f'{row["median_seconds"]:.6f}s'
    )

print("\n=== WARNINGS ===")
for warning in warnings:
    print(warning)

print("\nSaved:", raw_path)
print("Saved:", summary_path)
