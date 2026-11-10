from pathlib import Path
import re
import csv
import statistics

root = Path("results")
records = []
errors = []

def parse_result(path, experiment, cores, n, version, repeat):
    text = path.read_text()

    def extract(pattern):
        match = re.search(pattern, text)
        if not match:
            raise ValueError("Missing field: " + pattern)
        return match.group(1)

    runtime = float(extract(r"Runtime \(seconds\):\s*([0-9.eE+-]+)"))
    iterations = int(extract(r"Iterations:\s*(\d+)"))
    inertia = float(extract(r"Inertia:\s*([0-9.eE+-]+)"))

    actual_n = int(extract(r"N=(\d+)"))
    actual_d = int(extract(r"D=(\d+)"))
    actual_k = int(extract(r"K=(\d+)"))
    actual_seed = int(extract(r"Seed=(\d+)"))
    ranks = int(extract(r"MPI ranks:\s*(\d+)"))

    if actual_n != n or actual_d != 10 or actual_k != 5 or actual_seed != 42:
        raise ValueError("Unexpected experimental parameters")

    if runtime <= 0 or iterations <= 0:
        raise ValueError("Invalid runtime or iterations")

    expected_ranks = cores if version == "MPI-only" else cores // 3
    if ranks != expected_ranks:
        raise ValueError("Unexpected MPI rank count")

    records.append({
        "experiment": experiment,
        "cores": cores,
        "N": n,
        "version": version,
        "repeat": repeat,
        "runtime": runtime,
        "iterations": iterations,
        "inertia": inertia
    })

for n in [100000, 500000, 1000000, 2000000]:
    for version, tag in [("MPI-only", "mpi"), ("Hybrid", "hybrid")]:
        for repeat in range(1, 6):
            path = root / "problem_size_scaling" / f"N{n}_{tag}_{repeat}.txt"
            try:
                parse_result(path, "Problem Size", 12, n, version, repeat)
            except (OSError, ValueError) as e:
                errors.append(f"{path}: {e}")

for cores, n in [(12, 120000), (48, 480000)]:
    for version, tag in [("MPI-only", "mpi"), ("Hybrid", "hybrid")]:
        for repeat in range(1, 6):
            path = root / f"weak_scaling_{cores}cpu" / f"{tag}_{repeat}.txt"
            try:
                parse_result(path, "Weak Scaling", cores, n, version, repeat)
            except (OSError, ValueError) as e:
                errors.append(f"{path}: {e}")

outdir = root / "analysis"
outdir.mkdir(parents=True, exist_ok=True)

with (outdir / "new_scaling_raw.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "experiment", "cores", "N", "version", "repeat",
        "runtime", "iterations", "inertia"
    ])
    writer.writeheader()
    writer.writerows(records)

groups = {}
for r in records:
    key = (r["experiment"], r["cores"], r["N"], r["version"])
    groups.setdefault(key, []).append(r)

summary = []

for key, rows in sorted(groups.items()):
    times = [r["runtime"] for r in rows]
    iterations = [r["iterations"] for r in rows]
    inertias = [r["inertia"] for r in rows]

    summary.append({
        "experiment": key[0],
        "cores": key[1],
        "N": key[2],
        "version": key[3],
        "repeats": len(rows),
        "mean_runtime": statistics.mean(times),
        "median_runtime": statistics.median(times),
        "std_runtime": statistics.stdev(times) if len(times) > 1 else 0,
        "min_runtime": min(times),
        "max_runtime": max(times),
        "iterations_min": min(iterations),
        "iterations_max": max(iterations),
        "inertia_min": min(inertias),
        "inertia_max": max(inertias)
    })

with (outdir / "new_scaling_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)

print("=== NEW SCALING ANALYSIS ===")
print("Valid records:", len(records))
print("Errors:", len(errors))

for error in errors:
    print("ERROR:", error)

for row in summary:
    print(
        f"{row['experiment']:13s} "
        f"{row['version']:8s} "
        f"Cores={row['cores']:2d} "
        f"N={row['N']:7d} "
        f"Median={row['median_runtime']:.6f}s "
        f"Std={row['std_runtime']:.6f}s "
        f"Iterations={row['iterations_min']}-{row['iterations_max']}"
    )

print("Saved:", outdir / "new_scaling_raw.csv")
print("Saved:", outdir / "new_scaling_summary.csv")
