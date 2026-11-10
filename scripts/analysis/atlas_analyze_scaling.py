from pathlib import Path
import csv
import math
import re
import statistics

ROOT = Path("results")
OUT = ROOT / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_ITERATIONS = 26
EXPECTED_INERTIA = 9749160.83504

CONFIGS = [
    ("Serial", 1, "serial_baseline", "serial"),
    ("MPI-only", 12, "benchmark_12cpu_uniform_v2", "mpi"),
    ("MPI-only", 24, "benchmark_24cpu", "mpi"),
    ("MPI-only", 48, "benchmark_48cpu", "mpi"),
    ("Hybrid", 12, "benchmark_12cpu_uniform_v2", "hybrid"),
    ("Hybrid", 24, "benchmark_24cpu", "hybrid"),
    ("Hybrid", 48, "benchmark_48cpu", "hybrid"),
]

def extract(pattern, content, label):
    match = re.search(pattern, content)
    if not match:
        raise ValueError("Missing " + label)
    return float(match.group(1))

def read_result(path):
    if not path.is_file():
        raise FileNotFoundError(str(path))

    content = path.read_text()

    runtime = extract(
        r"Runtime \(seconds\):\s*([0-9.eE+-]+)",
        content, "runtime"
    )
    iterations = int(extract(
        r"Iterations:\s*(\d+)",
        content, "iterations"
    ))
    inertia = extract(
        r"Inertia:\s*([0-9.eE+-]+)",
        content, "inertia"
    )

    if runtime <= 0 or not math.isfinite(runtime):
        raise ValueError("Invalid runtime: " + str(path))

    if iterations != EXPECTED_ITERATIONS:
        raise ValueError("Unexpected iterations: " + str(path))

    if not math.isclose(
        inertia, EXPECTED_INERTIA, rel_tol=1e-8, abs_tol=1e-6
    ):
        raise ValueError("Unexpected inertia: " + str(path))

    return runtime

rows = []
raw_rows = []

for version, cpus, directory, prefix in CONFIGS:
    runtimes = []

    for i in range(1, 6):
        path = ROOT / directory / "{}_{}.txt".format(prefix, i)
        runtime = read_result(path)
        runtimes.append(runtime)

        raw_rows.append({
            "version": version,
            "cpus": cpus,
            "repeat": i,
            "runtime_seconds": runtime,
        })

    rows.append({
        "version": version,
        "cpus": cpus,
        "mean_runtime": statistics.mean(runtimes),
        "median_runtime": statistics.median(runtimes),
        "std_runtime": statistics.stdev(runtimes),
        "min_runtime": min(runtimes),
        "max_runtime": max(runtimes),
    })

baseline = next(
    r["median_runtime"]
    for r in rows
    if r["version"] == "Serial"
)

for row in rows:
    row["speedup"] = baseline / row["median_runtime"]
    row["efficiency"] = row["speedup"] / row["cpus"]

def save_csv(path, records):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

save_csv(OUT / "raw_runtime.csv", raw_rows)
save_csv(OUT / "scaling_summary.csv", rows)

print("=== STRONG SCALING SUMMARY ===")
print("Serial baseline median: {:.6f} s".format(baseline))
print()

for row in rows:
    print(
        "{:<9} {:>2} CPUs | Median={:.6f}s | "
        "Speedup={:.2f}x | Efficiency={:.2f}%".format(
            row["version"],
            row["cpus"],
            row["median_runtime"],
            row["speedup"],
            row["efficiency"] * 100
        )
    )

print()
print("Analysis completed successfully.")
print("Saved:", OUT / "raw_runtime.csv")
print("Saved:", OUT / "scaling_summary.csv")
