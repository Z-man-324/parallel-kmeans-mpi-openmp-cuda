import csv
from pathlib import Path

base = Path("results/analysis")

with (base / "vanda_summary.csv").open() as f:
    rows = list(csv.DictReader(f))

lookup = {
    (r["experiment"], r["implementation"], r["cores"], r["N"]): r
    for r in rows
}

results = []

for version in ("mpi", "hybrid"):
    baseline = lookup.get(("strong", version, "36", "2000000"))

    if baseline:
        t36 = float(baseline["median_seconds"])

        for cores in (36, 72, 144, 288):
            row = lookup.get(("strong", version, str(cores), "2000000"))
            if not row:
                continue

            runtime = float(row["median_seconds"])
            speedup = t36 / runtime
            efficiency = speedup / (cores / 36)

            results.append({
                "experiment": "strong",
                "implementation": version,
                "cores": cores,
                "N": 2000000,
                "median_seconds": runtime,
                "relative_speedup": speedup,
                "efficiency": efficiency
            })

    weak_baseline = lookup.get(
        ("weak", version, "36", "360000")
    )

    if weak_baseline:
        t36 = float(weak_baseline["median_seconds"])

        for cores in (36, 72, 144, 288):
            n = cores * 10000
            row = lookup.get(("weak", version, str(cores), str(n)))
            if not row:
                continue

            runtime = float(row["median_seconds"])

            results.append({
                "experiment": "weak",
                "implementation": version,
                "cores": cores,
                "N": n,
                "median_seconds": runtime,
                "relative_speedup": "",
                "efficiency": t36 / runtime
            })

out = base / "vanda_scaling_metrics.csv"

with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "experiment", "implementation", "cores", "N",
        "median_seconds", "relative_speedup", "efficiency"
    ])
    writer.writeheader()
    writer.writerows(results)

print("=== SCALING METRICS ===")
for r in results:
    print(r)

print("Saved:", out)
