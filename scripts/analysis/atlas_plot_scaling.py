from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT = Path("results/analysis/scaling_summary.csv")
OUT = Path("results/analysis/figures")
OUT.mkdir(parents=True, exist_ok=True)

with INPUT.open(newline="") as f:
    rows = list(csv.DictReader(f))

data = {}
for row in rows:
    version = row["version"]
    if version == "Serial":
        continue

    data.setdefault(version, []).append({
        "cpus": int(row["cpus"]),
        "runtime": float(row["median_runtime"]),
        "speedup": float(row["speedup"]),
        "efficiency": float(row["efficiency"]) * 100
    })

for version in data:
    data[version].sort(key=lambda x: x["cpus"])

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300
})

def save_figure(filename):
    plt.tight_layout()
    plt.savefig(OUT / (filename + ".png"), dpi=300)
    plt.savefig(OUT / (filename + ".pdf"))
    plt.close()
    print("Saved:", filename)

def draw(metric, ylabel, title, filename,
         ideal=False, percent=False):
    plt.figure(figsize=(7.5, 5))

    for version, records in data.items():
        x = [r["cpus"] for r in records]
        y = [r[metric] for r in records]

        plt.plot(
            x, y,
            marker="o",
            linewidth=2,
            markersize=7,
            label=version
        )

    if ideal:
        plt.plot(
            [12, 24, 48],
            [12, 24, 48],
            linestyle="--",
            color="gray",
            label="Ideal linear speedup"
        )

    if percent:
        plt.axhline(
            100,
            linestyle="--",
            color="gray",
            label="Ideal efficiency (100%)"
        )

    plt.xticks([12, 24, 48])
    plt.xlabel("Number of CPUs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    save_figure(filename)

draw(
    "runtime",
    "Median Core Runtime (seconds)",
    "K-means Strong Scaling: Runtime",
    "runtime_vs_cpus"
)

draw(
    "speedup",
    "Core Computation Speedup",
    "K-means Strong Scaling: Speedup",
    "speedup_vs_cpus",
    ideal=True
)

draw(
    "efficiency",
    "Parallel Efficiency (%)",
    "K-means Strong Scaling: Efficiency",
    "efficiency_vs_cpus",
    percent=True
)

print("All three figures generated successfully.")
