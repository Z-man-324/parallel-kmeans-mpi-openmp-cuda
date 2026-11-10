from pathlib import Path
from xml.sax.saxutils import escape
import csv
import math

BASE = Path("results/analysis")
OUT = BASE / "figures_log"
OUT.mkdir(parents=True, exist_ok=True)

with (BASE / "vanda_summary.csv").open() as f:
    rows = list(csv.DictReader(f))

def get(exp, impl, cores=None, n=None):
    for r in rows:
        if (r["experiment"] == exp
            and r["implementation"] == impl
            and (cores is None or r["cores"] == str(cores))
            and (n is None or r["N"] == str(n))):
            return float(r["median_seconds"])
    return None

def plot(filename, title, xlabel, ylabel, datasets, ideal=None):
    W, H = 960, 620
    L, R, T, B = 115, 65, 100, 110

    all_points = [
        (x, y) for _, pts, _ in datasets
        for x, y in pts if x > 0 and y > 0
    ]
    if ideal:
        all_points += ideal

    if not all_points:
        print("Skipped empty chart:", filename)
        return

    xs = [math.log10(x) for x, y in all_points]
    ys = [math.log10(y) for x, y in all_points]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    xpad = max((xmax-xmin)*0.08, 0.08)
    ypad = max((ymax-ymin)*0.12, 0.12)
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    def xp(x):
        return L+(math.log10(x)-xmin)/(xmax-xmin)*(W-L-R)

    def yp(y):
        return H-B-(math.log10(y)-ymin)/(ymax-ymin)*(H-T-B)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{W/2}" y="42" text-anchor="middle" '
        f'font-family="Arial" font-size="24" font-weight="bold">'
        f'{escape(title)}</text>'
    ]

    # Logarithmic grid and ticks
    for axis in ("x", "y"):
        lo, hi = (xmin, xmax) if axis == "x" else (ymin, ymax)
        ticks = []
        for power in range(math.floor(lo)-1, math.ceil(hi)+2):
            for m in (1, 2, 5):
                value = m * 10**power
                if lo <= math.log10(value) <= hi:
                    ticks.append(value)

        for value in ticks:
            if axis == "x":
                pos = xp(value)
                parts.append(
                    f'<line x1="{pos}" y1="{T}" x2="{pos}" '
                    f'y2="{H-B}" stroke="#e5e7eb"/>'
                    f'<text x="{pos}" y="{H-B+26}" text-anchor="middle" '
                    f'font-family="Arial" font-size="13">{value:g}</text>'
                )
            else:
                pos = yp(value)
                parts.append(
                    f'<line x1="{L}" y1="{pos}" x2="{W-R}" '
                    f'y2="{pos}" stroke="#e5e7eb"/>'
                    f'<text x="{L-12}" y="{pos+5}" text-anchor="end" '
                    f'font-family="Arial" font-size="13">{value:g}</text>'
                )

    if ideal:
        pts = " ".join(f"{xp(x)},{yp(y)}" for x,y in ideal)
        parts.append(
            f'<polyline points="{pts}" fill="none" '
            f'stroke="#777777" stroke-width="2" '
            f'stroke-dasharray="7,5"/>'
        )

    for j, (name, points, color) in enumerate(datasets):
        points = sorted(points)
        pts = " ".join(f"{xp(x)},{yp(y)}" for x,y in points)
        parts.append(
            f'<polyline points="{pts}" fill="none" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        for x,y in points:
            parts.append(
                f'<circle cx="{xp(x)}" cy="{yp(y)}" '
                f'r="6" fill="{color}"/>'
            )
        lx = L + j*240
        parts.append(
            f'<circle cx="{lx}" cy="73" r="5" fill="{color}"/>'
            f'<text x="{lx+13}" y="78" font-family="Arial" '
            f'font-size="15">{escape(name)}</text>'
        )

    parts += [
        f'<text x="{W/2}" y="{H-24}" text-anchor="middle" '
        f'font-family="Arial" font-size="17">{escape(xlabel)} (log scale)</text>',
        f'<text transform="translate(28,{H/2}) rotate(-90)" '
        f'text-anchor="middle" font-family="Arial" font-size="17">'
        f'{escape(ylabel)} (log scale)</text>',
        '</svg>'
    ]

    path = OUT / filename
    path.write_text("\n".join(parts))
    print("Created:", path)

cores = (36, 72, 144, 288)
colors = {"mpi":"#2563eb", "hybrid":"#dc2626"}

def scaling_points(exp, impl):
    result = []
    for c in cores:
        n = 2000000 if exp == "strong" else c*10000
        t = get(exp, impl, c, n)
        if t is not None:
            result.append((c,t))
    return result

plot(
    "figure1_strong_runtime_log.svg",
    "Strong Scaling: Median Runtime",
    "CPU cores", "Runtime (seconds)",
    [(v.upper(), scaling_points("strong",v), colors[v])
     for v in ("mpi","hybrid")]
)

speedup_data = []
for v in ("mpi","hybrid"):
    points = scaling_points("strong",v)
    if points:
        baseline = points[0][1]
        speedup_data.append(
            (v.upper(), [(c,baseline/t) for c,t in points],colors[v])
        )

plot(
    "figure2_strong_speedup_log.svg",
    "Strong Scaling: Relative Speedup",
    "CPU cores", "Speedup",
    speedup_data,
    ideal=[(c,c/36) for c in cores]
)

plot(
    "figure3_weak_runtime_log.svg",
    "Weak Scaling: Median Runtime",
    "CPU cores", "Runtime (seconds)",
    [(v.upper(),scaling_points("weak",v),colors[v])
     for v in ("mpi","hybrid")]
)

sizes = (100000,500000,1000000,2000000)
plot(
    "figure4_gpu_runtime_log.svg",
    "CPU vs GPU: Median Runtime",
    "Number of data points", "Runtime (seconds)",
    [
        (v.upper(),
         [(n,get("gpu_benchmark",v,n=n)) for n in sizes
          if get("gpu_benchmark",v,n=n) is not None],
         color)
        for v,color in (("cpu","#2563eb"),("gpu","#dc2626"))
    ]
)

print("=== LOG-SCALE FIGURES COMPLETED ===")
