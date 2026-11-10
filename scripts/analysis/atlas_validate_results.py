#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path

def parse_result(path):
    text = Path(path).read_text()

    def get_number(label, cast):
        match = re.search(
            rf"^{re.escape(label)}:\s*(\S+)",
            text, re.MULTILINE
        )
        if not match:
            raise ValueError(f"{path}: missing {label}")
        return cast(match.group(1))

    iterations = get_number("Iterations", int)
    inertia = get_number("Inertia", float)

    centroids = {}
    for match in re.finditer(
        r"^Centroid\s+(\d+):\s*(.*)$",
        text, re.MULTILINE
    ):
        index = int(match.group(1))
        if index in centroids:
            raise ValueError(f"{path}: duplicate centroid {index}")
        centroids[index] = [
            float(x) for x in match.group(2).split()
        ]

    if not centroids:
        raise ValueError(f"{path}: no centroids found")

    if not math.isfinite(inertia) or inertia < 0:
        raise ValueError(f"{path}: invalid inertia")

    for values in centroids.values():
        if not values or not all(map(math.isfinite, values)):
            raise ValueError(f"{path}: invalid centroid")

    return iterations, inertia, centroids

def compare(reference, candidate, name, rtol, atol):
    ref_iter, ref_inertia, ref_centroids = reference
    got_iter, got_inertia, got_centroids = candidate

    errors = []

    if ref_iter != got_iter:
        errors.append(
            f"iterations: {ref_iter} vs {got_iter}"
        )

    if not math.isclose(
        ref_inertia, got_inertia, rel_tol=rtol, abs_tol=atol
    ):
        errors.append(
            f"inertia: {ref_inertia} vs {got_inertia}"
        )

    if ref_centroids.keys() != got_centroids.keys():
        errors.append("centroid indices differ")
    else:
        for k in sorted(ref_centroids):
            a = ref_centroids[k]
            b = got_centroids[k]

            if len(a) != len(b):
                errors.append(f"centroid {k}: dimension mismatch")
                continue

            for d, (x, y) in enumerate(zip(a, b)):
                if not math.isclose(
                    x, y, rel_tol=rtol, abs_tol=atol
                ):
                    errors.append(
                        f"centroid {k}, dimension {d}: {x} vs {y}"
                    )

    if errors:
        print(f"FAIL: {name}")
        for error in errors[:20]:
            print("  " + error)
        return False

    print(f"PASS: {name}")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("serial")
    parser.add_argument("mpi")
    parser.add_argument("hybrid")
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args()

    reference = parse_result(args.serial)
    mpi = parse_result(args.mpi)
    hybrid = parse_result(args.hybrid)

    ok_mpi = compare(
        reference, mpi, "Serial vs MPI",
        args.rtol, args.atol
    )
    ok_hybrid = compare(
        reference, hybrid, "Serial vs Hybrid",
        args.rtol, args.atol
    )

    if ok_mpi and ok_hybrid:
        print("ALL VALIDATIONS PASSED")
    else:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
