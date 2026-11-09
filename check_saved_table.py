"""Check Appendix J removal contrasts using the bundled saved metric rows."""

import csv
from hashlib import sha256
from math import isclose, isfinite
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
HASHES = {
    "case-summary.csv": "272fc2839ae0f3b4fa9e3c92f091d00f2b6bcc0ca028fbda98a3c4eadbe92838",
    "removal-deltas.csv": "94f604d2d6ddb09b9c1f963fe1bd92419db9ca98f214561643687e4dfeb49844",
}
UNITS = ("g1-visa-pcb1", "g2-visa-pcb1", "g1-mvtec-tile")
METHODS = ("AB", "AW", "BW", "ABW")
CONTROLS = {"A": "BW", "B": "AW", "W": "AB"}
METRICS = ("ap", "auroc", "aupro_fpr30", "small_aupro_fpr30", "aupro_fpr01",
           "small_coverage_at_fpr01", "background_fpr", "clean_fpr")


def read_rows(name: str) -> list[dict[str, str]]:
    path = RESULTS / name
    if sha256(path.read_bytes()).hexdigest() != HASHES[name]:
        raise ValueError(f"saved input hash differs for {name}")
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    case_rows = read_rows("case-summary.csv")
    delta_rows = read_rows("removal-deltas.csv")
    cases = {(row["unit"], row["method"]): row for row in case_rows}
    deltas = {(row["unit"], row["removed_module"]): row for row in delta_rows}
    if (len(case_rows) != len(cases) or set(cases) != {(u, m) for u in UNITS for m in METHODS}
            or len(delta_rows) != len(deltas) or set(deltas) != {(u, m) for u in UNITS for m in CONTROLS}):
        raise ValueError("saved cases or removal rows have an unexpected population")

    checked = 0
    for (unit, removed), row in deltas.items():
        control = CONTROLS[removed]
        if row["control"] != control or row["comparison"] != f"ABW-minus-{control}":
            raise ValueError(f"unexpected comparison for {unit}, removal {removed}")
        for metric in METRICS:
            column = f"{metric}_percent"
            full = cases[unit, "ABW"][column]
            base = cases[unit, control][column]
            expected = row[f"{metric}_delta_pp"]
            if unit == "g1-mvtec-tile" and metric in METRICS[3:]:
                if any((full, base, expected)):
                    raise ValueError(f"unexpected tile-only value for {metric}")
                continue
            computed = float(full) - float(base)
            if not isfinite(computed) or not isclose(computed, float(expected), rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"contrast differs for {unit}, removal {removed}, {metric}")
            checked += 1

    print("Pooled native pixel AP (%) from saved metric rows")
    print("| Configuration | VisA pcb1 g1 | VisA pcb1 g2 | MVTec AD tile g1 |")
    print("| --- | ---: | ---: | ---: |")
    for method in METHODS:
        values = " | ".join(f"{float(cases[unit, method]['ap_percent']):.4f}" for unit in UNITS)
        print(f"| {method} | {values} |")
    print(f"Checked {len(HASHES)} CSV hashes, {len(cases)} cases and {checked} metric contrasts.")


if __name__ == "__main__":
    main()
