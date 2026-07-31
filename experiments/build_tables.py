#!/usr/bin/env python3
#Build publication-facing CSV and LaTeX table summaries from fresh outputs.
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = Path(os.environ.get("LPG_RESULT_DIR", ROOT / "results")).resolve()
TABLE_DIR = Path(os.environ.get("LPG_TABLE_DIR", ROOT / "tables")).resolve()
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_json(name: str) -> dict[str, Any]:
    path = RESULT_DIR / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {name}")
    with (TABLE_DIR / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def state_dependent_table() -> list[dict[str, Any]]:
    data = load_json("state_dependent_policy.json")
    rows = []
    for key, label in (("constrained", "LPG, R0=2"), ("unconstrained", "Unconstrained")):
        group = data[key]
        rows.append({
            "branch": label,
            "final_return_mean": group["final_mean"],
            "final_return_sd": group["final_sd"],
            "final_norm_mean": group["norm_mean"],
            "mapping_norm_mean": group["mapping_mean"],
            "mapping_norm_sd": group["mapping_sd"],
            "active_updates_percent": 100.0 * group["active_mean"] if key == "constrained" else "",
        })
    comparison = data["paired_constrained_minus_unconstrained"]
    write_csv("state_dependent_policy.csv", rows)
    write_csv("state_dependent_paired_difference.csv", [{
        "mean": comparison["mean"],
        "sd": comparison["sd"],
        "ci95_low": comparison["ci95"][0],
        "ci95_high": comparison["ci95"][1],
    }])
    return rows


def conditioning_table() -> list[dict[str, Any]]:
    data = load_json("conditioning_stress_test.json")
    labels = {
        "direct": "Direct ordinary gradient",
        "redundant": "Redundant ordinary gradient",
        "redundant_ng": "Redundant ridge metric correction",
    }
    rows = []
    for key in ("direct", "redundant", "redundant_ng"):
        arm = data["arms"][key]
        rows.append({
            "update_geometry": labels[key],
            "final_return_mean": arm["final_mean"],
            "final_return_sd": arm["final_sd"],
            "auc_times_1e5_mean": arm["auc_mean"] / 1e5,
            "auc_times_1e5_sd": arm["auc_sd"] / 1e5,
        })
    write_csv("conditioning_stress_test.csv", rows)
    comparisons = []
    for name, value in data["comparisons"].items():
        if isinstance(value, dict):
            comparisons.append({
                "comparison": name,
                "mean": value["mean"],
                "sd": value["sd"],
                "ci95_low": value["ci95"][0],
                "ci95_high": value["ci95"][1],
            })
        else:
            comparisons.append({"comparison": name, "mean": value, "sd": "", "ci95_low": "", "ci95_high": ""})
    write_csv("conditioning_comparisons.csv", comparisons)
    return rows


def learning_rate_table() -> list[dict[str, Any]]:
    data = load_json("learning_rate_ablation.json")
    rows = []
    for arm in ("direct", "redundant", "redundant_ng"):
        for entry in data[arm]["tuning"]:
            rows.append({
                "arm": arm,
                "learning_rate": entry["lr"],
                "tuning_final_mean": entry["final_mean"],
                "tuning_final_sd": entry["final_sd"],
                "tuning_auc_mean": entry["auc_mean"],
                "tuning_auc_sd": entry["auc_sd"],
                "selected": entry["lr"] == data[arm]["selected_lr"],
            })
    write_csv("learning_rate_study.csv", rows)
    selected_rows = []
    for arm in ("direct", "redundant", "redundant_ng"):
        held = data[arm]["heldout"]
        selected_rows.append({
            "arm": arm,
            "selected_learning_rate": data[arm]["selected_lr"],
            "heldout_final_mean": held["final_mean"],
            "heldout_final_sd": held["final_sd"],
            "heldout_auc_mean": held["auc_mean"],
            "heldout_auc_sd": held["auc_sd"],
            "grid_bracketed": (
                min(data["grid"]) < data[arm]["selected_lr"] < max(data["grid"])
            ),
        })
    write_csv("learning_rate_selected.csv", selected_rows)
    return rows


def armijo_table() -> list[dict[str, Any]]:
    source = RESULT_DIR / "pullback_armijo.csv"
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    selected_indices = {0, 1, 2, 4, 7}
    selected = [row for row in rows if int(row["iteration"]) in selected_indices]
    summary_values = {}
    for line in (RESULT_DIR / "pullback_armijo_summary.txt").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            summary_values[key] = value
    terminal = {key: "" for key in rows[0].keys()}
    terminal.update({
        "iteration": "8",
        "objective": summary_values["final_objective"],
        "mapping_norm": summary_values["final_mapping_norm"],
    })
    selected.append(terminal)
    shutil.copy2(source, TABLE_DIR / "pullback_armijo_all_iterations.csv")
    write_csv("pullback_armijo_selected_iterations.csv", selected)
    return selected


def fixed_ratio_table() -> None:
    source = RESULT_DIR / "fixed_ratio_fisher.csv"
    shutil.copy2(source, TABLE_DIR / "fixed_ratio_fisher.csv")


def latex_fragment(state_rows: list[dict[str, Any]], condition_rows: list[dict[str, Any]]) -> None:
    lines = [
        "% Automatically built from freshly generated numerical outputs.",
        "% These fragments are optional; the manuscript currently contains its tables inline.",
        "",
        "% State-dependent policy table rows",
    ]
    for row in state_rows:
        active = row["active_updates_percent"]
        active_text = "--" if active == "" else f"{float(active):.0f}\\%"
        lines.append(
            f"{row['branch']} & ${float(row['final_return_mean']):.2f}\\pm{float(row['final_return_sd']):.2f}$ "
            f"& ${float(row['final_norm_mean']):.3f}$ & "
            f"${float(row['mapping_norm_mean']):.2f}\\pm{float(row['mapping_norm_sd']):.2f}$ & {active_text} \\\\"
        )
    lines.extend(["", "% Conditioning stress-test table rows"])
    for row in condition_rows:
        lines.append(
            f"{row['update_geometry']} & ${float(row['final_return_mean']):.2f}\\pm{float(row['final_return_sd']):.2f}$ "
            f"& ${float(row['auc_times_1e5_mean']):.3f}\\pm{float(row['auc_times_1e5_sd']):.3f}$ \\\\"
        )
    (TABLE_DIR / "manuscript_table_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    state_rows = state_dependent_table()
    condition_rows = conditioning_table()
    learning_rate_table()
    armijo_table()
    fixed_ratio_table()
    latex_fragment(state_rows, condition_rows)
    print(f"table outputs written to {TABLE_DIR}")


if __name__ == "__main__":
    main()
