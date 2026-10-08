#!/usr/bin/env python
"""Merge the deep-model and hand-crafted summaries into results/comparison.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _cell(entry, key, use_pooled):
    if not entry:
        return "-"
    if use_pooled:
        v = entry.get(key, {})
        return f"{v['eer']*100:.1f}" if v else "-"
    v = entry.get(key, {})
    return f"{v['mean']*100:.1f} ± {v['std']*100:.1f}" if v else "-"


def _short_b(key: str) -> str:
    """'train_multi_protocol|single_enrol_cross_verify|MSE_STD' -> readable."""
    parts = key.split("|")
    if len(parts) == 3:
        regime = {"handcrafted": "hand-crafted",
                  "train_single_protocol": "train single-proto",
                  "train_multi_protocol": "train multi-proto"}.get(parts[0], parts[0])
        return f"{regime} | cross-task | {parts[2]}"
    return key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "results")
    args = ap.parse_args()
    res = args.results_dir

    deep = _load(res / "summary.json")
    base = _load(res / "baseline_summary.json")
    if deep is None and base is None:
        print("no summaries found")
        return

    lines = ["# Deep (channel-specific siamese CNN) vs hand-crafted (AR+MFCC)\n"]
    if deep and "config" in deep:
        c = deep["config"]
        lines.append(f"_Configuration_: {c['n_subjects']} subjects, "
                     f"{len(c['channels'])} channels, {c['folds']} folds, "
                     f"{c['epochs']} epochs, "
                     f"{c['pairs_per_subject']} pairs/subject.\n")

    lines.append("## A. Within-task (paper Table 3 analogue)\n")
    lines.append("| Scenario | Hand-crafted (EER %) | Deep (EER %) |")
    lines.append("|---|---|---|")
    keys = sorted((deep or {}).get("A_pooled", {}) or
                  (base or {}).get("A_pooled", {}))
    for k in keys:
        lines.append(f"| {k} | {_cell(base.get('A_pooled'), k, True)} "
                     f"| {_cell(deep.get('A_pooled'), k, True)} |")

    lines.append("\n## B. Task-independent (paper Table 4 analogue)\n")
    lines.append("| Cross-task verification | Hand-crafted (EER %) | "
                 "Deep, single-proto training (EER %) | "
                 "Deep, multi-proto training (EER %) |")
    lines.append("|---|---|---|---|")
    base_b = (base or {}).get("B_pooled", {})
    deep_b = (deep or {}).get("B_pooled", {})

    def _pick(bag, needle):
        for k, v in bag.items():
            if k.endswith(needle):
                return f"{v['eer']*100:.1f}"
        return "-"

    for scen in ("SSE_STD", "MSE_STD"):
        lines.append(
            f"| single-task enrolment, `{scen}` | {_pick(base_b, scen)} | "
            f"{_pick({k: v for k, v in deep_b.items() if 'single_protocol' in k}, scen)} | "
            f"{_pick({k: v for k, v in deep_b.items() if 'multi_protocol' in k}, scen)} |")

    lines.append("\n## C. Per-channel capability (paper Table 5 analogue)\n")
    lines.append("| Channel | Hand-crafted (EER %) | Deep (EER %) |")
    lines.append("|---|---|---|")
    deep_c = (deep or {}).get("C_per_channel", {})
    base_c = (base or {}).get("C_per_channel", {})
    for k in sorted(deep_c, key=lambda c: deep_c[c]["mean"]):
        b = base_c.get(k)
        bcell = f"{b['mean']*100:.1f}" if b else "-"
        lines.append(f"| {k} | {bcell} | {deep_c[k]['mean']*100:.1f} |")

    lines.append("\n_(Deep numbers are per-fold means; see `tables.md` for the "
                 "pooled estimates and standard deviations.)_\n")
    out = res / "comparison.md"
    out.write_text("\n".join(lines).rstrip("\n") + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
