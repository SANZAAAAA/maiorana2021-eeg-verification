#!/usr/bin/env python
"""Hand-crafted AR+MFCC baseline (paper Table 3a / Table 4a analogue).

Uses the same scenarios, folds and verification scheme as
``run_experiment.py`` but replaces the learned channel representations with
hand-crafted features (see ``src/baseline.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))
sys.path.insert(0, str(SCRIPTS))

from src import data as data_mod                      # noqa: E402
from src import evaluate as eval_mod                  # noqa: E402
from src.baseline import handcrafted_features         # noqa: E402
from src.config import ExperimentConfig                # noqa: E402
import run_experiment as rx                           # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-probe-frames", type=int, default=20)
    args = ap.parse_args()

    cfg = ExperimentConfig()
    cache = data_mod.load_cache(cfg.cache_dir)
    frames = cache["frames"]
    channels = cache["index"]["channels"]
    subjects = sorted({k.split("|")[0] for k in frames})

    encoders = [handcrafted_features] * len(channels)

    rng = np.random.default_rng(cfg.train.seed)
    order = rng.permutation(len(subjects))
    folds = np.array_split(order, args.folds)

    A, B, C = {}, {}, {}
    pool = {"A": {}, "B": {}}

    def pool_add(section, key, r):
        d = pool[section].setdefault(key, {"gen": [], "imp": []})
        d["gen"].extend(np.asarray(r["genuine_scores"]).tolist())
        d["imp"].extend(np.asarray(r["impostor_scores"]).tolist())

    t0 = time.time()
    for fi, test_idx in enumerate(folds):
        test_subjects = [subjects[i] for i in test_idx]
        print(f"=== fold {fi+1}/{len(folds)} | test={test_subjects} "
              f"[{time.time()-t0:.0f}s] ===")

        for task in ["T1", "T2"]:
            enrol, probe = rx.scenario_within(task, "R1", "R2", test_subjects)
            r = eval_mod.evaluate_scenario(encoders, frames, channels, enrol, probe,
                                           args.max_probe_frames, seed=fi)
            A.setdefault(f"{task}|SSE_STD", []).append(r["eer"])
            pool_add("A", f"{task}|SSE_STD", r)
            enrol, probe = rx.scenario_within_mse(task, ["R1", "R2"], "R3",
                                                  test_subjects)
            r2 = eval_mod.evaluate_scenario(encoders, frames, channels, enrol, probe,
                                            args.max_probe_frames, seed=fi + 100)
            A.setdefault(f"{task}|MSE_STD", []).append(r2["eer"])
            pool_add("A", f"{task}|MSE_STD", r2)

        enrol, probe = rx.scenario_cross("T1", "T2", "R1", "R2", test_subjects)
        r = eval_mod.evaluate_scenario(encoders, frames, channels, enrol, probe,
                                       args.max_probe_frames, seed=fi + 200)
        B.setdefault("handcrafted|single_enrol_cross_verify|SSE_STD", []).append(r["eer"])
        pool_add("B", "handcrafted|single_enrol_cross_verify|SSE_STD", r)

        enrol = {s: [("R1", "T1"), ("R2", "T1")] for s in test_subjects}
        probe = {s: [("R3", "T2")] for s in test_subjects}
        r2 = eval_mod.evaluate_scenario(encoders, frames, channels, enrol, probe,
                                        args.max_probe_frames, seed=fi + 300)
        B.setdefault("handcrafted|single_enrol_cross_verify|MSE_STD", []).append(r2["eer"])
        pool_add("B", "handcrafted|single_enrol_cross_verify|MSE_STD", r2)
        for ch, v in r2["per_channel_eer"].items():
            C.setdefault(ch, []).append(v)
        print(f"  A: { {k: round(v[-1]*100,1) for k,v in A.items()} }")
        print(f"  B: cross SSE={r['eer']*100:.1f}%  cross MSE={r2['eer']*100:.1f}%")

    def agg(d):
        return {k: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)),
                    "values": [float(x) for x in v]} for k, v in d.items()}

    def pooled(section):
        return {k: {"eer": float(eval_mod.compute_eer(np.asarray(d["gen"]),
                                                      np.asarray(d["imp"]))[0]),
                    "n_genuine": len(d["gen"]), "n_impostor": len(d["imp"])}
                for k, d in pool[section].items()}

    summary = {"A_within_task": agg(A), "B_task_independent": agg(B),
               "C_per_channel": agg(C), "A_pooled": pooled("A"),
               "B_pooled": pooled("B"), "elapsed_seconds": time.time() - t0}
    out = Path(cfg.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out/'baseline_summary.json'} in {summary['elapsed_seconds']:.0f}s")
    for section in ("A_within_task", "B_task_independent"):
        print(f"\n{section}:")
        for k, v in sorted(summary[section].items()):
            print(f"  {k}: {v['mean']*100:.1f} ± {v['std']*100:.1f} %")
    for section in ("A_pooled", "B_pooled"):
        print(f"\n{section}:")
        for k, v in sorted(summary[section].items()):
            print(f"  {k}: {v['eer']*100:.1f} %  "
                  f"(gen={v['n_genuine']}, imp={v['n_impostor']})")


if __name__ == "__main__":
    main()
