#!/usr/bin/env python
"""Run the reproduction experiments of Maiorana (2021).

Three experiments mirror the three analysis blocks of the paper:

  A. within-task recognition  (Table 3 analogue)
  B. task-independent recognition (Table 4 analogue)
  C. per-channel recognition capability (Table 5 analogue)

Models are always trained on a subject-disjoint training split and evaluated
on held-out subjects (open-set conditions, 5-fold cross-validation as in the
paper).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as data_mod                     # noqa: E402
from src import evaluate as eval_mod                  # noqa: E402
from src import train as train_mod                    # noqa: E402
from src.config import ExperimentConfig, PAPER_CHANNELS  # noqa: E402

def train_channel_models(frames, train_subjects, train_tasks, allow_cross_task,
                         cfg: ExperimentConfig, channel_ids, workers: int):
    """Train one CNN per channel; returns {channel_id: state_dict}."""
    out: dict[int, dict] = {}

    def job(ci):
        model = train_mod.train_channel_model(
            frames, train_subjects, cfg.sessions, train_tasks, ci, cfg.train,
            allow_cross_task=allow_cross_task, device="cpu",
        )
        return ci, model.state_dict()

    if workers > 1:
        # Channels are independent, and torch releases the GIL during its ops,
        # so thread-level parallelism gives a near-linear speedup here.
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for ci, sd in ex.map(job, channel_ids):
                    out[ci] = sd
        finally:
            torch.set_num_threads(old)
    else:
        for ci in channel_ids:
            ci, sd = job(ci)
            out[ci] = sd
    return out


def load_models(state_dicts, cfg: ExperimentConfig, device="cpu"):
    from src.model import ChannelCNN
    models = []
    for ci in sorted(state_dicts):
        m = ChannelCNN(dropout=cfg.train.dropout,
                       embedding_dim=cfg.train.embedding_dim).to(device)
        m.load_state_dict(state_dicts[ci])
        m.eval()
        models.append(m)
    return models


# --------------------------------------------------------------------------- #
# Scenario builders
# --------------------------------------------------------------------------- #
def _spec(pairs):
    """pairs: {subject: [(session, task), ...]}."""
    return {s: list(v) for s, v in pairs.items()}


def scenario_within(task, enrol, probe, subjects):
    return (_spec({s: [(enrol, task)] for s in subjects}),
            _spec({s: [(probe, task)] for s in subjects}))


def scenario_within_mse(task, enrol_list, probe, subjects):
    return (_spec({s: [(e, task) for e in enrol_list] for s in subjects}),
            _spec({s: [(probe, task)] for s in subjects}))


def scenario_cross(train_task, probe_task, enrol, probe, subjects):
    return (_spec({s: [(enrol, train_task)] for s in subjects}),
            _spec({s: [(probe, probe_task)] for s in subjects}))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--pairs-per-subject", type=int, default=None)
    ap.add_argument("--channels", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-probe-frames", type=int, default=40)
    ap.add_argument("--quick", action="store_true",
                    help="small, fast configuration for a smoke/demo run")
    args = ap.parse_args()

    cfg = ExperimentConfig()
    if args.quick:
        args.folds = min(args.folds, 3)
        cfg.train.epochs = args.epochs or 8
        cfg.train.pairs_per_subject = args.pairs_per_subject or 30
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.pairs_per_subject:
        cfg.train.pairs_per_subject = args.pairs_per_subject
    if args.channels:
        cfg.channels = args.channels

    cache = data_mod.load_cache(cfg.cache_dir)
    frames = cache["frames"]
    channels = cache["index"]["channels"]
    channel_ids = list(range(len(channels)))
    subjects = sorted({k.split("|")[0] for k in frames})
    tasks = cfg.tasks

    print(f"subjects={len(subjects)} channels={len(channels)} "
          f"tasks={tasks} folds={args.folds} epochs={cfg.train.epochs} "
          f"pairs_per_subject={cfg.train.pairs_per_subject} workers={args.workers}")

    rng = np.random.default_rng(cfg.train.seed)
    order = rng.permutation(len(subjects))
    folds = np.array_split(order, args.folds)

    results = {
        "config": {
            "folds": args.folds, "channels": channels, "tasks": tasks,
            "epochs": cfg.train.epochs,
            "pairs_per_subject": cfg.train.pairs_per_subject,
            "margin": cfg.train.margin,
            "batch_size": cfg.train.batch_size,
            "lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay,
            "n_subjects": len(subjects),
        },
        "A_within_task": {},   # regimen -> scenario -> list over folds
        "B_task_independent": {},
        "C_per_channel": {},   # channel -> list over folds
    }
    # pooled genuine/impostor scores, used for a variance-reduced EER estimate
    pool: dict[str, dict[str, dict[str, list]]] = {"A": {}, "B": {}}

    def pool_add(section, key, r):
        d = pool[section].setdefault(key, {"gen": [], "imp": []})
        d["gen"].extend(np.asarray(r["genuine_scores"]).tolist())
        d["imp"].extend(np.asarray(r["impostor_scores"]).tolist())

    out_dir = Path(cfg.results_dir) / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    for fi, test_idx in enumerate(folds):
        test_subjects = [subjects[i] for i in test_idx]
        train_subjects = [s for j, s in enumerate(subjects) if j not in set(test_idx)]
        print(f"\n=== fold {fi+1}/{len(folds)} | train={len(train_subjects)} "
              f"test={len(test_subjects)} | {test_subjects} ===  "
              f"[{time.time()-t_start:.0f}s elapsed]")

        # -- train the three model sets used by experiments A/B -----------------
        regimes = {
            "single_T1": (["T1"], False),
            "single_T2": (["T2"], False),
            "multi": (tasks, True),
        }
        fold_models: dict[str, dict[int, dict]] = {}
        for name, (tt, cross) in regimes.items():
            t0 = time.time()
            fold_models[name] = train_channel_models(
                frames, train_subjects, tt, cross, cfg, channel_ids, args.workers)
            torch.save(fold_models[name], out_dir / f"fold{fi}_{name}.pt")
            print(f"  trained regime '{name}' in {time.time()-t0:.0f}s")

        def record_A(key, r):
            results["A_within_task"].setdefault(key, []).append(r["eer"])
            pool_add("A", key, r)

        def record_B(key, r):
            results["B_task_independent"].setdefault(key, []).append(r["eer"])
            pool_add("B", key, r)

        def record_C(ch, val):
            results["C_per_channel"].setdefault(ch, []).append(val)

        # -- Experiment A: within-task (Table 3 analogue) ----------------------
        for task, regime in (("T1", "single_T1"), ("T2", "single_T2")):
            models = load_models(fold_models[regime], cfg)
            enrol, probe = scenario_within(task, "R1", "R2", test_subjects)
            r = eval_mod.evaluate_scenario(models, frames, channels, enrol, probe,
                                           args.max_probe_frames, seed=fi)
            record_A(f"{task}|SSE_STD", r)
            enrol, probe = scenario_within_mse(task, ["R1", "R2"], "R3", test_subjects)
            r2 = eval_mod.evaluate_scenario(models, frames, channels, enrol, probe,
                                            args.max_probe_frames, seed=fi + 100)
            record_A(f"{task}|MSE_STD", r2)
            print(f"  [A] {task}: SSE_STD EER={r['eer']*100:.1f}%  "
                  f"MSE_STD EER={r2['eer']*100:.1f}%")

        # -- Experiment B: task-independent (Table 4 analogue) -----------------
        # (b) training on a single protocol, (c) training on multiple protocols
        for regime_name, regime in (("train_single_protocol", "single_T1"),
                                    ("train_multi_protocol", "multi")):
            models = load_models(fold_models[regime], cfg)
            # single-task enrolment + cross-task verification (T1 enrol -> T2 probe)
            enrol, probe = scenario_cross("T1", "T2", "R1", "R2", test_subjects)
            r = eval_mod.evaluate_scenario(models, frames, channels, enrol, probe,
                                           args.max_probe_frames, seed=fi + 200)
            record_B(f"{regime_name}|single_enrol_cross_verify|SSE_STD", r)
            # single-task enrolment + cross-task, multiple-session enrolment
            enrol = {s: [("R1", "T1"), ("R2", "T1")] for s in test_subjects}
            probe = {s: [("R3", "T2")] for s in test_subjects}
            r2 = eval_mod.evaluate_scenario(models, frames, channels, enrol, probe,
                                            args.max_probe_frames, seed=fi + 300)
            record_B(f"{regime_name}|single_enrol_cross_verify|MSE_STD", r2)
            print(f"  [B] {regime_name}: cross SSE={r['eer']*100:.1f}% "
                  f"MSE={r2['eer']*100:.1f}%")

            if regime_name == "train_multi_protocol":
                # per-channel capability for the cross-task MSE scenario
                enrol = {s: [("R1", "T1"), ("R2", "T1")] for s in test_subjects}
                probe = {s: [("R3", "T2")] for s in test_subjects}
                r3 = eval_mod.evaluate_scenario(models, frames, channels, enrol, probe,
                                                args.max_probe_frames, seed=fi + 400)
                for ch, v in r3["per_channel_eer"].items():
                    record_C(ch, v)

    # ------------------------------------------------------------------ #
    # Aggregate
    # ------------------------------------------------------------------ #
    def agg(d):
        return {k: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)),
                    "values": [float(x) for x in v]} for k, v in d.items()}

    def pooled(section):
        """EER computed on the scores pooled over all folds (lower variance)."""
        out = {}
        for k, d in pool[section].items():
            eer, _ = eval_mod.compute_eer(np.asarray(d["gen"]), np.asarray(d["imp"]))
            out[k] = {"eer": float(eer), "n_genuine": len(d["gen"]),
                      "n_impostor": len(d["imp"])}
        return out

    summary = {
        "config": results["config"],
        "A_within_task": agg(results["A_within_task"]),
        "B_task_independent": agg(results["B_task_independent"]),
        "C_per_channel": agg(results["C_per_channel"]),
        "A_pooled": pooled("A"),
        "B_pooled": pooled("B"),
        "elapsed_seconds": time.time() - t_start,
    }
    out = Path(cfg.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out/'summary.json'} (elapsed {summary['elapsed_seconds']:.0f}s)")

    write_tables(summary, out)
    make_figures(summary, out)


def write_tables(summary: dict, out: Path) -> None:
    lines = ["# Reproduction results (EER, %)\n",
             f"- subjects: {summary['config']['n_subjects']}, "
             f"channels: {len(summary['config']['channels'])}, "
             f"folds: {summary['config']['folds']}\n"]
    lines.append("\n## A. Within-task (Table 3 analogue)\n")
    lines.append("| Protocol / Scenario | EER % (mean ± std over folds) | EER % (pooled) |\n|---|---|---|")
    for k, v in sorted(summary["A_within_task"].items()):
        p = summary.get("A_pooled", {}).get(k, {})
        pv = f"{p['eer']*100:.1f}" if p else "-"
        lines.append(f"| {k} | {v['mean']*100:.1f} ± {v['std']*100:.1f} | {pv} |")
    lines.append("\n## B. Task-independent (Table 4 analogue)\n")
    lines.append("| Training / Verification | EER % (mean ± std over folds) | EER % (pooled) |\n|---|---|---|")
    for k, v in sorted(summary["B_task_independent"].items()):
        p = summary.get("B_pooled", {}).get(k, {})
        pv = f"{p['eer']*100:.1f}" if p else "-"
        lines.append(f"| {k} | {v['mean']*100:.1f} ± {v['std']*100:.1f} | {pv} |")
    lines.append("\n## C. Per-channel capability (Table 5 analogue)\n")
    lines.append("| Channel | EER % |\n|---|---|")
    for k, v in sorted(summary["C_per_channel"].items(),
                       key=lambda kv: kv[1]["mean"]):
        lines.append(f"| {k} | {v['mean']*100:.1f} ± {v['std']*100:.1f} |")
    (out / "tables.md").write_text("\n".join(lines) + "\n")


def make_figures(summary: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # A: within-task
    items = sorted(summary["A_within_task"].items())
    if items:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([k for k, _ in items], [v["mean"] * 100 for _, v in items],
               yerr=[v["std"] * 100 for _, v in items], capsize=4, color="#4C72B0")
        ax.set_ylabel("EER (%)")
        ax.set_title("A. Within-task recognition (Table 3 analogue)")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(fig_dir / "A_within_task.png", dpi=150)
        plt.close(fig)

    # B: task-independent
    items = sorted(summary["B_task_independent"].items())
    if items:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(range(len(items)), [v["mean"] * 100 for _, v in items],
               yerr=[v["std"] * 100 for _, v in items], capsize=4, color="#DD8452")
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels([k.replace("|", "\n") for k, _ in items], fontsize=7)
        ax.set_ylabel("EER (%)")
        ax.set_title("B. Task-independent recognition (Table 4 analogue)")
        fig.tight_layout()
        fig.savefig(fig_dir / "B_task_independent.png", dpi=150)
        plt.close(fig)

    # C: per-channel
    items = sorted(summary["C_per_channel"].items(), key=lambda kv: kv[1]["mean"])
    if items:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar([k for k, _ in items], [v["mean"] * 100 for _, v in items],
               yerr=[v["std"] * 100 for _, v in items], capsize=3, color="#55A868")
        ax.set_ylabel("EER (%)")
        ax.set_title("C. Per-channel capability (Table 5 analogue)")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(fig_dir / "C_per_channel.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
