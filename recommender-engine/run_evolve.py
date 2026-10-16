"""Intuition: Phase-4 entry — evolve the policy knobs with a (1+lambda) ES against a small TRAIN
swarm, select the winner on a held-out VAL swarm, and measure it exactly once on TEST. Writes the
evolved knobs to `config/policy_evolved.yaml`, the generation log to `notes/GENERATIONS.md`, and
all numbers to `artifacts/evolve_results.json`.
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import time

import numpy as np
import yaml

from feed import evolve
from feed.eval import paired_bootstrap
from feed.world import ROOT, load_config
from run_swarm import _clean, load_policy, seed_all

ART = ROOT / "artifacts"
EVOLVED_PATH = ROOT / "config" / "policy_evolved.yaml"
GENERATIONS_MD = ROOT.parent / "notes" / "GENERATIONS.md"
EVOLVE_HEADER = "## Evolution log (Phase 4)"


def evaluate_many(pool, tasks: list[dict]) -> list[dict]:
    return [evolve.evaluate_task(t) for t in tasks] if pool is None else pool.map(
        evolve.evaluate_task, tasks)


def run_stage(pool, cfg: dict, policy: dict, knobs_list: list[dict], stage: dict,
              threads: int) -> list[dict]:
    tasks = [evolve.build_task(cfg, policy, k, stage["seed"], stage["bots"], stage["days"], threads)
             for k in knobs_list]
    return evaluate_many(pool, tasks)


def search(pool, cfg, policy, spec, ev, threads, tol, penalty, budget, log):
    """(1+lambda)-ES on TRAIN: sample lambda offspring, keep the fittest (parent competes)."""
    rng = np.random.default_rng(int(cfg["seed"]) + 100)
    tr = ev["train"]
    start_knobs = evolve.vector_to_knobs(evolve.start_vector(spec), spec)
    base = run_stage(pool, cfg, policy, [start_knobs], tr, threads)[0]
    ref = base["guardrails"]
    parent_knobs, parent_vec = start_knobs, evolve.knobs_to_vector(start_knobs, spec)
    parent_fit, _ = evolve.fitness(base["iwe"], ref, ref, tol, penalty)
    sigma = float(ev["sigma_frac"])
    gens, pop = int(ev["generations"]), int(ev["population"])
    candidates = {tuple(sorted(start_knobs.items())): start_knobs}
    history, g, t0 = [], 0, time.time()
    while g < gens:
        children = [evolve.vector_to_knobs(evolve.mutate(parent_vec, spec, sigma, rng), spec)
                    for _ in range(pop)]
        tgen = time.time()
        results = run_stage(pool, cfg, policy, children, tr, threads)
        gen_secs = time.time() - tgen
        scored = []
        for k, r in zip(children, results):
            fit, nv = evolve.fitness(r["iwe"], ref, r["guardrails"], tol, penalty)
            scored.append((fit, nv, k, r))
        scored.sort(key=lambda t: t[0], reverse=True)
        best_fit, best_nv, best_k, best_r = scored[0]
        improved = best_fit > parent_fit
        if improved:
            parent_knobs, parent_vec = best_k, evolve.knobs_to_vector(best_k, spec)
            parent_fit = best_fit
        candidates.setdefault(tuple(sorted(best_k.items())), best_k)
        history.append({"generation": g + 1, "sigma": float(sigma),
                        "best_child_knobs": best_k, "best_child_iwe": best_r["iwe"],
                        "best_child_fitness": best_fit, "violations": int(best_nv),
                        "improved": bool(improved), "seconds": gen_secs})
        log(f"[evolve] gen {g + 1}/{gens} sigma={sigma:.3f} child_iwe={best_r['iwe']:.2f} "
            f"fit={best_fit:.2f} viol={best_nv} {'IMPROVE' if improved else 'hold'} "
            f"({gen_secs:.0f}s)")
        sigma *= float(ev["sigma_decay"])
        g += 1
        if g == 1 and budget > 0:
            projected = (time.time() - t0) * gens
            if projected > budget:
                feasible = max(1, int(budget / max(gen_secs, 1e-9)))
                if feasible < gens:
                    log(f"[evolve] projected {projected:.0f}s > budget {budget}s -> shrink to "
                        f"{feasible} generations")
                    gens = feasible
    candidates.setdefault(tuple(sorted(parent_knobs.items())), parent_knobs)
    return history, list(candidates.values()), parent_knobs, ref


def select_on_val(pool, cfg, policy, ev, candidates, base_knobs, threads, tol, penalty, log):
    va = ev["val"]
    results = run_stage(pool, cfg, policy, [base_knobs] + candidates, va, threads)
    ref = results[0]["guardrails"]
    rows, best = [], None
    for r in results[1:]:
        fit, nv = evolve.fitness(r["iwe"], ref, r["guardrails"], tol, penalty)
        rows.append({"knobs": r["knobs"], "iwe": r["iwe"], "fitness": fit, "violations": int(nv)})
        if best is None or fit > best[0]:
            best = (fit, r["knobs"], r)
    log(f"[evolve] VAL: {len(rows)} candidates, best iwe={best[2]['iwe']:.2f} fit={best[0]:.2f}")
    return {"fitness": best[0], "knobs": best[1], "iwe": best[2]["iwe"],
            "guardrails": best[2]["guardrails"]}, rows, ref


def write_evolved(policy: dict, knobs: dict) -> dict:
    eff = evolve.knobs_to_policy(policy, knobs)
    out = {b: eff[b] for b in ("composer", "score_weights", "retrieval")}
    with open(EVOLVED_PATH, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    return out


def update_generations_md(history: list[dict], selected: dict, val_iwe: float,
                          test: dict, log) -> None:
    if not GENERATIONS_MD.exists():
        log("[evolve] GENERATIONS.md missing; skipped log append")
        return
    text = GENERATIONS_MD.read_text()
    text = text.split("\n" + EVOLVE_HEADER)[0].rstrip() + "\n"
    lines = ["", EVOLVE_HEADER, "",
             "`run_evolve.py`: (1+lambda)-ES over composer + score weights + retrieval mix, fitness",
             "= IWE discounted 5% per guardrail violated >10% vs the search-start composer. Search on",
             "TRAIN (seed 1), select on VAL (seed 2), TEST once (seed 3).", "",
             "| gen | best child IWE | fitness | violations | vs parent |", "|---|---|---|---|---|"]
    for h in history:
        lines.append(f"| {h['generation']} | {h['best_child_iwe']:.2f} | {h['best_child_fitness']:.2f}"
                     f" | {h['violations']} | {'improved' if h['improved'] else 'held'} |")
    lines += ["", f"Selected knobs (VAL IWE {val_iwe:.2f}):", "", "```yaml",
              yaml.safe_dump(selected, sort_keys=False).strip(), "```", "",
              f"TEST (seed 3) IWE/bot: evolved {test['evolved_iwe']:.2f} vs "
              f"`taste_composer` {test['base_iwe']:.2f} "
              f"(delta {test['delta']:+.2f}, 95% CI "
              f"[{test['ci'][0]:+.2f}, {test['ci'][1]:+.2f}]).", ""]
    GENERATIONS_MD.write_text(text + "\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4: evolve policy knobs against the swarm")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--budget-sec", type=float, default=900.0)
    ap.add_argument("--tol", type=float, default=0.10)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    t_start = time.time()
    cfg = load_config()
    policy = load_policy()
    cfg["policy"] = policy
    spec = policy["evolve"]["knobs"]
    ev = copy.deepcopy(policy["evolve"])
    seed_all(int(args.seed if args.seed is not None else cfg["seed"]))
    threads = int(policy["python_threads"])
    penalty = float(ev["penalty_frac"])
    n_boot = args.n_boot or (200 if args.fast else int(cfg["metrics"]["bootstrap_resamples"]))

    if args.fast:
        for stage in ("train", "val", "test"):
            ev[stage] = {**ev[stage], "bots": 400, "days": 2}
        ev["population"], ev["generations"] = 4, 3
    if args.pop:
        ev["population"] = args.pop
    if args.gens:
        ev["generations"] = args.gens

    def log(*a):
        print(*a, flush=True)

    pool = None if args.workers <= 1 else mp.get_context("fork").Pool(args.workers)
    try:
        t0 = time.time()
        history, candidates, parent_knobs, ref = search(
            pool, cfg, policy, spec, ev, 1 if pool else threads, args.tol, penalty,
            args.budget_sec, log)
        t_search = time.time() - t0
        base_knobs = evolve.vector_to_knobs(evolve.start_vector(spec), spec)

        t0 = time.time()
        best, val_rows, _ = select_on_val(pool, cfg, policy, ev, candidates, base_knobs,
                                          1 if pool else threads, args.tol, penalty, log)
        t_val = time.time() - t0

        t0 = time.time()
        test = ev["test"]
        res = run_stage(pool, cfg, policy, [base_knobs, best["knobs"]], test,
                        1 if pool else threads)
        ci = paired_bootstrap(res[1]["iwe_per_bot"], res[0]["iwe_per_bot"], n_boot, test["seed"])
        t_test = time.time() - t0
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    write_evolved(policy, best["knobs"])
    test_summary = {"base_iwe": float(res[0]["iwe"]), "evolved_iwe": float(res[1]["iwe"]),
                    "delta": float(ci["mean_diff"]), "ci": [float(ci["lo"]), float(ci["hi"])],
                    "pass": bool(ci["lo"] > 0),
                    "base_guardrails": res[0]["guardrails"],
                    "evolved_guardrails": res[1]["guardrails"]}
    update_generations_md(history, best["knobs"], best["iwe"], test_summary, log)

    result = {
        "status": "ok", "seed": int(cfg["seed"]), "n_boot": n_boot, "tol": args.tol,
        "penalty_frac": penalty, "population": ev["population"], "generations": ev["generations"],
        "stages": {s: {k: ev[s][k] for k in ("bots", "days", "seed")} for s in ("train", "val", "test")},
        "search_history": history, "val_candidates": val_rows,
        "start_knobs": base_knobs, "selected_knobs": best["knobs"],
        "val": {"iwe": best["iwe"], "fitness": best["fitness"], "guardrails": best["guardrails"]},
        "test": test_summary,
        "evolved_policy": write_evolved(policy, best["knobs"]),
        "search_train_guardrails": ref,
        "timings_sec": {"search": t_search, "select": t_val, "test": t_test,
                        "total": time.time() - t_start},
    }
    ART.mkdir(exist_ok=True)
    with open(ART / "evolve_results.json", "w") as f:
        json.dump(_clean(result), f, indent=1)
    log(f"[evolve] selected -> config/policy_evolved.yaml | TEST evolved {res[1]['iwe']:.2f} vs "
        f"base {res[0]['iwe']:.2f} (d={ci['mean_diff']:+.2f} "
        f"[{ci['lo']:+.2f},{ci['hi']:+.2f}]) | total {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
