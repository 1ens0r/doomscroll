"""Intuition: The five headline figures of the whole build — the policy ladder, the composer
generation log, the calibration verdict, the cohort headroom heatmap and the return curves —
rendered straight from the committed artifacts so every claim in the READMEs traces to a PNG.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
FIG_DIR = ROOT.parent / "reports" / "figures"
SWARM_PATH = ROOT / "config" / "swarm.yaml"

LADDER_ORDER = ["random", "popularity", "personal_affinity", "taste", "taste_composer",
                "evolved", "oracle"]
# CI on the gap scale: accumulate adjacent paired-bootstrap diff CIs in quadrature.
LADDER_CHAIN = ["popularity_gt_random", "personal_affinity_gt_popularity",
                "taste_gt_personal_affinity", "taste_composer_gt_taste",
                "evolved_gt_taste_composer", "oracle_gt_taste"]
LADDER_LABELS = ["random", "popularity", "personal\naffinity", "taste",
                 "taste_\ncomposer", "evolved", "oracle"]


def _fig_cfg() -> tuple[list[str], int]:
    with open(SWARM_PATH) as f:
        fg = yaml.safe_load(f)["figures"]
    return list(fg["palette"]), int(fg["dpi"])


def _art(name: str) -> dict:
    with open(ART / name) as f:
        return json.load(f)


def _style(ax, axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def _save(fig, name: str, dpi: int) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    return out


def _gap_cis() -> dict[str, float]:
    """Per-policy CI half-width on the % gap-closed scale, complemented from adjacent diffs."""
    sim, out, acc = _art("sim_results.json"), {}, 0.0
    span = sim["iwe"]["oracle"]["mean"] - sim["iwe"]["random"]["mean"]
    for key in LADDER_CHAIN:
        ci = sim["ladder"].get(key) or sim["extra_pairs"][key]
        acc += ((ci["hi"] - ci["lo"]) / 2.0) ** 2
        policy = key.split("_gt_")[0]
        out[policy] = float(np.sqrt(acc) / span)
    return out


def ladder(policy_order: list[str] = LADDER_ORDER, out: str = "01_ladder.png") -> Path:
    """THE LADDER: % of the personalized-oracle gap closed, CIs from paired per-bot bootstraps."""
    import matplotlib.pyplot as plt

    palette, dpi = _fig_cfg()
    sim, hw = _art("sim_results.json"), _gap_cis()
    base = sim["iwe"]["random"]["mean"]
    span = sim["iwe"]["oracle"]["mean"] - base
    gaps = np.array([sim["oracle_gap_closed"][p] for p in policy_order])
    ci = np.array([0.0] + [hw[p] for p in policy_order[1:-1]] + [0.0])
    iwe = np.array([sim["iwe"][p]["mean"] for p in policy_order])

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    x = np.arange(len(policy_order))
    colors = [palette[i % len(palette)] for i in range(len(policy_order))]
    colors[-1], colors[0] = "#8C8C8C", "#BBBBBB"
    bars = ax.bar(x, 100 * gaps, width=0.68, color=colors, edgecolor="white", lw=0.5)
    err = 100 * ci
    ax.errorbar(x[1:-1], 100 * gaps[1:-1], yerr=err[1:-1], fmt="none", ecolor="#333333",
                lw=1.2, capsize=3.5, zorder=10)
    for xi, bar, g, v in zip(x, bars, 100 * gaps, iwe):
        ax.text(xi, bar.get_height() + 2.6, f"{g:.1f}%", ha="center", fontsize=9.5, weight="bold")
        if g < 0.02:  # zero-height bar: label above the gap text instead of colliding on the axis
            ax.text(xi, bar.get_height() + 9.2, f"{v:.1f}", ha="center",
                    fontsize=8.4, color="#444444", style="italic")
        else:
            ax.text(xi, bar.get_height() / 2, f"{v:.1f}", ha="center", va="center",
                    fontsize=8.4, color="white" if g > 0.12 else "#444444", style="italic")
    ax.set_xticks(x, LADDER_LABELS, fontsize=8.6)
    ax.set_ylabel("% of oracle gap closed", fontsize=10)
    ax.set_ylim(0, 116)
    ax.set_title("The ladder — 10,000 bots × 7 days, seed 42, calibrated, paired bootstrap"
                 "\n(bold = % oracle gap closed · italic = IWE/bot)", fontsize=10.5, pad=10)
    _style(ax)
    ax.text(0.99, 0.02, "every adjacent rung's 95% CI excludes zero · oracle = greedy ceiling",
            transform=ax.transAxes, ha="right", fontsize=7.4, color="#666666",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.0))
    fig.tight_layout()
    return _save(fig, out, dpi)


def generations(out: str = "02_generations.png") -> Path:
    """Composer generations: per-component ΔIWE with CIs, and the accepted-IWE trajectory."""
    palette, dpi = _fig_cfg()
    g = _art("composer_generations.json")
    gens = g["generations"]
    names = {"gamma": "seen-penalty\nγ", "alpha": "diversity\nα", "eta": "session\nadaptation η",
             "beta": "cold\nbonus β", "epsilon": "epsilon\nexploration"}
    n = len(gens)
    delta = np.array([e["delta_vs_accepted"] for e in gens])
    lo = delta - np.array([e["ci"][0] for e in gens])
    hi = np.array([e["ci"][1] for e in gens]) - delta
    x = np.arange(n)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    keep_color, reject_color = palette[2 % len(palette)], palette[3 % len(palette)]
    bars = ax.bar(x, delta, width=0.62, color=[keep_color if e["kept"] else reject_color
                                               for e in gens], edgecolor="white", lw=0.5)
    ax.errorbar(x, delta, yerr=[lo, hi], fmt="none", ecolor="#333333", lw=1.2, capsize=3.5)
    for xi, e in zip(x, gens):
        ax.text(xi, delta[xi] + hi[xi] + 0.6, "KEEP" if e["kept"] else "REJECT", ha="center",
                fontsize=8.6, weight="bold", color=keep_color if e["kept"] else reject_color)
    # Accepted-IWE trajectory: flat while a generation is rejected.
    traj, cur, iwe_vals = [], g["iwe_neutral"], []
    for e in gens:
        cur = e["iwe"] if e["kept"] else cur
        traj.append(cur)
    for xi, e in zip(x, gens):  # rejected components show their would-be level, open marker
        iwe_vals.append(e["iwe"])
    ax2 = ax.twinx()
    ax2.plot(x, traj, color=palette[0 % len(palette)], lw=1.8, marker="o", ms=5, zorder=5,
             label="accepted IWE/bot")
    ax2.plot(x, iwe_vals, color=palette[0 % len(palette)], lw=0.9, ls=":", marker="o", ms=3.5,
             mfc="white", zorder=4, label="candidate IWE/bot")
    ax2.set_ylabel("IWE / bot", fontsize=9.5)
    ax2.set_ylim(115, 145)
    ax2.spines[["top"]].set_visible(False)
    ax2.legend(fontsize=8, loc="upper right", frameon=False)
    ax.set_xticks(x, [names[e["knob"]] for e in gens], fontsize=8.6)
    ax.set_ylabel("ΔIWE vs accepted so far  (95% CI)", fontsize=9.5)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_title("Composer generations — keep only what pays (α +4.45 · η +8.98 kept;"
                 " γ, β, ε rejected)", fontsize=10.5, pad=10)
    _style(ax)
    fig.tight_layout()
    return _save(fig, out, dpi)


def calibration(out: str = "03_calibration.png") -> Path:
    """Sim vs real marginals: relative error per target with the 15% acceptance band."""
    palette, dpi = _fig_cfg()
    rep = _art("calibration_report.json")["checks"]
    items = sorted(rep.items(), key=lambda kv: -float(np.clip(abs(kv[1]["rel_err"]), 0, 1.0)))
    labels = [k for k, _ in items]
    rel = np.array([c["rel_err"] for _, c in items], float)
    passed = [c["pass"] for _, c in items]
    y = np.arange(len(items))[::-1]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ok, bad = palette[2 % len(palette)], palette[3 % len(palette)]
    ax.barh(y, rel, height=0.62, color=[ok if p else bad for p in passed],
            edgecolor="white", lw=0.5)
    ax.axvline(0.15, color="#444444", lw=1.1, ls="--")
    ax.set_ylim(-0.6, 12.4)
    ax.text(0.155, y[0] + 0.75, "±15% tolerance", fontsize=8, color="#444444")
    for yi, (lab, c) in zip(y, items):
        ax.text(c["rel_err"] + 0.012, yi, f"sim {c['sim']:.3f} · real {c['target']:.3f}",
                va="center", fontsize=7.8, color="#444444")
    ax.set_yticks(y, labels, fontsize=9)
    ax.set_xlabel("|relative error|  (calibrated sim vs real marginals)", fontsize=9.5)
    ax.set_xlim(0, 1.02)
    ax.set_title("Calibration — 7 / 12 targets inside 15%; the 5 misses are structural,"
                 " not tuning failures", fontsize=10.5, pad=10)
    _style(ax, axis="x")
    fig.tight_layout()
    return _save(fig, out, dpi)


def cohorts(policy_order: list[str] = LADDER_ORDER, out: str = "04_cohorts.png") -> Path:
    """Per-cohort oracle-gap-closed heatmap: personalization headroom concentrates in heavy users."""
    palette, dpi = _fig_cfg()
    sim = _art("sim_results.json")["cohorts"]
    cells = [f"A{a}_B{b}" for a in range(3) for b in range(3)]
    val = np.zeros((len(cells), len(policy_order)))
    for j, p in enumerate(policy_order):
        for i, c in enumerate(cells):
            lo, hi = sim["random"][c], sim["oracle"][c]
            val[i, j] = (sim[p][c] - lo) / (hi - lo)

    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    cmap = LinearSegmentedColormap.from_list("doom", ["#F7F7F7", palette[0 % len(palette)]])
    im = ax.imshow(100 * val, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(policy_order)), policy_order, fontsize=8.8,
                  rotation=20, ha="right")
    rows = ["A0_B0 · light / narrow", "A0_B1 · light / mid", "A0_B2 · light / diverse",
            "A1_B0 · mid / narrow", "A1_B1 · mid / mid", "A1_B2 · mid / diverse",
            "A2_B0 · heavy / narrow", "A2_B1 · heavy / mid", "A2_B2 · heavy / diverse"]
    ax.set_yticks(np.arange(len(cells)), rows, fontsize=7.8)
    for i in range(len(cells)):
        for j in range(len(policy_order)):
            v = 100 * val[i, j]
            ax.text(j, i, f"{v:.0f}" if abs(v) >= 0.5 else "0", ha="center", va="center",
                    fontsize=7.8, color="white" if v > 55 else "#333333")
    ax.set_title("Where the gap closes — IWE gap-closed % by cohort\n"
                 "(A = activity tercile, B = taste-breadth tercile; column-normalised per cell)",
                 fontsize=10.5, pad=10)
    fig.colorbar(im, ax=ax, shrink=0.85, label="% oracle gap closed")
    fig.tight_layout()
    return _save(fig, out, dpi)


def survival(policy_order: list[str] = LADDER_ORDER, out: str = "05_retention.png") -> Path:
    """Return curves: share of bots still active each simulated day, policy overlay."""
    palette, dpi = _fig_cfg()
    diag = _art("sim_results.json")["diagnostics"]
    days = len(diag["random"]["daily_active"])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for i, p in enumerate(policy_order):
        ys = np.array(diag[p]["daily_active"], float)
        color = "#8C8C8C" if p == "oracle" else palette[i % len(palette)]
        lw, alpha = (2.6, 1.0) if p == "evolved" else (1.8, 0.95)
        ls = "--" if p == "oracle" else "-"
        ax.plot(np.arange(days), 100 * ys, color=color, lw=lw, alpha=alpha, ls=ls,
                marker="o", ms=3.4, label=p)
    # Right-edge labels, de-collided: walk ascending and push any label within 1.5pp of the last.
    ends = [100 * float(diag[p]["daily_active"][-1]) for p in policy_order]
    y_lab, last = {}, None
    for k in sorted(range(len(ends)), key=lambda i: ends[i]):
        y = ends[k] if last is None or ends[k] - last >= 1.5 else last + 1.5
        y_lab[k], last = y, y
    for i, p in enumerate(policy_order):
        color = "#8C8C8C" if p == "oracle" else palette[i % len(palette)]
        ax.text(days - 1 + 0.12, y_lab[i], f"{ends[i]:.1f}%", fontsize=7.8, va="center",
                color=color, weight="bold" if p == "evolved" else "normal")
    ax.set_xticks(np.arange(days), [f"D{d}" for d in range(days)])
    ax.set_xlabel("simulated day", fontsize=9.5)
    ax.set_ylabel("% of bots active", fontsize=9.5)
    ax.set_xlim(-0.3, days - 0.25)
    ax.set_ylim(38, 104)
    ax.set_title("Return curves — engagement is bought back daily (D7: random 45.9 → evolved"
                 " 56.9 → oracle 67.1%)", fontsize=10.5, pad=10)
    ax.legend(fontsize=7.8, ncol=2, loc="lower left", frameon=False)
    _style(ax)
    fig.tight_layout()
    return _save(fig, out, dpi)


def main() -> None:
    names = {"ladder": ladder, "generations": generations, "calibration": calibration,
             "cohorts": cohorts, "survival": survival}
    for name, fn in names.items():
        print(f"[plot] {name:<12} -> {_fig_dir_rel(fn())}")
    print(f"[plot] {len(names)} figures")


def _fig_dir_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT.parent))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
