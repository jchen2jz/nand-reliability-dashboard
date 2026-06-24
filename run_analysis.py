"""
run_analysis.py
===============
Generates the figure suite and a numerical summary for the NAND reliability
study. Run from this directory:  python run_analysis.py

Figures written to ./figures/:
  1_vt_distributions.png      worn vs fresh Vt distributions (TLC)
  2_rber_vs_cycles.png        RBER endurance curves, SLC->QLC
  3_uber_endurance.png        UBER vs cycles + endurance-by-cell-type bars
  4_mc_validation.png         Monte Carlo vs analytical, with process variation
  5_importance_sampling.png   rare-event tail estimation, naive MC vs IS
  6_adaptive_ref.png          adaptive (read-retry) vs fixed read references
  7_lifetime_heatmap.png      endurance vs temperature and retention time
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

from nand_model import (NANDConfig, level_means, base_sigmas,
                        degraded_distributions, midpoint_refs, optimal_refs,
                        analytical_rber, uncorrectable_prob, endurance_cycles,
                        rber_curve)
from monte_carlo import (simulate_population, crossing_prob_naive,
                         crossing_prob_is)

FIG = "figures"
os.makedirs(FIG, exist_ok=True)
NAMES = {1: "SLC", 2: "MLC", 3: "TLC", 4: "QLC"}
COLORS = {1: "#2E86AB", 2: "#1B998B", 3: "#E07A5F", 4: "#8E44AD"}
plt.rcParams.update({"figure.dpi": 120, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

# Reference stress scenario
RET_H, TEMP_C, READS = 8760.0, 85.0, 1.0e6   # 1 year, 85 C, 1e6 reads


def _gauss(x, m, s):
    return norm.pdf(x, m, s)


# --------------------------------------------------------------------------- #
def fig1_distributions():
    cfg = NANDConfig(bits_per_cell=3)
    x = np.linspace(cfg.v_low, cfg.v_high, 2000)
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    for row, (cyc, lab) in enumerate([(0, "Fresh (0 cycles)"),
                                      (3500, "Worn (3500 cycles, 1 yr @ 85 C)")]):
        m, s = degraded_distributions(cfg, cyc, RET_H if cyc else 0, TEMP_C, READS if cyc else 0)
        refs = midpoint_refs(m)
        for k in range(cfg.L):
            ax[row].plot(x, _gauss(x, m[k], s[k]), color="#333", lw=1.4)
            ax[row].fill_between(x, _gauss(x, m[k], s[k]), alpha=0.12, color=COLORS[3])
        for r in refs:
            ax[row].axvline(r, color="#999", ls="--", lw=0.8)
        ax[row].set_title(lab, loc="left", fontsize=11, fontweight="bold")
        ax[row].set_ylabel("density")
        ax[row].set_yticks([])
    ax[1].set_xlabel("threshold voltage  Vt  (normalized)")
    fig.suptitle("TLC threshold-voltage distributions: wear closes the read margins",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG}/1_vt_distributions.png", bbox_inches="tight")
    plt.close(fig)


def fig2_rber():
    cyc = np.arange(0, 20001, 100)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for n in [1, 2, 3, 4]:
        cfg = NANDConfig(bits_per_cell=n)
        r = rber_curve(cfg, cyc, RET_H, TEMP_C, READS)
        r = np.clip(r, 1e-18, None)
        ax.semilogy(cyc, r, color=COLORS[n], lw=2, label=NAMES[n])
    # ECC operating point: RBER that the codeword can typically still correct
    cfg = NANDConfig()
    # find RBER where uncorrectable prob hits the target (the usable RBER ceiling)
    rr = np.logspace(-6, -1.3, 400)
    ceil = rr[np.argmax([uncorrectable_prob(x, cfg) > cfg.uber_target for x in rr])]
    ax.axhline(ceil, color="k", ls=":", lw=1.3)
    ax.text(200, ceil * 1.3, f"ECC ceiling (UBER = {cfg.uber_target:.0e})", fontsize=9)
    ax.set_xlabel("program / erase cycles")
    ax.set_ylabel("raw bit error rate (RBER)")
    ax.set_ylim(1e-12, 1)
    ax.set_title("Endurance: more bits per cell means a tighter Vt budget and shorter life",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(title="cell type", frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/2_rber_vs_cycles.png", bbox_inches="tight")
    plt.close(fig)


def fig3_uber():
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    lives = {}
    for n in [2, 3, 4]:
        cfg = NANDConfig(bits_per_cell=n)
        cyc = np.arange(0, 20001, 100)
        r = rber_curve(cfg, cyc, RET_H, TEMP_C, READS)
        u = np.clip([uncorrectable_prob(x, cfg) for x in r], 1e-40, None)
        ax[0].semilogy(cyc, u, color=COLORS[n], lw=2, label=NAMES[n])
        lives[n] = endurance_cycles(cfg, RET_H, TEMP_C, READS)
    ax[0].axhline(NANDConfig().uber_target, color="k", ls=":", lw=1.3)
    ax[0].text(200, NANDConfig().uber_target * 2, "end-of-life target", fontsize=9)
    ax[0].set_xlabel("program / erase cycles")
    ax[0].set_ylabel("uncorrectable bit error rate (UBER)")
    ax[0].set_ylim(1e-30, 1)
    ax[0].set_title("Post-ECC failure vs cycling", loc="left", fontweight="bold")
    ax[0].legend(frameon=False)

    for n in [1, 2, 3, 4]:
        if n not in lives:
            lives[n] = endurance_cycles(NANDConfig(bits_per_cell=n), RET_H, TEMP_C, READS)
    order = [1, 2, 3, 4]
    ax[1].bar([NAMES[n] for n in order], [lives[n] for n in order],
              color=[COLORS[n] for n in order])
    for i, n in enumerate(order):
        ax[1].text(i, lives[n], f"{lives[n]:,}", ha="center", va="bottom", fontsize=9)
    ax[1].set_ylabel("endurance (P/E cycles to EOL)")
    ax[1].set_title("Usable endurance by cell type", loc="left", fontweight="bold")
    fig.suptitle("Density vs endurance: the core advanced-memory roadmap tradeoff",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG}/3_uber_endurance.png", bbox_inches="tight")
    plt.close(fig)
    return lives


def fig4_mc_validation():
    cfg = NANDConfig(bits_per_cell=3)
    cyc = np.arange(500, 4001, 500)
    ana, mc0, mcv, lo, hi = [], [], [], [], []
    for N in cyc:
        ana.append(rber_curve(cfg, np.array([N]), RET_H, TEMP_C, READS)[0])
        r0, _ = simulate_population(cfg, N, RET_H, TEMP_C, READS, n_cells=4_000_000, proc_var=0.0)
        rv, ci = simulate_population(cfg, N, RET_H, TEMP_C, READS, n_cells=4_000_000, proc_var=0.05)
        mc0.append(r0); mcv.append(rv); lo.append(ci[0]); hi.append(ci[1])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(cyc, ana, "k-", lw=2, label="analytical (closed form)")
    ax.semilogy(cyc, mc0, "o", color="#2E86AB", ms=7, label="Monte Carlo (no variation)")
    ax.errorbar(cyc, mcv, yerr=[np.array(mcv) - lo, np.array(hi) - mcv],
                fmt="s", color="#E07A5F", ms=6, capsize=3,
                label="Monte Carlo (with process variation)")
    ax.set_xlabel("program / erase cycles")
    ax.set_ylabel("RBER (TLC)")
    ax.set_title("Monte Carlo validates the model, and exposes what it misses",
                 loc="left", fontweight="bold", fontsize=12)
    ax.legend(frameon=False)
    ax.text(0.02, 0.04,
            "MC with no variation lands on the analytical curve.\n"
            "Adding cell-to-cell process variation lifts RBER:\n"
            "weak cells in the tail drive real-world failure.",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#ccc"))
    fig.tight_layout()
    fig.savefig(f"{FIG}/4_mc_validation.png", bbox_inches="tight")
    plt.close(fig)


def fig5_importance_sampling():
    Rs = np.arange(2.5, 7.6, 0.5)
    truth = norm.sf(Rs)
    naive = [crossing_prob_naive(0, 1, R, 2_000_000) for R in Rs]
    iss, ise = zip(*[crossing_prob_is(0, 1, R, 200_000) for R in Rs])
    iss = np.array(iss, float)
    naive = np.array(naive, float)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(Rs, truth, "k-", lw=2, label="analytical truth")
    ax.semilogy(Rs, np.where(iss > 0, iss, np.nan), "s", color="#1B998B", ms=7,
                label="importance sampling (200k samples)")
    nv = np.where(naive > 0, naive, np.nan)
    ax.semilogy(Rs, nv, "o", color="#E07A5F", ms=7, label="naive Monte Carlo (2M samples)")
    floor = 1.0 / 2_000_000
    ax.axhspan(1e-16, floor, color="#E07A5F", alpha=0.08)
    ax.axhline(floor, color="#E07A5F", ls="--", lw=1)
    ax.text(2.6, floor * 1.4, "naive MC resolution floor (1 / N)", fontsize=9, color="#b5503a")
    ax.set_xlabel("boundary distance from state mean (sigmas)")
    ax.set_ylabel("tail / crossing probability")
    ax.set_ylim(1e-15, 1e-1)
    ax.set_title("Rare-event estimation: importance sampling reaches UBER-class probabilities",
                 loc="left", fontweight="bold", fontsize=11.5)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{FIG}/5_importance_sampling.png", bbox_inches="tight")
    plt.close(fig)


def fig6_adaptive_ref():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cyc = np.arange(0, 8001, 100)
    res = {}
    for n in [3, 4]:
        cfg = NANDConfig(bits_per_cell=n)
        rf = rber_curve(cfg, cyc, RET_H, TEMP_C, READS, ref_mode="factory")
        ro = rber_curve(cfg, cyc, RET_H, TEMP_C, READS, ref_mode="optimal")
        ax.semilogy(cyc, np.clip(rf, 1e-18, None), color=COLORS[n], lw=2, ls="--",
                    label=f"{NAMES[n]} factory reference (fixed)")
        ax.semilogy(cyc, np.clip(ro, 1e-18, None), color=COLORS[n], lw=2,
                    label=f"{NAMES[n]} adaptive (read-retry)")
        lf = endurance_cycles(cfg, RET_H, TEMP_C, READS, ref_mode="factory")
        lo = endurance_cycles(cfg, RET_H, TEMP_C, READS, ref_mode="optimal")
        res[n] = (lf, lo)
    ax.set_xlabel("program / erase cycles")
    ax.set_ylabel("RBER")
    ax.set_ylim(1e-9, 1)
    ax.set_title("Adaptive read references (read-retry) recover endurance",
                 loc="left", fontweight="bold", fontsize=12)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    txt = "\n".join(
        f"{NAMES[n]}: {lf:,} -> {lo:,} cycles"
        + (f"  (+{(lo/lf-1)*100:.0f}%)" if lf > 0 else "  (recovers a failed corner)")
        for n, (lf, lo) in res.items())
    ax.text(0.02, 0.04, "endurance gain\n" + txt, transform=ax.transAxes,
            fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#ccc"))
    fig.tight_layout()
    fig.savefig(f"{FIG}/6_adaptive_ref.png", bbox_inches="tight")
    plt.close(fig)
    return res


def fig7_heatmap():
    cfg = NANDConfig(bits_per_cell=3)
    temps = np.array([25, 40, 55, 70, 85, 100])
    rets = np.array([24, 168, 720, 2160, 8760, 26280])  # 1d,1w,1mo,3mo,1yr,3yr
    Z = np.zeros((len(temps), len(rets)))
    for i, T in enumerate(temps):
        for j, R in enumerate(rets):
            Z[i, j] = endurance_cycles(cfg, R, T, READS, cyc_grid=np.arange(0, 20001, 250))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(rets)))
    ax.set_xticklabels(["1 day", "1 wk", "1 mo", "3 mo", "1 yr", "3 yr"])
    ax.set_yticks(range(len(temps)))
    ax.set_yticklabels([f"{t} C" for t in temps])
    ax.set_xlabel("required data-retention time")
    ax.set_ylabel("operating temperature")
    for i in range(len(temps)):
        for j in range(len(rets)):
            ax.text(j, i, f"{int(Z[i,j]/1000)}k" if Z[i, j] >= 1000 else f"{int(Z[i,j])}",
                    ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, label="TLC endurance (P/E cycles to EOL)")
    ax.set_title("Endurance collapses with retention demand and temperature",
                 loc="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIG}/7_lifetime_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def summary(lives, adaptive):
    print("\n" + "=" * 60)
    print("NAND RELIABILITY MODEL  —  SUMMARY  (1 yr @ 85 C, 1e6 reads)")
    print("=" * 60)
    print(f"{'cell':<6}{'bits':<6}{'endurance (P/E)':<18}")
    for n in [1, 2, 3, 4]:
        print(f"{NAMES[n]:<6}{n:<6}{lives[n]:<18,}")
    print("\nAdaptive read-reference (read-retry) endurance gain:")
    for n, (lf, lo) in adaptive.items():
        gain = f"+{(lo/lf-1)*100:.0f}%" if lf > 0 else "recovers a failed corner"
        print(f"  {NAMES[n]}: {lf:,} -> {lo:,} cycles  ({gain})")
    print("\nRare-event check  P(N(0,1) > 7):")
    truth = norm.sf(7)
    nv = crossing_prob_naive(0, 1, 7, 2_000_000)
    est, se = crossing_prob_is(0, 1, 7, 200_000)
    print(f"  analytical = {truth:.2e}")
    print(f"  naive MC   = {nv:.2e}   (resolution floor 5e-7)")
    print(f"  importance = {est:.2e} +/- {se:.1e}")
    print("=" * 60)


if __name__ == "__main__":
    print("Generating figures ...")
    fig1_distributions(); print("  [1/7] Vt distributions")
    fig2_rber();          print("  [2/7] RBER endurance curves")
    lives = fig3_uber();  print("  [3/7] UBER + endurance bars")
    fig4_mc_validation(); print("  [4/7] Monte Carlo validation")
    fig5_importance_sampling(); print("  [5/7] importance sampling")
    adaptive = fig6_adaptive_ref(); print("  [6/7] adaptive references")
    fig7_heatmap();       print("  [7/7] lifetime heatmap")
    summary(lives, adaptive)
