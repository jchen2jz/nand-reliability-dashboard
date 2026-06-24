# 3D NAND Flash Reliability and Endurance Model

A physics-based simulation of how 3D NAND threshold-voltage distributions wear
out under program/erase cycling, data retention, and read disturb, and what that
means for raw bit error rate (RBER), post-ECC failure (UBER), and usable
endurance across SLC, MLC, TLC, and QLC.

The project pairs a **closed-form analytical model** with a **Monte Carlo cell
population simulator**, and uses **importance sampling** to estimate the
ultra-rare tail probabilities that set real reliability targets.

## Why each piece is here

| Component | What it shows |
|---|---|
| Analytical RBER/UBER model | Device-physics understanding: Vt margin, Gray coding, ECC limits |
| Monte Carlo population sim | Simulation + data analysis; validates the closed form to ~1% |
| Process variation (weak cells) | Why tails, not averages, drive real failure |
| Importance sampling | Estimating ~1e-15 events that naive MC cannot reach |
| Adaptive read references | Read-retry recovers endurance lost to a fixed factory reference |
| Temperature / retention sweep | The reliability tradeoffs an advanced-memory team designs around |

## Files

- `nand_model.py` — physics, distributions, analytical RBER, UBER, endurance
- `monte_carlo.py` — population simulator + naive vs importance-sampling tail estimators
- `run_analysis.py` — generates all figures and a summary
- `figures/` — output plots

## Run

```bash
pip install numpy scipy matplotlib
python run_analysis.py
```

Runs end to end in well under a minute and writes seven figures plus a printed
summary. Works as-is in Google Colab (drop the three `.py` files in and run).

## Headline results (1 year @ 85 C, 1e6 reads)

- Endurance ordering SLC > MLC > TLC > QLC, spanning ~17k down to ~600 P/E cycles
- Monte Carlo with no process variation matches the analytical RBER to ~1 percent
- Process variation roughly doubles RBER, shifting end of life earlier
- Importance sampling recovers a 1.3e-12 tail probability that naive Monte Carlo
  reads as exactly zero
- Adaptive read references extend TLC endurance ~50 percent versus a fixed
  factory reference and recover a QLC corner that otherwise fails outright

## How to talk about it (interview)

- The central tradeoff is **Vt margin versus bits per cell**. Adding a bit halves
  the spacing between states for the same window, so the same wear closes the
  read margins sooner. This is the logical-scaling vector of NAND roadmaps.
- **RBER comes from distribution overlap**; ECC corrects up to a strength limit;
  **UBER crossing a target defines end of life**. Endurance is where those meet.
- The **Monte Carlo step is validation plus discovery**: it confirms the analytical
  model, then process variation reveals the closed form is optimistic because it
  ignores weak cells in the tail.
- **Importance sampling** is the answer to "you can't simulate a 1e-15 event with
  1e9 samples": shift the sampling distribution onto the decision boundary and
  reweight by the likelihood ratio.

## Ready extensions

- Layer-to-layer (wordline-position) variation for the 3D stack (BiCS-style)
- Device-to-drive bridge: per-cell endurance to drive TBW via write amplification
- Calibrate coefficients against published 3D NAND RBER-vs-cycle data
- LDPC vs BCH code-rate trade study (capacity overhead vs correction strength)

## Resume mapping

> **NAND Flash Reliability and Endurance Model** — Python, NumPy, SciPy, Matplotlib
> - Built a physics-based 3D NAND model of threshold-voltage distributions across
>   SLC through QLC under program/erase cycling, retention, and read disturb
> - Quantified RBER from Vt-margin loss and derived UBER and usable endurance via
>   an ECC model, validated with a Monte Carlo population simulator to ~1 percent
> - Estimated UBER-class rare events with importance sampling, reaching tail
>   probabilities below 1e-12 that direct Monte Carlo cannot resolve
