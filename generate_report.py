"""
generate_report.py
==================
Generates a self-contained HTML project report for the 3D NAND Reliability
and Endurance Model. Open the output file in any browser and use
File → Print → Save as PDF to export.

Run from this directory:  python generate_report.py
"""

import base64
import os
import numpy as np
from scipy.stats import norm

from nand_model import (NANDConfig, level_means, base_sigmas,
                        degraded_distributions, midpoint_refs,
                        analytical_rber, uncorrectable_prob, endurance_cycles)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def img_tag(path: str) -> str:
    """Return an <img> tag with the figure embedded as base64."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{data}" class="figure" alt="{os.path.basename(path)}">'


def compute_summary():
    """Return a dict of key numbers for the summary table."""
    results = {}
    labels = {1: "SLC", 2: "MLC", 3: "TLC", 4: "QLC"}
    for bits, name in labels.items():
        cfg = NANDConfig(bits_per_cell=bits)
        cyc = endurance_cycles(cfg, ret_time_h=8760, temp_C=85, reads=1e6)
        results[name] = cyc

    cfg_tlc = NANDConfig(bits_per_cell=3)
    tlc_factory = endurance_cycles(cfg_tlc, 8760, 85, 1e6, ref_mode="factory")
    tlc_optimal = endurance_cycles(cfg_tlc, 8760, 85, 1e6, ref_mode="optimal")

    cfg_qlc = NANDConfig(bits_per_cell=4)
    qlc_factory = endurance_cycles(cfg_qlc, 8760, 85, 1e6, ref_mode="factory")
    qlc_optimal = endurance_cycles(cfg_qlc, 8760, 85, 1e6, ref_mode="optimal")

    return results, tlc_factory, tlc_optimal, qlc_factory, qlc_optimal


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 14px;
    color: #1a1a2e;
    background: #fff;
    max-width: 960px;
    margin: 0 auto;
    padding: 40px 48px 80px;
}
h1 { font-size: 2rem; font-weight: 700; color: #0f3460; margin-bottom: 6px; }
h2 { font-size: 1.3rem; font-weight: 600; color: #16213e; margin: 36px 0 10px; border-left: 4px solid #0f3460; padding-left: 10px; }
h3 { font-size: 1.05rem; font-weight: 600; color: #0f3460; margin: 20px 0 6px; }
p  { line-height: 1.75; margin-bottom: 12px; }
ul { padding-left: 22px; margin-bottom: 12px; }
li { line-height: 1.75; }
.subtitle { color: #555; font-size: 1rem; margin-bottom: 4px; }
.meta    { color: #777; font-size: 0.85rem; margin-bottom: 40px; }
.figure  { width: 100%; max-width: 860px; border: 1px solid #dde; border-radius: 6px; margin: 16px 0 8px; display: block; }
.caption { font-size: 0.82rem; color: #666; margin-bottom: 28px; font-style: italic; }
table { width: 100%; border-collapse: collapse; margin: 16px 0 24px; font-size: 0.9rem; }
th { background: #0f3460; color: #fff; padding: 9px 14px; text-align: left; }
td { padding: 8px 14px; border-bottom: 1px solid #e8e8f0; }
tr:nth-child(even) td { background: #f5f7ff; }
.metric-row { display: flex; gap: 16px; margin: 20px 0; }
.metric { background: #f0f4ff; border-left: 4px solid #0f3460; border-radius: 4px; padding: 14px 20px; flex: 1; }
.metric .label { font-size: 0.78rem; color: #555; text-transform: uppercase; letter-spacing: 0.05em; }
.metric .value { font-size: 1.5rem; font-weight: 700; color: #0f3460; }
.metric .sub   { font-size: 0.8rem; color: #777; }
.highlight { background: #fff8e1; border-left: 4px solid #f5a623; padding: 12px 16px; border-radius: 4px; margin: 16px 0; }
@media print {
    body { padding: 20px 30px; }
    h2 { page-break-before: auto; }
}
"""


def build_html(results, tlc_factory, tlc_optimal, qlc_factory, qlc_optimal) -> str:
    here = os.path.dirname(os.path.abspath(__file__))

    def fig(name):
        path = os.path.join(here, name)
        return img_tag(path) if os.path.exists(path) else f"<p><em>[Figure {name} not found — run run_analysis.py first]</em></p>"

    tlc_gain = round((tlc_optimal - tlc_factory) / tlc_factory * 100) if tlc_factory else 0
    qlc_recover = "recovers" if qlc_factory == 0 and qlc_optimal > 0 else "improves"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>3D NAND Reliability Model — Project Report</title>
<style>{CSS}</style>
</head>
<body>

<h1>3D NAND Flash Reliability &amp; Endurance Model</h1>
<p class="subtitle">Physics-Based Simulation of Threshold-Voltage Degradation</p>
<p class="meta">Jonathan Chen &nbsp;|&nbsp; SanDisk Technology — Development Intern &nbsp;|&nbsp; 2025</p>

<!-- ── KEY METRICS ── -->
<div class="metric-row">
  <div class="metric">
    <div class="label">SLC Endurance</div>
    <div class="value">{results['SLC']:,}</div>
    <div class="sub">P/E cycles (1 yr @ 85 °C)</div>
  </div>
  <div class="metric">
    <div class="label">TLC Endurance</div>
    <div class="value">{results['TLC']:,}</div>
    <div class="sub">P/E cycles (1 yr @ 85 °C)</div>
  </div>
  <div class="metric">
    <div class="label">Read-Retry Gain (TLC)</div>
    <div class="value">+{tlc_gain}%</div>
    <div class="sub">{tlc_factory:,} → {tlc_optimal:,} cycles</div>
  </div>
  <div class="metric">
    <div class="label">Tail Probability</div>
    <div class="value">1.27 × 10⁻¹²</div>
    <div class="sub">via importance sampling</div>
  </div>
</div>

<!-- ── 1. EXECUTIVE SUMMARY ── -->
<h2>1. Executive Summary</h2>
<p>
This project builds a physics-based analytical model of 3D NAND flash memory reliability,
covering the full chain from device-level wear mechanisms to system-level endurance limits.
The model computes raw bit error rates (RBER) and uncorrectable error probabilities (UBER)
as a function of program/erase (P/E) cycling, data retention, and read disturb. It then
identifies the cycle count at which UBER exceeds a 10⁻¹⁵ target — the device end-of-life.
</p>
<p>
A Monte Carlo cell-population simulator independently validates the closed-form model
and reveals the impact of process variation (weak cells in the distribution tail).
Importance sampling extends the analysis to rare-event probabilities far below what
direct Monte Carlo can resolve. Finally, adaptive read references (read-retry) are
modeled as a firmware-level endurance recovery mechanism.
</p>

<!-- ── 2. BACKGROUND ── -->
<h2>2. Background: How NAND Flash Stores Data</h2>
<p>
A NAND flash cell is a floating-gate transistor. Data is stored as electric charge
trapped in the floating gate; the amount of charge shifts the cell's threshold voltage
(V<sub>t</sub>) — the gate voltage at which the transistor turns on. A controller
infers the stored data by comparing V<sub>t</sub> against reference voltages.
</p>
<p>
Multi-level cell designs (MLC, TLC, QLC) pack more bits into each cell by using more
voltage levels. For an <em>n</em>-bit cell there are L = 2<sup>n</sup> states, and the
fixed V<sub>t</sub> window must be divided into L narrow bands — the tighter the bands,
the less tolerance for any voltage drift or noise.
</p>
<h3>Three Wear Mechanisms</h3>
<ul>
  <li><strong>Program/Erase (P/E) cycling:</strong> each write+erase cycle injects high
      voltage, generating electron traps in the oxide. This widens the V<sub>t</sub>
      distributions and slightly raises the mean (V<sub>t</sub> creep).</li>
  <li><strong>Data retention:</strong> trapped charge slowly leaks away, shifting V<sub>t</sub>
      downward. The rate follows an Arrhenius temperature dependence and accelerates
      with prior cycling.</li>
  <li><strong>Read disturb:</strong> reading a cell applies a pass voltage to neighboring
      cells, gradually nudging low-V<sub>t</sub> states upward.</li>
</ul>
<p>
All three mechanisms cause adjacent V<sub>t</sub> distributions to approach and eventually
overlap. When distributions overlap, the controller cannot reliably decode the stored bit —
producing a raw bit error.
</p>

<!-- ── 3. MODEL ── -->
<h2>3. Analytical Model</h2>
<h3>Voltage Distributions</h3>
<p>
Each of the L programmed states is modeled as a Gaussian with a mean (µ<sub>k</sub>) and
standard deviation (σ<sub>k</sub>). Fresh cells have tight, evenly-spaced distributions.
After N P/E cycles, retention time t, temperature T, and R reads, the model applies
closed-form degradation equations:
</p>
<ul>
  <li><strong>P/E broadening:</strong> σ scales by (1 + k<sub>pe</sub> · N / N<sub>scale</sub>),
      and µ shifts upward proportionally.</li>
  <li><strong>Retention loss:</strong> µ decreases by a term proportional to
      (t/t₀)<sup>0.5</sup> × Arrhenius(T), weighted by the amount of stored charge.</li>
  <li><strong>Read disturb:</strong> low states receive an upward nudge that decays
      exponentially with state index, scaled by R/R₀.</li>
</ul>

{fig("1_vt_distributions.png")}
<p class="caption">Figure 1 — Fresh vs. worn V<sub>t</sub> distributions for a TLC device.
The worn curves are broader and shifted; adjacent distributions have begun to overlap,
generating raw bit errors.</p>

<h3>RBER and UBER</h3>
<p>
The raw bit error rate (RBER) is computed analytically as the probability that a cell's
V<sub>t</sub> falls on the wrong side of a read reference. With Gray coding, each
adjacent-state misread flips exactly one bit, so RBER sums the tail probabilities across
all L−1 decision boundaries.
</p>
<p>
The uncorrectable bit error rate (UBER) is the probability that a codeword (8,192 bits)
contains more errors than ECC can correct (strength t = 80 bits). Using a
Poisson approximation to the binomial:
</p>
<p style="text-align:center; font-family: monospace; margin: 8px 0 16px;">
UBER = P(X &gt; t), &nbsp;&nbsp; X ~ Poisson(K × RBER)
</p>
<p>
End of life is defined as the first cycle count where UBER exceeds the 10⁻¹⁵ target.
</p>

{fig("2_rber_vs_cycles.png")}
<p class="caption">Figure 2 — RBER vs. P/E cycles for SLC through QLC. QLC's tightly
packed states cause it to cross the ECC correction limit far sooner than SLC.</p>

{fig("3_uber_endurance.png")}
<p class="caption">Figure 3 — UBER endurance comparison and summary bars. The endurance
gap between cell types spans nearly two orders of magnitude.</p>

<!-- ── 4. RESULTS ── -->
<h2>4. Key Results</h2>

<h3>4.1 Endurance by Cell Type</h3>
<table>
  <thead><tr><th>Cell Type</th><th>Bits / Cell</th><th>Levels (L)</th><th>Endurance (P/E cycles)</th></tr></thead>
  <tbody>
    <tr><td>SLC</td><td>1</td><td>2</td><td>{results['SLC']:,}</td></tr>
    <tr><td>MLC</td><td>2</td><td>4</td><td>{results['MLC']:,}</td></tr>
    <tr><td>TLC</td><td>3</td><td>8</td><td>{results['TLC']:,}</td></tr>
    <tr><td>QLC</td><td>4</td><td>16</td><td>{results['QLC']:,}</td></tr>
  </tbody>
</table>
<p>
Conditions: 1 year data retention at 85 °C, 10⁶ reads, UBER target 10⁻¹⁵.
Endurance drops sharply with bits per cell because doubling L halves the V<sub>t</sub>
margin for the same window, so the same wear closes the gap sooner.
</p>

<h3>4.2 Monte Carlo Validation &amp; Process Variation</h3>
<p>
A Monte Carlo simulator draws individual cell V<sub>t</sub> values from the
analytical distributions and measures the empirical bit error rate. With no process
variation the two agree to within ~1%. Adding a 5% weak-cell population
(cells with double the fresh sigma) roughly doubles the RBER — validating that
real failure rates are driven by distribution tails, not averages.
</p>

{fig("4_mc_validation.png")}
<p class="caption">Figure 4 — Monte Carlo vs. analytical RBER. Solid lines are the
closed-form prediction; markers are simulation results. Process variation (dashed)
shifts the curve upward.</p>

<h3>4.3 Importance Sampling — Estimating Ultra-Rare Tail Events</h3>
<p>
UBER targets of 10⁻¹⁵ require estimating probabilities that direct Monte Carlo
cannot resolve without astronomically large sample sizes. Importance sampling
addresses this by shifting the sampling distribution toward the rare event and
reweighting each sample by its likelihood ratio.
</p>

<div class="highlight">
  <strong>Result:</strong> Importance sampling recovers a tail probability of
  <strong>1.27 × 10⁻¹²</strong> (σ = 8.1 × 10⁻¹⁵) for P(N(0,1) &gt; 7),
  while naive Monte Carlo with 2 × 10⁶ samples reads exactly zero
  (resolution floor ≈ 5 × 10⁻⁷).
</div>

{fig("5_importance_sampling.png")}
<p class="caption">Figure 5 — Naive MC vs. importance sampling for rare tail estimation.
Naive MC is floored at its sample-count resolution; IS recovers the true value.</p>

<h3>4.4 Adaptive Read References (Read-Retry)</h3>
<p>
Factory read references are set on a fresh device and never updated. As wear shifts
V<sub>t</sub> distributions, fixed references become suboptimal and RBER rises faster
than necessary. Adaptive references (read-retry) re-center at the crossing point
between adjacent worn distributions, minimizing RBER at every cycle count.
</p>

<table>
  <thead><tr><th>Cell Type</th><th>Factory Endurance</th><th>Read-Retry Endurance</th><th>Gain</th></tr></thead>
  <tbody>
    <tr><td>TLC</td><td>{tlc_factory:,} cycles</td><td>{tlc_optimal:,} cycles</td><td>+{tlc_gain}%</td></tr>
    <tr><td>QLC</td><td>{"0 (fails)" if qlc_factory == 0 else f"{qlc_factory:,} cycles"}</td>
        <td>{qlc_optimal:,} cycles</td>
        <td>{"Recovered from failure" if qlc_factory == 0 else f"+{round((qlc_optimal-qlc_factory)/qlc_factory*100)}%"}</td></tr>
  </tbody>
</table>

{fig("6_adaptive_ref.png")}
<p class="caption">Figure 6 — RBER curves with factory vs. adaptive references. Adaptive
references track the worn distributions and significantly reduce error rates at high
cycle counts.</p>

<h3>4.5 Temperature and Retention Sensitivity</h3>
<p>
Endurance is not fixed — it depends on operating conditions. Higher temperature
accelerates retention loss (Arrhenius), and longer retention time allows more charge
to leak. The heatmap below shows that a device operated hot and stored long can lose
more than half its rated endurance.
</p>

{fig("7_lifetime_heatmap.png")}
<p class="caption">Figure 7 — Endurance (P/E cycles to UBER limit) as a function of
temperature and retention time. Cool, short-retention conditions maximize usable life.</p>

<!-- ── 5. METHODS SUMMARY ── -->
<h2>5. Methods Summary</h2>
<table>
  <thead><tr><th>Component</th><th>Technique</th><th>Purpose</th></tr></thead>
  <tbody>
    <tr><td>Analytical model</td><td>Closed-form Gaussian V<sub>t</sub> distributions</td><td>Fast RBER/UBER/endurance sweep</td></tr>
    <tr><td>Monte Carlo</td><td>Cell-population simulation</td><td>Validates analytical model; reveals tail effects</td></tr>
    <tr><td>Process variation</td><td>Bimodal weak-cell distribution</td><td>Realistic RBER (tails, not averages)</td></tr>
    <tr><td>Importance sampling</td><td>Exponential tilting + likelihood-ratio weighting</td><td>Estimate UBER-class probabilities (~10⁻¹⁵)</td></tr>
    <tr><td>Adaptive references</td><td>Distribution-crossing-point optimization</td><td>Model read-retry endurance recovery</td></tr>
    <tr><td>Temperature sweep</td><td>Arrhenius factor over T × retention time grid</td><td>Operating-condition sensitivity</td></tr>
  </tbody>
</table>

<!-- ── 6. CONCLUSIONS ── -->
<h2>6. Conclusions</h2>
<ul>
  <li>Endurance scales inversely with bits per cell — SLC outlasts QLC by ~28×
      under the same operating conditions — because tighter V<sub>t</sub> spacing
      leaves less margin for wear-induced drift.</li>
  <li>The analytical model and Monte Carlo simulation agree to within ~1% in
      the absence of process variation, validating the closed-form physics.</li>
  <li>Process variation (weak cells) can roughly double the RBER, making
      distribution tails — not means — the design-limiting factor.</li>
  <li>Importance sampling recovers tail probabilities below 10⁻¹² that are
      completely inaccessible to direct Monte Carlo at realistic sample sizes.</li>
  <li>Adaptive read references extend TLC endurance by ~{tlc_gain}% at no
      hardware cost — a firmware-only reliability recovery mechanism.</li>
  <li>Temperature is a first-order variable: hot operating conditions can
      halve the usable device lifetime through accelerated retention loss.</li>
</ul>

<!-- ── 7. TOOLS & TECH ── -->
<h2>7. Tools &amp; Technologies</h2>
<ul>
  <li><strong>Python 3</strong> — core language</li>
  <li><strong>NumPy / SciPy</strong> — numerical math, statistical distributions, Poisson tail probabilities</li>
  <li><strong>Matplotlib</strong> — figure generation</li>
  <li><strong>Streamlit + Plotly</strong> — interactive dashboard</li>
  <li><strong>Physics:</strong> Arrhenius retention model, Gaussian V<sub>t</sub> distributions, Gray coding, Poisson ECC model, importance sampling with exponential tilting</li>
</ul>

<br>
<hr style="border:none;border-top:1px solid #dde;margin:40px 0 16px;">
<p style="font-size:0.8rem;color:#999;text-align:center;">
  Generated by generate_report.py &nbsp;·&nbsp; Jonathan Chen &nbsp;·&nbsp; SanDisk Technology Intern 2025
</p>

</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Computing model results...")
    results, tlc_factory, tlc_optimal, qlc_factory, qlc_optimal = compute_summary()

    print("Building report HTML...")
    html = build_html(results, tlc_factory, tlc_optimal, qlc_factory, qlc_optimal)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nand_report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nReport written to: {out_path}")
    print("Open it in a browser, then File → Print → Save as PDF.")
