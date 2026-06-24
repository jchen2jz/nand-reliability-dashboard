import os
import sys
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from nand_model import (
    NANDConfig, level_means, base_sigmas,
    degraded_distributions, analytical_rber, uncorrectable_prob,
    endurance_cycles, rber_curve, _refs_for,
)

st.set_page_config(layout="wide", page_title="NAND Reliability Simulator", page_icon="⚡")

# ── Fonts & global styles ─────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
* { font-family: 'Space Grotesk', sans-serif !important; }
.stApp { background: #0d1117; }
.main .block-container { padding-top: 2rem; max-width: 1160px; }

[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] label {
    font-size: 0.76rem !important;
    color: #8b949e !important;
    letter-spacing: 0.02em;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #21262d;
    gap: 0;
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.76rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8b949e !important;
    background: transparent !important;
    border-bottom: 2px solid transparent;
    padding: 0.6rem 1.4rem;
}
.stTabs [aria-selected="true"] {
    color: #e6edf3 !important;
    border-bottom: 2px solid #e8651a !important;
}

[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 4px;
    padding: 1rem 1.2rem;
}
[data-testid="metric-container"] label {
    font-size: 0.7rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8b949e !important;
}
[data-testid="stMetricValue"] div {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.3rem !important;
    color: #e6edf3 !important;
}

[data-testid="stExpander"] {
    border: 1px solid #21262d !important;
    border-radius: 4px;
    background: #161b22;
}
details summary span { font-size: 0.78rem !important; color: #8b949e !important; }

[data-testid="stDataFrame"] * {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid #21262d;">
    <div style="font-size: 0.68rem; font-weight: 600; letter-spacing: 0.18em;
                text-transform: uppercase; color: #e8651a; margin-bottom: 0.5rem;">
        3D NAND Flash · Physics Simulation
    </div>
    <div style="font-size: 1.9rem; font-weight: 600; color: #e6edf3;
                letter-spacing: -0.02em; line-height: 1.1;">
        Reliability Simulator
    </div>
    <div style="font-size: 0.83rem; color: #8b949e; margin-top: 0.5rem;">
        Threshold-voltage degradation model &nbsp;·&nbsp; RBER / UBER &nbsp;·&nbsp; Endurance
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-size: 0.65rem; font-weight: 600; letter-spacing: 0.18em;
                text-transform: uppercase; color: #e8651a;
                padding-bottom: 0.75rem; margin-bottom: 1rem;
                border-bottom: 1px solid #21262d;">
        Parameters
    </div>
    """, unsafe_allow_html=True)

    cell_label = st.selectbox("Cell type", ["SLC (1 bit)", "MLC (2 bit)", "TLC (3 bit)", "QLC (4 bit)"])
    bits = {"SLC (1 bit)": 1, "MLC (2 bit)": 2, "TLC (3 bit)": 3, "QLC (4 bit)": 4}[cell_label]

    st.markdown('<div style="height:0.4rem"></div>', unsafe_allow_html=True)
    temp_C      = st.slider("Temperature (°C)", 25, 125, 85)
    ret_time_h  = st.slider("Retention time (h)", 0, 8760, 8760, step=100, help="8760 h = 1 year")
    reads       = st.slider("Read count", 0, 2_000_000, 1_000_000, step=10_000)
    cycle_view  = st.slider("P/E cycles (Vt view)", 0, 20_000, 0, step=100)

    st.markdown('<div style="height:0.4rem"></div>', unsafe_allow_html=True)
    ref_label = st.selectbox("Read reference mode",
                             ["Track current", "Optimal read-retry", "Factory fixed"])
    ref_mode = {"Track current": "track",
                "Optimal read-retry": "optimal",
                "Factory fixed": "factory"}[ref_label]

# ── Compute ───────────────────────────────────────────────────────────────────
cfg       = NANDConfig(bits_per_cell=bits)
cyc_grid  = np.arange(0, 20_001, 200)

@st.cache_data
def get_rber_curve(bits, temp_C, ret_time_h, reads, ref_mode):
    c = NANDConfig(bits_per_cell=bits)
    return rber_curve(c, np.arange(0, 20_001, 200), ret_time_h, temp_C, reads, ref_mode)

@st.cache_data
def get_endurance(bits, temp_C, ret_time_h, reads, ref_mode):
    return endurance_cycles(NANDConfig(bits_per_cell=bits), ret_time_h, temp_C, reads, ref_mode)

@st.cache_data
def get_all_endurances(temp_C, ret_time_h, reads, ref_mode):
    return {
        label: endurance_cycles(NANDConfig(bits_per_cell=b), ret_time_h, temp_C, reads, ref_mode)
        for b, label in [(1, "SLC"), (2, "MLC"), (3, "TLC"), (4, "QLC")]
    }

rber_vals  = get_rber_curve(bits, temp_C, ret_time_h, reads, ref_mode)
end_cycles = get_endurance(bits, temp_C, ret_time_h, reads, ref_mode)
means_now, sig_now = degraded_distributions(cfg, cycle_view, ret_time_h, temp_C, reads)
refs_now   = _refs_for(cfg, means_now, sig_now, ref_mode)
rber_now   = analytical_rber(means_now, sig_now, refs_now, bits)
uber_now   = uncorrectable_prob(rber_now, cfg)

# ── Shared chart base ─────────────────────────────────────────────────────────
_AXIS = dict(
    gridcolor="#1c2128", linecolor="#21262d", zeroline=False,
    tickfont=dict(family="JetBrains Mono, monospace", size=10, color="#8b949e"),
    title_font=dict(size=11, color="#8b949e"),
)
CHART = dict(
    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
    font=dict(family="Space Grotesk, sans-serif", color="#8b949e", size=11),
    xaxis=_AXIS, yaxis=_AXIS,
    margin=dict(t=16, b=48, l=64, r=24),
)
LEGEND = dict(bgcolor="#161b22", bordercolor="#21262d", borderwidth=1,
              font=dict(family="Space Grotesk", color="#8b949e", size=10))

STATE_COLORS = [
    "#60a5fa", "#34d399", "#f59e0b", "#f87171",
    "#a78bfa", "#fb923c", "#4ade80", "#38bdf8",
    "#e879f9", "#94a3b8", "#fbbf24", "#6ee7b7",
    "#93c5fd", "#fca5a5", "#c4b5fd", "#fdba74",
]

def section_label(text):
    st.markdown(
        f'<div style="font-size:0.68rem;font-weight:600;letter-spacing:0.14em;'
        f'text-transform:uppercase;color:#8b949e;margin:0 0 1rem;'
        f'padding-bottom:0.5rem;border-bottom:1px solid #21262d;">{text}</div>',
        unsafe_allow_html=True,
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Vt Distributions", "RBER & Endurance", "Cell Comparison"])

# ── Tab 1: Vt Distributions ───────────────────────────────────────────────────
with tab1:
    section_label("Threshold Voltage Distributions")
    st.markdown(
        f'<div style="font-size:0.75rem;color:#8b949e;margin-bottom:1rem;">'
        f'<span style="border-bottom:1px dashed #8b949e;padding-bottom:1px;">dashed</span>'
        f' = fresh &nbsp;·&nbsp; solid = {cycle_view:,} cycles &nbsp;·&nbsp; '
        f'<span style="color:#e8651a;">orange lines</span> = read refs ({ref_label})</div>',
        unsafe_allow_html=True,
    )

    x            = np.linspace(cfg.v_low, cfg.v_high, 800)
    means_fresh  = level_means(cfg)
    sig_fresh    = base_sigmas(cfg)

    fig1 = go.Figure()
    for i, (m, s) in enumerate(zip(means_fresh, sig_fresh)):
        fig1.add_trace(go.Scatter(
            x=x.tolist(), y=norm.pdf(x, m, s).tolist(),
            mode="lines",
            line=dict(color=STATE_COLORS[i % len(STATE_COLORS)], dash="dash", width=1.2),
            name=f"S{i} fresh", legendgroup=f"s{i}",
        ))
    for i, (m, s) in enumerate(zip(means_now, sig_now)):
        fig1.add_trace(go.Scatter(
            x=x.tolist(), y=norm.pdf(x, m, s).tolist(),
            mode="lines",
            line=dict(color=STATE_COLORS[i % len(STATE_COLORS)], width=2),
            name=f"S{i} worn", legendgroup=f"s{i}",
        ))
    for r in refs_now:
        fig1.add_vline(x=float(r), line_width=1, line_dash="dot",
                       line_color="#e8651a", opacity=0.6)

    fig1.update_layout(
        **CHART, height=460,
        xaxis_title="Threshold Voltage (normalized)",
        yaxis_title="Probability Density",
        legend=dict(**LEGEND, orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    st.plotly_chart(fig1, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("RBER", f"{rber_now:.3e}")
    c2.metric("UBER", f"{uber_now:.3e}")
    c3.metric("Endurance", f"{end_cycles:,} cycles")

# ── Tab 2: RBER & Endurance ───────────────────────────────────────────────────
with tab2:
    section_label("RBER vs P/E Cycles")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=cyc_grid.tolist(), y=rber_vals.tolist(),
        mode="lines",
        line=dict(color="#60a5fa", width=2),
        fill="tozeroy", fillcolor="rgba(96,165,250,0.06)",
        name="RBER",
    ))
    if end_cycles <= cyc_grid[-1]:
        fig2.add_vline(
            x=end_cycles, line_width=1.5, line_dash="dash", line_color="#f87171",
            annotation_text=f"EOL  {end_cycles:,} cycles",
            annotation_position="top right",
            annotation_font_color="#f87171", annotation_font_size=11,
        )
    fig2.update_layout(**CHART, height=400,
                       xaxis_title="P/E Cycles", yaxis_title="RBER",
                       legend=dict(**LEGEND))
    fig2.update_yaxes(type="log")
    st.plotly_chart(fig2, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("RBER", f"{rber_now:.3e}")
    c2.metric("UBER", f"{uber_now:.3e}")
    c3.metric("Endurance", f"{end_cycles:,} cycles")

    with st.expander("Glossary"):
        st.markdown("""
**RBER** — Raw Bit Error Rate. Fraction of bits read incorrectly before ECC. Lower is better.

**UBER** — Uncorrectable Bit Error Rate. Fraction of errors ECC cannot fix. End-of-life is UBER > 10⁻¹⁵.

**Endurance** — P/E cycle count at which UBER first exceeds the target. The rated lifetime of the cell.
""")

# ── Tab 3: Cell Comparison ────────────────────────────────────────────────────
with tab3:
    section_label("Endurance by Cell Type")
    st.markdown(
        f'<div style="font-size:0.75rem;color:#8b949e;margin-bottom:1rem;">'
        f'{temp_C}°C &nbsp;·&nbsp; {ret_time_h:,} h retention &nbsp;·&nbsp; '
        f'{reads:,} reads &nbsp;·&nbsp; {ref_label}</div>',
        unsafe_allow_html=True,
    )

    all_end = get_all_endurances(temp_C, ret_time_h, reads, ref_mode)
    labels  = list(all_end.keys())
    values  = list(all_end.values())

    fig3 = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=["#60a5fa", "#34d399", "#f59e0b", "#f87171"],
        marker_line_width=0,
        text=[f"{v:,}" for v in values],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=11, color="#8b949e"),
    ))
    fig3.update_layout(
        **CHART, height=380,
        xaxis_title="Cell Type", yaxis_title="Endurance (P/E Cycles)",
        yaxis=dict(**_AXIS, range=[0, max(values) * 1.18]),
        legend=dict(**LEGEND),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(
        {"Cell Type": labels, "Bits / Cell": [1, 2, 3, 4],
         "States": [2, 4, 8, 16],
         "Endurance (cycles)": [f"{v:,}" for v in values]},
        use_container_width=True, hide_index=True,
    )

    with st.expander("Why does QLC have lower endurance?"):
        st.markdown("""
QLC stores **4 bits per cell**, requiring **16 voltage levels** in the same window as SLC's 2.
The margins between levels are tiny — any threshold drift from wear causes adjacent distributions
to overlap, producing errors ECC cannot recover. SLC has only 2 levels with a large margin
and tolerates far more cycles before reaching end-of-life.
""")
