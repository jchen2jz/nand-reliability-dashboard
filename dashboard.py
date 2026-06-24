import os
import sys
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from nand_model import (
    NANDConfig, level_means, base_sigmas,
    degraded_distributions, midpoint_refs, optimal_refs, factory_refs,
    analytical_rber, uncorrectable_prob, endurance_cycles, rber_curve,
    _refs_for,
)

st.set_page_config(layout="wide", page_title="NAND Reliability Dashboard")

st.title("3D NAND Flash Reliability Model")
st.caption("Physics-based model of threshold-voltage degradation, RBER, and endurance")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")

    cell_label = st.selectbox("Cell type", ["SLC (1 bit)", "MLC (2 bit)", "TLC (3 bit)", "QLC (4 bit)"])
    bits_map = {"SLC (1 bit)": 1, "MLC (2 bit)": 2, "TLC (3 bit)": 3, "QLC (4 bit)": 4}
    bits = bits_map[cell_label]

    temp_C = st.slider("Temperature (°C)", 25, 125, 85)
    ret_time_h = st.slider("Retention time (hours), 8760 = 1 year", 0, 8760, 8760, step=100)
    reads = st.slider("Read count", 0, 2_000_000, 1_000_000, step=10_000)
    cycle_view = st.slider("Cycle count (distribution view)", 0, 20_000, 0, step=100)

    ref_label = st.selectbox(
        "Read reference mode",
        ["Track current", "Optimal read-retry", "Factory fixed"],
    )
    ref_map = {"Track current": "track", "Optimal read-retry": "optimal", "Factory fixed": "factory"}
    ref_mode = ref_map[ref_label]

cfg = NANDConfig(bits_per_cell=bits)

# Pre-compute shared values
cyc_grid = np.arange(0, 20_001, 200)

@st.cache_data
def get_rber_curve(bits, temp_C, ret_time_h, reads, ref_mode):
    c = NANDConfig(bits_per_cell=bits)
    return rber_curve(c, np.arange(0, 20_001, 200), ret_time_h, temp_C, reads, ref_mode)

@st.cache_data
def get_endurance(bits, temp_C, ret_time_h, reads, ref_mode):
    c = NANDConfig(bits_per_cell=bits)
    return endurance_cycles(c, ret_time_h, temp_C, reads, ref_mode)

@st.cache_data
def get_all_endurances(temp_C, ret_time_h, reads, ref_mode):
    results = {}
    for b, label in [(1, "SLC"), (2, "MLC"), (3, "TLC"), (4, "QLC")]:
        c = NANDConfig(bits_per_cell=b)
        results[label] = endurance_cycles(c, ret_time_h, temp_C, reads, ref_mode)
    return results

rber_vals = get_rber_curve(bits, temp_C, ret_time_h, reads, ref_mode)
end_cycles = get_endurance(bits, temp_C, ret_time_h, reads, ref_mode)

means_now, sig_now = degraded_distributions(cfg, cycle_view, ret_time_h, temp_C, reads)
refs_now = _refs_for(cfg, means_now, sig_now, ref_mode)
rber_now = analytical_rber(means_now, sig_now, refs_now, bits)
uber_now = uncorrectable_prob(rber_now, cfg)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Vt Distributions", "RBER & Endurance", "Cell Type Comparison"])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Vt Distributions
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Threshold Voltage Distributions")
    st.caption(
        f"Dashed = fresh (0 cycles) · Solid = worn ({cycle_view:,} cycles) · "
        f"Vertical lines = read references ({ref_label})"
    )

    x = np.linspace(cfg.v_low, cfg.v_high, 800)

    means_fresh = level_means(cfg)
    sig_fresh = base_sigmas(cfg)

    colors = px.colors.qualitative.Plotly

    fig1 = go.Figure()

    for i, (m, s) in enumerate(zip(means_fresh, sig_fresh)):
        y = norm.pdf(x, m, s)
        fig1.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines",
            line=dict(color=colors[i % len(colors)], dash="dash", width=1.5),
            name=f"State {i} (fresh)",
            legendgroup=f"state{i}",
            showlegend=True,
        ))

    for i, (m, s) in enumerate(zip(means_now, sig_now)):
        y = norm.pdf(x, m, s)
        fig1.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines",
            line=dict(color=colors[i % len(colors)], width=2.5),
            name=f"State {i} (worn)",
            legendgroup=f"state{i}",
            showlegend=True,
        ))

    for r in refs_now:
        fig1.add_vline(x=float(r), line_width=1, line_dash="dot", line_color="gray")

    fig1.update_layout(
        xaxis_title="Threshold Voltage (normalized)",
        yaxis_title="Probability Density",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60),
    )
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Current RBER", f"{rber_now:.3e}")
    col2.metric("Current UBER", f"{uber_now:.3e}")
    col3.metric("Endurance (this config)", f"{end_cycles:,} cycles")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: RBER & Endurance
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("RBER vs P/E Cycles")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=cyc_grid, y=rber_vals,
        mode="lines",
        line=dict(color="#1f77b4", width=2.5),
        name="RBER",
    ))

    if end_cycles <= cyc_grid[-1]:
        fig2.add_vline(
            x=end_cycles,
            line_width=2,
            line_dash="dash",
            line_color="red",
            annotation_text=f"End of life: {end_cycles:,} cycles",
            annotation_position="top right",
            annotation_font_color="red",
        )

    fig2.update_layout(
        xaxis_title="P/E Cycles",
        yaxis_title="Raw Bit Error Rate (RBER)",
        yaxis_type="log",
        height=420,
        margin=dict(t=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Current operating point metrics**")
    c1, c2, c3 = st.columns(3)
    c1.metric("RBER @ selected cycles", f"{rber_now:.3e}")
    c2.metric("UBER @ selected cycles", f"{uber_now:.3e}")
    c3.metric("Endurance (P/E cycles)", f"{end_cycles:,}")

    with st.expander("What do these numbers mean?"):
        st.markdown("""
- **RBER** (Raw Bit Error Rate): fraction of bits read incorrectly before ECC correction. Lower is better.
- **UBER** (Uncorrectable Bit Error Rate): fraction of errors ECC cannot fix. End-of-life is when UBER > 10⁻¹⁵.
- **Endurance**: cycle count at which UBER first exceeds the target — the rated lifetime of the cell.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Cell Type Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Endurance by Cell Type")
    st.caption(f"Temperature: {temp_C}°C · Retention: {ret_time_h:,} h · Reads: {reads:,} · Ref mode: {ref_label}")

    all_end = get_all_endurances(temp_C, ret_time_h, reads, ref_mode)
    labels = list(all_end.keys())
    values = list(all_end.values())

    fig3 = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=["#2ecc71", "#3498db", "#e67e22", "#e74c3c"],
        text=[f"{v:,}" for v in values],
        textposition="outside",
    ))
    fig3.update_layout(
        xaxis_title="Cell Type",
        yaxis_title="Endurance (P/E Cycles)",
        height=420,
        margin=dict(t=40),
        yaxis=dict(range=[0, max(values) * 1.2]),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Endurance summary table**")
    table_data = {
        "Cell Type": labels,
        "Bits/Cell": [1, 2, 3, 4],
        "States": [2, 4, 8, 16],
        "Endurance (P/E cycles)": [f"{v:,}" for v in values],
    }
    st.dataframe(table_data, use_container_width=True, hide_index=True)

    with st.expander("Why does QLC have so much lower endurance?"):
        st.markdown("""
QLC packs **4 bits into one cell**, requiring **16 distinct voltage levels** in the same window.
The gaps between levels are tiny. Any widening or drift from wear quickly causes adjacent
distributions to overlap — producing errors ECC cannot handle. SLC has only 2 levels
with a large margin, so it tolerates far more wear before reaching end-of-life.
""")
