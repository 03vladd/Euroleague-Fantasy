"""
Generate an interactive HTML dashboard from backtest_results.csv.

Run:
    python3 dashboard.py
Output: dashboard.html  (self-contained, no server needed)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv("backtest_results.csv")
df = df.sort_values("round").reset_index(drop=True)

rounds = df["round"]

# ── Cumulative totals ─────────────────────────────────────────────────────────
df["cum_ml"]     = df["predicted_team_fp"].cumsum()
df["cum_naive"]  = df["naive_baseline_fp"].cumsum()
df["cum_oracle"] = df["oracle_constrained_fp"].cumsum()

# ── Captain frequency ─────────────────────────────────────────────────────────
cap_counts = df["captain"].value_counts().head(10)

# ── Gap: ML vs Oracle per round ───────────────────────────────────────────────
df["gap"] = df["predicted_team_fp"] - df["oracle_constrained_fp"]
df["gap_color"] = df["gap"].apply(lambda x: "#2ecc71" if x >= 0 else "#e74c3c")

# ── Summary stats ─────────────────────────────────────────────────────────────
ml_mean     = df["predicted_team_fp"].mean()
naive_mean  = df["naive_baseline_fp"].mean()
oracle_mean = df["oracle_constrained_fp"].mean()
ml_vs_naive = ml_mean - naive_mean
ml_pct      = ml_mean / oracle_mean * 100
beats_naive = (df["predicted_team_fp"] > df["naive_baseline_fp"]).sum()
n           = len(df)

# ═══════════════════════════════════════════════════════════════════════════════
# Layout: 3 rows × 2 cols
# ═══════════════════════════════════════════════════════════════════════════════
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        "FP per Round — ML vs Oracle vs Naive",
        "Cumulative FP over Regular Season",
        "ML Gap vs Oracle (per round)",
        "Oracle Overlap (shared players out of 10)",
        "Captain Picks (top 10 frequency)",
        "Round-by-Round % of Oracle Achieved",
    ),
    vertical_spacing=0.13,
    horizontal_spacing=0.10,
    row_heights=[0.38, 0.32, 0.30],
)

# ── Row 1 Left: grouped bar — ML / Oracle / Naive ────────────────────────────
fig.add_trace(go.Bar(
    x=rounds, y=df["oracle_constrained_fp"],
    name="Oracle (constrained)", marker_color="#bdc3c7",
    legendgroup="oracle", showlegend=True,
), row=1, col=1)
fig.add_trace(go.Bar(
    x=rounds, y=df["naive_baseline_fp"],
    name="Naive (fp_last_3)", marker_color="#f39c12",
    legendgroup="naive", showlegend=True,
), row=1, col=1)
fig.add_trace(go.Bar(
    x=rounds, y=df["predicted_team_fp"],
    name="ML optimizer", marker_color="#2980b9",
    legendgroup="ml", showlegend=True,
), row=1, col=1)

# Annotation: avg lines
for y_val, color, label in [
    (ml_mean,    "#2980b9", f"ML avg {ml_mean:.1f}"),
    (naive_mean, "#f39c12", f"Naive avg {naive_mean:.1f}"),
]:
    fig.add_hline(y=y_val, line_dash="dash", line_color=color,
                  annotation_text=label, annotation_position="top right",
                  row=1, col=1)

# ── Row 1 Right: cumulative lines ─────────────────────────────────────────────
fig.add_trace(go.Scatter(
    x=rounds, y=df["cum_oracle"],
    name="Oracle cumul.", mode="lines+markers",
    line=dict(color="#bdc3c7", width=2), marker=dict(size=4),
    legendgroup="oracle", showlegend=False,
), row=1, col=2)
fig.add_trace(go.Scatter(
    x=rounds, y=df["cum_naive"],
    name="Naive cumul.", mode="lines+markers",
    line=dict(color="#f39c12", width=2), marker=dict(size=4),
    legendgroup="naive", showlegend=False,
), row=1, col=2)
fig.add_trace(go.Scatter(
    x=rounds, y=df["cum_ml"],
    name="ML cumul.", mode="lines+markers",
    line=dict(color="#2980b9", width=2), marker=dict(size=4),
    legendgroup="ml", showlegend=False,
), row=1, col=2)

# ── Row 2 Left: ML gap vs oracle ──────────────────────────────────────────────
fig.add_trace(go.Bar(
    x=rounds, y=df["gap"],
    name="ML − Oracle",
    marker_color=df["gap_color"].tolist(),
    showlegend=False,
), row=2, col=1)
fig.add_hline(y=0, line_color="white", line_width=1, row=2, col=1)

# ── Row 2 Right: oracle overlap ───────────────────────────────────────────────
fig.add_trace(go.Bar(
    x=rounds, y=df["oracle_overlap"],
    name="Shared players",
    marker_color="#9b59b6",
    showlegend=False,
), row=2, col=2)
fig.add_hline(
    y=df["oracle_overlap"].mean(), line_dash="dash", line_color="#9b59b6",
    annotation_text=f"avg {df['oracle_overlap'].mean():.1f}",
    annotation_position="top right",
    row=2, col=2,
)

# ── Row 3 Left: captain bar ───────────────────────────────────────────────────
fig.add_trace(go.Bar(
    x=cap_counts.values,
    y=cap_counts.index,
    orientation="h",
    name="Captain frequency",
    marker_color="#1abc9c",
    showlegend=False,
), row=3, col=1)

# ── Row 3 Right: % of oracle achieved per round ───────────────────────────────
pct_series = (df["predicted_team_fp"] / df["oracle_constrained_fp"] * 100).round(1)
fig.add_trace(go.Scatter(
    x=rounds, y=pct_series,
    mode="lines+markers",
    line=dict(color="#e67e22", width=2),
    marker=dict(size=5, color=pct_series,
                colorscale="RdYlGn", cmin=20, cmax=100,
                showscale=False),
    name="% of oracle",
    showlegend=False,
    hovertemplate="Round %{x}: %{y:.1f}% of oracle<extra></extra>",
), row=3, col=2)
fig.add_hline(
    y=ml_pct, line_dash="dash", line_color="#e67e22",
    annotation_text=f"avg {ml_pct:.1f}%",
    annotation_position="top right",
    row=3, col=2,
)
fig.add_hline(y=100, line_dash="dot", line_color="#2ecc71",
              annotation_text="Oracle 100%", row=3, col=2)

# ── Global layout ─────────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text=(
            f"EuroLeague Fantasy — 2025-26 Regular Season Backtest  "
            f"(Rounds 6–38)  |  "
            f"ML avg <b>{ml_mean:.1f}</b>  ·  "
            f"Naive avg <b>{naive_mean:.1f}</b>  ·  "
            f"Oracle avg <b>{oracle_mean:.1f}</b>  |  "
            f"ML beats Naive {beats_naive}/{n} rounds  (+{ml_vs_naive:.1f} FP avg)"
        ),
        font=dict(size=14),
        x=0.5,
    ),
    height=1050,
    template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    barmode="overlay",
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#16213e",
)

# Y-axis labels
fig.update_yaxes(title_text="Fantasy Points", row=1, col=1)
fig.update_yaxes(title_text="Cumulative FP", row=1, col=2)
fig.update_yaxes(title_text="FP gap", row=2, col=1)
fig.update_yaxes(title_text="Players in common", row=2, col=2, range=[0, 10])
fig.update_yaxes(title_text="Rounds as captain", row=3, col=1)
fig.update_yaxes(title_text="% of oracle", row=3, col=2, range=[0, 110])
fig.update_xaxes(title_text="Round", row=3, col=1)
fig.update_xaxes(title_text="Round", row=3, col=2)

out = Path("dashboard.html")
fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
print(f"✓ Dashboard saved to: {out.resolve()}")
print(f"\nKey stats (rounds {df['round'].min()}–{df['round'].max()}):")
print(f"  ML avg FP/round:      {ml_mean:.1f}")
print(f"  Naive avg FP/round:   {naive_mean:.1f}  (ML +{ml_vs_naive:.1f})")
print(f"  Oracle avg FP/round:  {oracle_mean:.1f}  (ML = {ml_pct:.1f}%)")
print(f"  ML beats naive:       {beats_naive}/{n} rounds")
print(f"  Avg oracle overlap:   {df['oracle_overlap'].mean():.1f}/10 players")