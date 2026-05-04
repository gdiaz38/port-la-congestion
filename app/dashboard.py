import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Port of LA Congestion", page_icon="🚢", layout="wide")

DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
LIVE_PATH    = os.path.join(DATA_DIR, "vessels_live.csv")
HISTORY_PATH = os.path.join(DATA_DIR, "vessels_history.csv")

@st.cache_data(ttl=600)
def load_live():
    if not os.path.exists(LIVE_PATH):
        return pd.DataFrame()
    df = pd.read_csv(LIVE_PATH)
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], utc=True)
    return df

@st.cache_data(ttl=600)
def load_history():
    if not os.path.exists(HISTORY_PATH):
        return pd.DataFrame()
    df = pd.read_csv(HISTORY_PATH)
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], utc=True)
    return df

live    = load_live()
history = load_history()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚢 Port of LA / Long Beach — Live Congestion Tracker")
if not live.empty:
    st.caption(f"Last AIS snapshot: {live['snapshot_time'].max().strftime('%Y-%m-%d %H:%M UTC')}  •  {len(live)} vessels tracked")
else:
    st.warning("No live data yet — run pipeline.py first.")
    st.stop()

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("🛳 Vessels Tracked",      len(live))
k2.metric("⚓ Anchored / Delayed",   int(live["is_delayed"].sum()))
k3.metric("📦 In Anchorage Zone",    int(live["in_anchorage"].sum()))
k4.metric("⏱ Avg Predicted Turnaround",
          f"{live['predicted_turnaround'].mean():.1f} hrs"
          if "predicted_turnaround" in live.columns else "N/A")

congestion_counts = live["congestion_level"].value_counts()
dominant = congestion_counts.index[0] if not congestion_counts.empty else "N/A"
color_map = {"Low": "🟢", "Moderate": "🟡", "High": "🟠", "Critical": "🔴"}
k5.metric("🚦 Port Status", f"{color_map.get(dominant,'')} {dominant}")

st.divider()

# ── Live Vessel Map ───────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Live Vessel Positions")

    color_col = "congestion_level" if "congestion_level" in live.columns else "vessel_type"
    cat_colors = {
        "Low":      "#2DC653",
        "Moderate": "#F4D03F",
        "High":     "#F77F00",
        "Critical": "#E63946",
    }

    fig_map = px.scatter_mapbox(
        live,
        lat="lat", lon="lon",
        color=color_col,
        color_discrete_map=cat_colors,
        hover_name="vessel_name",
        hover_data={
            "vessel_type":            True,
            "speed_knots":            True,
            "predicted_turnaround":   True,
            "congestion_level":       True,
            "terminal":               True,
            "is_delayed":             True,
            "lat": False, "lon": False,
        },
        size_max=12,
        zoom=11,
        center={"lat": 33.745, "lon": -118.22},
        mapbox_style="carto-positron",
        height=480,
    )
    fig_map.update_traces(marker=dict(size=10))
    fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig_map, use_container_width=True)

with col2:
    st.subheader("Congestion Level Breakdown")
    if "congestion_level" in live.columns:
        cdf = live["congestion_level"].value_counts().reset_index()
        cdf.columns = ["Level", "Count"]
        order  = ["Low", "Moderate", "High", "Critical"]
        cdf["Level"] = pd.Categorical(cdf["Level"], categories=order, ordered=True)
        cdf = cdf.sort_values("Level")

        fig_bar = px.bar(
            cdf, x="Level", y="Count",
            color="Level", color_discrete_map=cat_colors,
            text="Count"
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(height=220, showlegend=False,
                              xaxis_title="", yaxis_title="Vessels")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Vessel Type Mix")
    type_counts = live["vessel_type"].value_counts().reset_index()
    type_counts.columns = ["Type", "Count"]
    fig_pie = px.pie(type_counts, names="Type", values="Count",
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     height=220)
    fig_pie.update_layout(margin=dict(t=10, b=10), showlegend=True)
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Row 2: Turnaround distribution + Terminal comparison ──────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Predicted Turnaround Distribution")
    if "predicted_turnaround" in live.columns:
        fig_hist = px.histogram(
            live, x="predicted_turnaround",
            nbins=20, color_discrete_sequence=["#00B4D8"],
            labels={"predicted_turnaround": "Predicted Turnaround (hrs)"}
        )
        fig_hist.update_layout(height=300, yaxis_title="Vessels")
        st.plotly_chart(fig_hist, use_container_width=True)

with col4:
    st.subheader("Average Turnaround by Terminal")
    if "terminal" in live.columns and "predicted_turnaround" in live.columns:
        term_df = (live.groupby("terminal")["predicted_turnaround"]
                   .mean().reset_index()
                   .sort_values("predicted_turnaround", ascending=True))
        fig_term = px.bar(
            term_df, x="predicted_turnaround", y="terminal",
            orientation="h", color="predicted_turnaround",
            color_continuous_scale="RdYlGn_r",
            labels={"predicted_turnaround": "Avg Turnaround (hrs)", "terminal": ""}
        )
        fig_term.update_layout(height=300, coloraxis_showscale=False)
        st.plotly_chart(fig_term, use_container_width=True)

# ── Historical trend ──────────────────────────────────────────────────────────
if not history.empty and "predicted_turnaround" in history.columns:
    st.subheader("📈 Historical Turnaround Trend")
    hist_daily = (
        history.groupby(history["snapshot_time"].dt.date)["predicted_turnaround"]
        .mean().reset_index()
    )
    hist_daily.columns = ["date", "avg_turnaround"]
    fig_line = px.line(
        hist_daily, x="date", y="avg_turnaround",
        labels={"date": "Date", "avg_turnaround": "Avg Predicted Turnaround (hrs)"},
        color_discrete_sequence=["#00B4D8"]
    )
    fig_line.update_layout(height=280)
    st.plotly_chart(fig_line, use_container_width=True)

st.divider()

# ── Vessel table ──────────────────────────────────────────────────────────────
st.subheader("🔍 Vessel Detail")

col_search, col_f1, col_f2 = st.columns([3, 1, 1])
with col_search:
    search = st.text_input("Search vessel name", placeholder="e.g. INDEPENDENCE, RUBY...",
                           label_visibility="collapsed")
with col_f1:
    delayed_only = st.checkbox("Delayed only", value=False)
with col_f2:
    levels = ["All"] + [l for l in ["Low","Moderate","High","Critical"]
                        if l in live.get("congestion_level", pd.Series()).values]
    level_filter = st.selectbox("Congestion level", levels, label_visibility="collapsed")

filtered = live.copy()
if search:
    filtered = filtered[filtered["vessel_name"].str.contains(search, case=False, na=False)]
if delayed_only:
    filtered = filtered[filtered["is_delayed"] == True]
if level_filter != "All" and "congestion_level" in filtered.columns:
    filtered = filtered[filtered["congestion_level"] == level_filter]

st.caption(f"Showing {len(filtered)} of {len(live)} vessels")

display_cols = {
    "vessel_name":           "Vessel",
    "vessel_type":           "Type",
    "terminal":              "Terminal",
    "speed_knots":           "Speed (kts)",
    "in_anchorage":          "Anchorage",
    "is_delayed":            "Delayed",
    "predicted_turnaround":  "Pred. Turnaround (hrs)",
    "congestion_level":      "Congestion",
}
show = (filtered[[c for c in display_cols if c in filtered.columns]]
        .rename(columns=display_cols)
        .reset_index(drop=True))

st.dataframe(show, use_container_width=True, height=400)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Port Status")
    st.markdown(f"**Vessels live:** {len(live)}")
    if not history.empty:
        st.markdown(f"**History records:** {len(history)}")
        st.markdown(f"**Tracking since:** {history['snapshot_time'].min().strftime('%Y-%m-%d')}")
    st.markdown("---")
    st.markdown("**Data Sources**")
    st.markdown("- 🛰 AIS Stream (aisstream.io)")
    st.markdown("- 🤖 GBR delay model (R²=0.871)")
    st.markdown("- 📍 LA/Long Beach bounding box")
    st.markdown("---")
    if st.button("🔄 Force Refresh"):
        st.cache_data.clear()
        st.rerun()
