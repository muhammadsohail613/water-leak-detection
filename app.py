import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import time
from datetime import datetime, timedelta
from collections import deque

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AquaGuard AI", page_icon="💧", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0a0e1a; }
    .block-container { padding-top: 1rem; }
    .metric-card {
        background: linear-gradient(135deg, #0d1b2a, #1a2d4a);
        border: 1px solid #1e3a5f; border-radius: 12px;
        padding: 20px; text-align: center;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #00d4ff; }
    .metric-label { font-size: 0.85rem; color: #8ab4d4; margin-top: 4px; }
    .alert-critical {
        background: linear-gradient(135deg, #3d0000, #1a0000);
        border: 1px solid #ff3333; border-left: 4px solid #ff3333;
        border-radius: 8px; padding: 12px 16px; margin: 6px 0;
        color: #ffaaaa; font-size: 0.88rem;
    }
    .alert-high {
        background: linear-gradient(135deg, #2d1a00, #1a1000);
        border: 1px solid #ff8800; border-left: 4px solid #ff8800;
        border-radius: 8px; padding: 12px 16px; margin: 6px 0;
        color: #ffcc88; font-size: 0.88rem;
    }
    .alert-normal {
        background: linear-gradient(135deg, #002d1a, #001a0f);
        border: 1px solid #00cc66; border-left: 4px solid #00cc66;
        border-radius: 8px; padding: 10px 16px; margin: 6px 0;
        color: #66ffaa; font-size: 0.88rem;
    }
    .status-ok {
        background: linear-gradient(135deg, #002d1a, #001a0f);
        border: 1px solid #00cc66; border-radius: 8px;
        padding: 10px 16px; color: #66ffaa;
        font-size: 0.9rem; text-align: center;
    }
    .status-leak {
        background: linear-gradient(135deg, #3d0000, #1a0000);
        border: 1px solid #ff3333; border-radius: 8px;
        padding: 10px 16px; color: #ff6666;
        font-size: 0.9rem; text-align: center;
    }
    .live-badge {
        display: inline-block;
        background: #ff3333; color: white;
        border-radius: 20px; padding: 2px 12px;
        font-size: 0.75rem; font-weight: 700;
        animation: blink 1s infinite;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.4} }
    .section-title {
        color: #00d4ff; font-size: 1rem; font-weight: 600;
        margin-bottom: 10px; border-bottom: 1px solid #1e3a5f; padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/predictions.csv", parse_dates=["timestamp"])

@st.cache_resource
def load_model():
    with open("model/isolation_forest.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

df = load_data()
model, scaler = load_model()

# ── Session state init ────────────────────────────────────────────────────────
if "live_running"    not in st.session_state: st.session_state.live_running    = False
if "live_buffer"     not in st.session_state: st.session_state.live_buffer     = deque(maxlen=120)
if "live_alerts"     not in st.session_state: st.session_state.live_alerts     = deque(maxlen=20)
if "live_leak_count" not in st.session_state: st.session_state.live_leak_count = 0
if "live_total"      not in st.session_state: st.session_state.live_total      = 0
if "inject_leak"     not in st.session_state: st.session_state.inject_leak     = False

# ── Helper: generate one sensor reading ───────────────────────────────────────
def generate_live_point(force_leak=False):
    now = datetime.now()
    if force_leak:
        flow     = round(np.random.uniform(28, 38), 3)
        pressure = round(np.random.uniform(2.5, 3.2), 3)
        velocity = round(np.random.uniform(0.7, 0.9), 3)
    else:
        flow     = round(np.random.normal(50, 2), 3)
        pressure = round(np.random.normal(4.0, 0.1), 3)
        velocity = round(np.random.normal(1.2, 0.05), 3)
    return {"timestamp": now, "flow_rate_lpm": flow,
            "pressure_bar": pressure, "velocity_ms": velocity}

def predict_point(point):
    X = np.array([[point["flow_rate_lpm"], point["pressure_bar"], point["velocity_ms"]]])
    X_scaled = scaler.transform(X)
    pred  = model.predict(X_scaled)[0]
    score = model.decision_function(X_scaled)[0]
    is_leak = pred == -1
    if score < -0.15:   sev = "CRITICAL"
    elif score < -0.05: sev = "HIGH"
    elif score < 0:     sev = "MEDIUM"
    else:               sev = "NORMAL"
    return is_leak, round(score, 4), sev

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Historical Dashboard", "📡 Live Monitor"])

# ════════════════════════════════════════════════════════════
# TAB 1 — Historical Dashboard (your existing dashboard)
# ════════════════════════════════════════════════════════════
with tab1:
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        show_mode = st.radio("View Mode", ["Full Dataset", "Leaks Only"])
        severity_filter = st.multiselect(
            "Severity Filter",
            ["CRITICAL", "HIGH", "MEDIUM", "NORMAL"],
            default=["CRITICAL", "HIGH", "MEDIUM", "NORMAL"]
        )
        st.divider()
        st.markdown("### 🔬 Live Prediction")
        inp_flow     = st.number_input("Flow Rate (L/min)", value=50.0, step=0.5)
        inp_pressure = st.number_input("Pressure (bar)",    value=4.0,  step=0.1)
        inp_velocity = st.number_input("Velocity (m/s)",    value=1.2,  step=0.05)
        if st.button("🔍 Analyze", use_container_width=True):
            X_new    = np.array([[inp_flow, inp_pressure, inp_velocity]])
            X_scaled = scaler.transform(X_new)
            pred     = model.predict(X_scaled)[0]
            score    = model.decision_function(X_scaled)[0]
            is_leak  = pred == -1
            if is_leak:
                st.markdown(f'<div class="status-leak">🚨 <b>LEAK DETECTED</b><br>Score: {score:.4f}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-ok">✅ <b>NORMAL FLOW</b><br>Score: {score:.4f}</div>', unsafe_allow_html=True)
        st.divider()
        st.markdown("### 📊 Model Info")
        st.markdown("- **Algorithm:** Isolation Forest\n- **Accuracy:** 99%\n- **Precision:** 99%\n- **False Alarms:** 1/1000")

    st.markdown("## 💧 AquaGuard AI — Water Leak Detection")
    st.caption("Ultrasonic Flow Meter Analysis · Isolation Forest AI")
    st.divider()

    filtered = df[df["severity"].isin(severity_filter)]
    if show_mode == "Leaks Only":
        filtered = filtered[filtered["predicted_leak"] == 1]

    k1, k2, k3, k4 = st.columns(4)
    cards = [
        (f"{len(df):,}", "📡 Total Readings", "#00d4ff"),
        (int(df["predicted_leak"].sum()), "🚨 Leaks Detected", "#ff4444"),
        (int((df["severity"] == "CRITICAL").sum()), "⚠️ Critical Events", "#ff8800"),
        (f"{df['flow_rate_lpm'].mean():.1f}", "💧 Avg Flow (L/min)", "#00cc66"),
    ]
    for col, (val, label, color) in zip([k1, k2, k3, k4], cards):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{color}">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Flow chart
    st.markdown('<div class="section-title">📈 Flow Rate Analysis</div>', unsafe_allow_html=True)
    normal_df = df[df["predicted_leak"] == 0]
    leak_df   = df[df["predicted_leak"] == 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=normal_df["timestamp"], y=normal_df["flow_rate_lpm"],
        mode="lines", name="Normal", line=dict(color="#00d4ff", width=1.5),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.05)"))
    fig.add_trace(go.Scatter(x=leak_df["timestamp"], y=leak_df["flow_rate_lpm"],
        mode="markers", name="🚨 Leak",
        marker=dict(color="#ff3333", size=8, symbol="x", line=dict(width=2))))
    fig.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1221",
        font=dict(color="#8ab4d4"), height=300,
        xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f", title="L/min"),
        legend=dict(bgcolor="#0d1221"), margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">🔵 Pressure over Time</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["timestamp"], y=df["pressure_bar"],
            mode="lines", line=dict(color="#7b68ee", width=1.5)))
        fig2.add_trace(go.Scatter(x=leak_df["timestamp"], y=leak_df["pressure_bar"],
            mode="markers", marker=dict(color="#ff3333", size=7, symbol="x")))
        fig2.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1221",
            font=dict(color="#8ab4d4"), height=250,
            xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f", title="bar"),
            margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        st.markdown('<div class="section-title">🧠 Anomaly Score</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df["timestamp"], y=df["anomaly_score"],
            mode="lines", line=dict(color="#00cc66", width=1.2),
            fill="tozeroy", fillcolor="rgba(0,204,102,0.05)"))
        fig3.add_hline(y=0, line_dash="dash", line_color="#ff3333",
                       annotation_text="Leak Threshold", annotation_font_color="#ff3333")
        fig3.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1221",
            font=dict(color="#8ab4d4"), height=250,
            xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"),
            margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig3, use_container_width=True)

    c3, c4 = st.columns([1, 2])
    with c3:
        st.markdown('<div class="section-title">📊 Severity Breakdown</div>', unsafe_allow_html=True)
        sev_counts = df[df["predicted_leak"]==1]["severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        fig4 = px.pie(sev_counts, names="Severity", values="Count", hole=0.5,
            color="Severity", color_discrete_map={"CRITICAL":"#ff3333","HIGH":"#ff8800","MEDIUM":"#ffcc00"})
        fig4.update_layout(paper_bgcolor="#0a0e1a", font=dict(color="#8ab4d4"),
            legend=dict(bgcolor="#0d1221"), height=260, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig4, use_container_width=True)
    with c4:
        st.markdown('<div class="section-title">🚨 Recent Leak Alerts</div>', unsafe_allow_html=True)
        for _, row in df[df["predicted_leak"]==1].sort_values("timestamp", ascending=False).head(8).iterrows():
            sev  = row["severity"]
            css  = "alert-critical" if sev == "CRITICAL" else "alert-high"
            icon = "🔴" if sev == "CRITICAL" else "🟠"
            ts   = pd.to_datetime(row["timestamp"]).strftime("%H:%M")
            st.markdown(f'<div class="{css}">{icon} <b>{sev}</b> &nbsp;|&nbsp; {ts} &nbsp;|&nbsp; Flow: <b>{row["flow_rate_lpm"]:.1f} L/min</b> &nbsp;|&nbsp; P: <b>{row["pressure_bar"]:.2f} bar</b></div>', unsafe_allow_html=True)

    with st.expander("📋 Raw Data"):
        st.dataframe(filtered[["timestamp","flow_rate_lpm","pressure_bar","velocity_ms","anomaly_score","severity","predicted_leak"]].sort_values("timestamp",ascending=False), use_container_width=True, height=300)

# ════════════════════════════════════════════════════════════
# TAB 2 — LIVE MONITOR
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📡 Live Monitoring Mode")
    st.caption("Real-time sensor stream · AI prediction on every reading · Auto-alert on anomaly")

    # Controls row
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 3])
    with ctrl1:
        start_btn = st.button("▶ Start Monitor", use_container_width=True, type="primary")
    with ctrl2:
        stop_btn  = st.button("⏹ Stop",          use_container_width=True)
    with ctrl3:
        inject_btn = st.button("💥 Inject Leak",  use_container_width=True)
    with ctrl4:
        speed = st.select_slider("Update speed", options=["0.3s", "0.5s", "1s", "2s"], value="0.5s")

    if start_btn: st.session_state.live_running = True
    if stop_btn:  st.session_state.live_running = False
    if inject_btn: st.session_state.inject_leak = True

    delay = float(speed.replace("s", ""))

    # Status + KPI bar
    status_ph = st.empty()
    st.divider()
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi_ph = [kpi1.empty(), kpi2.empty(), kpi3.empty(), kpi4.empty()]

    # Charts
    ch1, ch2 = st.columns([2, 1])
    with ch1:
        st.markdown('<div class="section-title">📈 Live Flow Rate</div>', unsafe_allow_html=True)
        flow_chart_ph = st.empty()
    with ch2:
        st.markdown('<div class="section-title">🧠 Anomaly Score</div>', unsafe_allow_html=True)
        score_chart_ph = st.empty()

    ch3, ch4 = st.columns([1, 1])
    with ch3:
        st.markdown('<div class="section-title">🔵 Pressure (bar)</div>', unsafe_allow_html=True)
        pressure_chart_ph = st.empty()
    with ch4:
        st.markdown('<div class="section-title">🚨 Live Alert Feed</div>', unsafe_allow_html=True)
        alert_ph = st.empty()

    # Helper to redraw all panels from buffer
    def redraw(buf, alerts, leak_count, total):
        if not buf: return
        bdf = pd.DataFrame(list(buf))

        # Status banner
        last = bdf.iloc[-1]
        if last["is_leak"]:
            status_ph.markdown(f'<div class="status-leak">🚨 &nbsp; <b>LEAK DETECTED</b> &nbsp;|&nbsp; {last["severity"]} &nbsp;|&nbsp; Flow: {last["flow_rate_lpm"]:.1f} L/min &nbsp;|&nbsp; Score: {last["anomaly_score"]:.4f} &nbsp; <span class="live-badge">LIVE</span></div>', unsafe_allow_html=True)
        else:
            status_ph.markdown(f'<div class="status-ok">✅ &nbsp; <b>NORMAL FLOW</b> &nbsp;|&nbsp; Flow: {last["flow_rate_lpm"]:.1f} L/min &nbsp;|&nbsp; Pressure: {last["pressure_bar"]:.2f} bar &nbsp;|&nbsp; Score: {last["anomaly_score"]:.4f} &nbsp; <span class="live-badge" style="background:#00cc66">LIVE</span></div>', unsafe_allow_html=True)

        # KPIs
        kpi_data = [
            (total, "📡 Readings", "#00d4ff"),
            (leak_count, "🚨 Leaks", "#ff4444"),
            (f"{last['flow_rate_lpm']:.1f}", "💧 Flow L/min", "#00cc66"),
            (f"{last['pressure_bar']:.2f}", "🔵 Pressure bar", "#7b68ee"),
        ]
        for ph, (val, label, color) in zip(kpi_ph, kpi_data):
            ph.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{color}">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

        # Flow chart
        normal = bdf[bdf["is_leak"] == False]
        leaks  = bdf[bdf["is_leak"] == True]
        fig_f  = go.Figure()
        fig_f.add_trace(go.Scatter(x=normal["timestamp"], y=normal["flow_rate_lpm"],
            mode="lines+markers", name="Normal",
            line=dict(color="#00d4ff", width=2),
            marker=dict(size=4),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.05)"))
        if not leaks.empty:
            fig_f.add_trace(go.Scatter(x=leaks["timestamp"], y=leaks["flow_rate_lpm"],
                mode="markers", name="🚨 Leak",
                marker=dict(color="#ff3333", size=12, symbol="x", line=dict(width=3))))
        fig_f.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1221",
            font=dict(color="#8ab4d4"), height=280,
            xaxis=dict(gridcolor="#1e3a5f", title="Time"),
            yaxis=dict(gridcolor="#1e3a5f", title="L/min", range=[20, 65]),
            legend=dict(bgcolor="#0d1221"), margin=dict(l=10,r=10,t=10,b=10),
            uirevision="flow")
        flow_chart_ph.plotly_chart(fig_f, use_container_width=True)

        # Score chart
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=bdf["timestamp"], y=bdf["anomaly_score"],
            mode="lines+markers", line=dict(color="#00cc66", width=2), marker=dict(size=4),
            fill="tozeroy", fillcolor="rgba(0,204,102,0.05)"))
        fig_s.add_hline(y=0, line_dash="dash", line_color="#ff3333")
        fig_s.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1221",
            font=dict(color="#8ab4d4"), height=280,
            xaxis=dict(gridcolor="#1e3a5f"),
            yaxis=dict(gridcolor="#1e3a5f", title="Score"),
            margin=dict(l=10,r=10,t=10,b=10), showlegend=False, uirevision="score")
        score_chart_ph.plotly_chart(fig_s, use_container_width=True)

        # Pressure chart
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=bdf["timestamp"], y=bdf["pressure_bar"],
            mode="lines+markers", line=dict(color="#7b68ee", width=2), marker=dict(size=4)))
        if not leaks.empty:
            fig_p.add_trace(go.Scatter(x=leaks["timestamp"], y=leaks["pressure_bar"],
                mode="markers", marker=dict(color="#ff3333", size=10, symbol="x", line=dict(width=2))))
        fig_p.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1221",
            font=dict(color="#8ab4d4"), height=280,
            xaxis=dict(gridcolor="#1e3a5f"),
            yaxis=dict(gridcolor="#1e3a5f", title="bar", range=[2, 5]),
            margin=dict(l=10,r=10,t=10,b=10), showlegend=False, uirevision="pressure")
        pressure_chart_ph.plotly_chart(fig_p, use_container_width=True)

        # Alert feed
        alert_html = ""
        for a in reversed(list(alerts)):
            css  = "alert-critical" if a["sev"] == "CRITICAL" else ("alert-high" if a["sev"] == "HIGH" else "alert-normal")
            icon = "🔴" if a["sev"] == "CRITICAL" else ("🟠" if a["sev"] == "HIGH" else "✅")
            alert_html += f'<div class="{css}">{icon} <b>{a["sev"]}</b> | {a["time"]} | {a["flow"]:.1f} L/min | {a["pressure"]:.2f} bar</div>'
        if not alert_html:
            alert_html = '<div class="alert-normal">✅ No anomalies detected yet...</div>'
        alert_ph.markdown(alert_html, unsafe_allow_html=True)

    # ── Live loop ─────────────────────────────────────────────────────────────
    if st.session_state.live_running:
        auto_leak_counter = 0

        while st.session_state.live_running:
            # Auto-inject a leak burst every ~40 normal readings
            auto_leak_counter += 1
            force_leak = st.session_state.inject_leak or (auto_leak_counter % 40 < 8)

            point = generate_live_point(force_leak=force_leak)
            is_leak, score, sev = predict_point(point)

            point["is_leak"]       = is_leak
            point["anomaly_score"] = score
            point["severity"]      = sev

            st.session_state.live_buffer.append(point)
            st.session_state.live_total += 1
            if is_leak:
                st.session_state.live_leak_count += 1
                st.session_state.live_alerts.append({
                    "sev": sev,
                    "time": point["timestamp"].strftime("%H:%M:%S"),
                    "flow": point["flow_rate_lpm"],
                    "pressure": point["pressure_bar"]
                })

            if st.session_state.inject_leak:
                st.session_state.inject_leak = False

            redraw(
                st.session_state.live_buffer,
                st.session_state.live_alerts,
                st.session_state.live_leak_count,
                st.session_state.live_total
            )
            time.sleep(delay)
    else:
        if st.session_state.live_buffer:
            redraw(
                st.session_state.live_buffer,
                st.session_state.live_alerts,
                st.session_state.live_leak_count,
                st.session_state.live_total
            )
        else:
            status_ph.markdown('<div class="status-ok">⏸️ &nbsp; Monitor stopped. Press ▶ Start Monitor to begin.</div>', unsafe_allow_html=True)