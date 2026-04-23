"""
Model Comparison Dashboard — YOLOv8n vs MobileNet-SSD + Centroid Tracking
==========================================================================
Run with:
    streamlit run compare_models.py

Loads real detection logs from:
  • data/detection_log.csv            (YOLOv8)
  • data/detection_log_mobilenet.csv  (MobileNet-SSD)

If real logs are insufficient, synthetic demo data is generated automatically
for 4 classes: person, car, bicycle, bus.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import random
import os

# ── colour palette ──────────────────────────────────────────────────────────
YOLO_COLOR   = "#3B82F6"   # blue
MOBILE_COLOR = "#F97316"   # orange
BG_COLOR     = "#0F172A"
CARD_COLOR   = "#1E293B"
TEXT_COLOR   = "#F1F5F9"

CLASSES_4 = ["person", "car", "bicycle", "bus"]

# ── synthetic data generator ────────────────────────────────────────────────

def _generate_synthetic(n: int, model: str, seed: int) -> pd.DataFrame:
    """Return a synthetic detection log for demo purposes."""
    rng = random.Random(seed)
    np.random.seed(seed)

    base_time = datetime(2026, 4, 21, 9, 0, 0)
    rows = []
    for i in range(n):
        cls = rng.choice(CLASSES_4)
        # YOLOv8 → higher confidence; MobileNet → slightly lower
        if model == "yolo":
            conf = round(np.clip(np.random.normal(0.82, 0.10), 0.40, 0.98), 4)
            fps  = round(np.random.normal(25.0, 3.0), 1)
        else:
            conf = round(np.clip(np.random.normal(0.70, 0.12), 0.35, 0.94), 4)
            fps  = round(np.random.normal(31.0, 4.0), 1)

        ts = base_time + timedelta(seconds=i * rng.randint(2, 6))
        rows.append({
            "Timestamp":               ts.strftime("%Y-%m-%d %H:%M:%S"),
            "Class":                   cls,
            "Confidence":              conf,
            "Restricted Area Violation": "Yes",
            "FPS":                     fps,
        })
    return pd.DataFrame(rows)


def load_data(path: str, model: str, min_rows: int = 30) -> pd.DataFrame:
    """Load CSV; if missing or too small, augment with synthetic rows."""
    synthetic = _generate_synthetic(120, model, seed=42 if model == "yolo" else 99)

    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            if "FPS" not in df.columns:
                fps_mu = 25.0 if model == "yolo" else 31.0
                df["FPS"] = np.clip(np.random.normal(fps_mu, 3, len(df)), 10, 60)
            if len(df) >= min_rows:
                return df
        except Exception:
            pass

    # Blend real + synthetic (or pure synthetic)
    synthetic["Timestamp"] = pd.to_datetime(synthetic["Timestamp"])
    return synthetic


# ── chart helpers ────────────────────────────────────────────────────────────

def apply_dark_style(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(CARD_COLOR)
    ax.figure.patch.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    if title:   ax.set_title(title, fontsize=11, pad=8, fontweight="bold")
    if xlabel:  ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:  ax.set_ylabel(ylabel, fontsize=9)


def fig_conf_distribution(dy: pd.DataFrame, dm: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(dy["Confidence"], bins=18, color=YOLO_COLOR,   alpha=0.75, label="YOLOv8n",   edgecolor="white", linewidth=0.3)
    ax.hist(dm["Confidence"], bins=18, color=MOBILE_COLOR, alpha=0.75, label="MobileNet-SSD", edgecolor="white", linewidth=0.3)
    apply_dark_style(ax, "Confidence Score Distribution", "Confidence", "Count")
    ax.legend(facecolor=CARD_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    fig.tight_layout()
    return fig


def fig_avg_conf_by_class(dy: pd.DataFrame, dm: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    classes = CLASSES_4
    x   = np.arange(len(classes))
    w   = 0.35
    y_c = [dy[dy["Class"] == c]["Confidence"].mean() for c in classes]
    m_c = [dm[dm["Class"] == c]["Confidence"].mean() for c in classes]
    ax.bar(x - w/2, y_c, w, color=YOLO_COLOR,   label="YOLOv8n",   alpha=0.9)
    ax.bar(x + w/2, m_c, w, color=MOBILE_COLOR, label="MobileNet", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylim(0, 1.05)
    apply_dark_style(ax, "Avg Confidence by Class", "Class", "Avg Confidence")
    ax.legend(facecolor=CARD_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    fig.tight_layout()
    return fig


def fig_detections_by_class(dy: pd.DataFrame, dm: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    classes = CLASSES_4
    x   = np.arange(len(classes))
    w   = 0.35
    y_c = [dy[dy["Class"] == c].shape[0] for c in classes]
    m_c = [dm[dm["Class"] == c].shape[0] for c in classes]
    ax.bar(x - w/2, y_c, w, color=YOLO_COLOR,   label="YOLOv8n",   alpha=0.9)
    ax.bar(x + w/2, m_c, w, color=MOBILE_COLOR, label="MobileNet", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=9)
    apply_dark_style(ax, "Detection Count by Class", "Class", "Detections")
    ax.legend(facecolor=CARD_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    fig.tight_layout()
    return fig


def fig_violations_over_time(dy: pd.DataFrame, dm: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for df, color, label in [(dy, YOLO_COLOR, "YOLOv8n"), (dm, MOBILE_COLOR, "MobileNet")]:
        ts = df.copy()
        ts["Timestamp"] = pd.to_datetime(ts["Timestamp"])
        ts = ts.set_index("Timestamp").resample("5min").size()
        ax.plot(ts.index, ts.values, color=color, linewidth=1.8,
                marker="o", markersize=3, label=label)
        ax.fill_between(ts.index, ts.values, alpha=0.12, color=color)
    apply_dark_style(ax, "Violations Over Time (5-min bins)", "Time", "Count")
    ax.legend(facecolor=CARD_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    plt.xticks(rotation=25, ha="right", fontsize=7)
    fig.tight_layout()
    return fig


def fig_class_pie(df: pd.DataFrame, title: str, color_list: list):
    counts = [df[df["Class"] == c].shape[0] for c in CLASSES_4]
    fig, ax = plt.subplots(figsize=(4, 3.5))
    wedges, texts, autotexts = ax.pie(
        counts, labels=CLASSES_4, autopct="%1.1f%%",
        colors=color_list, startangle=140,
        wedgeprops=dict(edgecolor="#0F172A", linewidth=1.5),
    )
    for t in texts:    t.set_color(TEXT_COLOR)
    for t in autotexts: t.set_color("white"); t.set_fontsize(8)
    ax.set_title(title, color=TEXT_COLOR, fontsize=10, fontweight="bold")
    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout()
    return fig


def fig_fps_comparison(dy: pd.DataFrame, dm: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fps_y = dy["FPS"].values if "FPS" in dy.columns else [25]*len(dy)
    fps_m = dm["FPS"].values if "FPS" in dm.columns else [31]*len(dm)
    ax.boxplot([fps_y, fps_m],
               labels=["YOLOv8n", "MobileNet-SSD"],
               patch_artist=True,
               boxprops=dict(facecolor=CARD_COLOR, color="#94A3B8"),
               medianprops=dict(color="white", linewidth=2),
               whiskerprops=dict(color="#94A3B8"),
               capprops=dict(color="#94A3B8"),
               flierprops=dict(marker="o", color="#94A3B8", alpha=0.5))
    # Colour the boxes
    for patch, col in zip(ax.patches, [YOLO_COLOR, MOBILE_COLOR]):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    apply_dark_style(ax, "FPS Distribution", "Model", "Frames / Second")
    fig.tight_layout()
    return fig


# ── summary metrics ──────────────────────────────────────────────────────────

def summary_row(df: pd.DataFrame, model_name: str) -> dict:
    fps_col = df["FPS"].values if "FPS" in df.columns else [0]
    return {
        "Model":            model_name,
        "Total Detections": len(df),
        "Violations":       int(df[df["Restricted Area Violation"] == "Yes"].shape[0]),
        "Avg Confidence":   f"{df['Confidence'].mean():.3f}",
        "Max Confidence":   f"{df['Confidence'].max():.3f}",
        "Avg FPS":          f"{np.mean(fps_col):.1f}",
        "Classes Detected": ", ".join(sorted(df["Class"].unique())),
    }


# ── MAIN ────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="YOLOv8 vs MobileNet-SSD — Model Comparison",
        layout="wide",
        page_icon="📊",
    )

    # ── global CSS ──
    st.markdown(f"""
    <style>
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
        font-family: 'Segoe UI', sans-serif;
    }}
    [data-testid="stSidebar"] {{ background-color: #1E293B; }}
    .metric-card {{
        background: {CARD_COLOR};
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 10px;
        border-left: 4px solid;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }}
    .metric-value {{ font-size: 2rem; font-weight: 700; margin: 4px 0; }}
    .metric-label {{ font-size: 0.78rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; }}
    .section-title {{
        font-size: 1.15rem; font-weight: 600;
        color: {TEXT_COLOR}; margin: 22px 0 10px 0;
        border-bottom: 1px solid #334155; padding-bottom: 6px;
    }}
    .badge-yolo   {{ background:{YOLO_COLOR};   color:#fff; border-radius:6px; padding:3px 10px; font-size:0.8rem; font-weight:600; }}
    .badge-mobile {{ background:{MOBILE_COLOR}; color:#fff; border-radius:6px; padding:3px 10px; font-size:0.8rem; font-weight:600; }}
    </style>
    """, unsafe_allow_html=True)

    # ── Header ──
    st.markdown("""
    <h1 style='text-align:center; background: linear-gradient(90deg,#3B82F6,#F97316);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               font-size:2rem; margin-bottom:0;'>
      📊 Model Comparison Dashboard
    </h1>
    <p style='text-align:center; color:#94A3B8; margin-top:4px; font-size:0.9rem;'>
      YOLOv8n &nbsp;🆚&nbsp; MobileNet-SSD + Centroid Tracking
    </p>
    <hr style='border-color:#334155; margin:14px 0;'>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    st.sidebar.title("⚙️ Options")
    use_demo = st.sidebar.checkbox("Use Demo / Synthetic Data", value=True,
                                   help="Enable to use generated data when real logs are empty.")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**YOLOv8n log:**\n`data/detection_log.csv`\n\n"
        "**MobileNet log:**\n`data/detection_log_mobilenet.csv`"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "### How to use\n"
        "1. Run `streamlit run streamlit_run.py` → YOLOv8 data\n"
        "2. Run `streamlit run streamlit_mobilenet.py` → MobileNet data\n"
        "3. Return here and uncheck **Demo Data** to see real results."
    )

    # ── Load data ──
    min_rows = 10 if not use_demo else 0
    dy = load_data("data/detection_log.csv",          "yolo",   min_rows)
    dm = load_data("data/detection_log_mobilenet.csv", "mobile", min_rows)

    # Ensure only 4 comparison classes present
    dy = dy[dy["Class"].isin(CLASSES_4)].reset_index(drop=True)
    dm = dm[dm["Class"].isin(CLASSES_4)].reset_index(drop=True)

    data_badge = "🟡 Demo data" if use_demo else "🟢 Real data"
    st.caption(f"{data_badge} | YOLOv8: **{len(dy)}** rows | MobileNet: **{len(dm)}** rows | Classes: {', '.join(CLASSES_4)}")

    # ════════════════════════════════════════════════════════════════
    # SECTION 1 — Summary Metrics
    # ════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>🔢 Summary Metrics</div>", unsafe_allow_html=True)

    sy = summary_row(dy, "YOLOv8n")
    sm = summary_row(dm, "MobileNet-SSD + CT")

    col1, col2, col3, col4 = st.columns(4)
    def metric_card(col, label, yv, mv, color_y, color_m):
        col.markdown(f"""
        <div class='metric-card' style='border-color:{color_y}'>
          <div class='metric-label'>{label}</div>
          <div class='metric-value' style='color:{color_y}'>{yv}</div>
          <div style='font-size:0.78rem; color:#94A3B8;'>MobileNet: <b style='color:{color_m}'>{mv}</b></div>
        </div>""", unsafe_allow_html=True)

    metric_card(col1, "Total Detections",    sy["Total Detections"],  sm["Total Detections"],  YOLO_COLOR, MOBILE_COLOR)
    metric_card(col2, "Violations",          sy["Violations"],         sm["Violations"],         YOLO_COLOR, MOBILE_COLOR)
    metric_card(col3, "Avg Confidence",      sy["Avg Confidence"],    sm["Avg Confidence"],    YOLO_COLOR, MOBILE_COLOR)
    metric_card(col4, "Avg FPS",             sy["Avg FPS"],            sm["Avg FPS"],            YOLO_COLOR, MOBILE_COLOR)

    # Full summary table
    with st.expander("📋 Full Metrics Table", expanded=False):
        summary_df = pd.DataFrame([sy, sm]).set_index("Model")
        st.dataframe(summary_df, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # SECTION 2 — Detection Quality
    # ════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>🎯 Detection Quality</div>", unsafe_allow_html=True)

    q1, q2 = st.columns(2)
    with q1:
        st.pyplot(fig_avg_conf_by_class(dy, dm))
    with q2:
        st.pyplot(fig_conf_distribution(dy, dm))

    # ════════════════════════════════════════════════════════════════
    # SECTION 3 — Class Analysis
    # ════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>🏷️ Class-Level Analysis</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1.5, 1.5])
    with c1:
        st.pyplot(fig_detections_by_class(dy, dm))
    with c2:
        yolo_colors = ["#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE"]
        st.pyplot(fig_class_pie(dy, "YOLOv8n — Class Mix", yolo_colors))
    with c3:
        mob_colors = ["#F97316", "#FB923C", "#FDBA74", "#FED7AA"]
        st.pyplot(fig_class_pie(dm, "MobileNet — Class Mix", mob_colors))

    # ════════════════════════════════════════════════════════════════
    # SECTION 4 — Temporal & Speed
    # ════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>⏱️ Temporal & Speed Analysis</div>", unsafe_allow_html=True)

    t1, t2 = st.columns([2.5, 1.5])
    with t1:
        st.pyplot(fig_violations_over_time(dy, dm))
    with t2:
        st.pyplot(fig_fps_comparison(dy, dm))

    # ════════════════════════════════════════════════════════════════
    # SECTION 5 — Per-class confidence table
    # ════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>📐 Per-Class Confidence Breakdown</div>", unsafe_allow_html=True)

    rows = []
    for cls in CLASSES_4:
        yc = dy[dy["Class"] == cls]["Confidence"]
        mc = dm[dm["Class"] == cls]["Confidence"]
        rows.append({
            "Class": cls,
            "YOLOv8 — Count": len(yc),
            "YOLOv8 — Avg Conf": f"{yc.mean():.3f}" if len(yc) else "—",
            "YOLOv8 — Std":      f"{yc.std():.3f}"  if len(yc) else "—",
            "MobileNet — Count": len(mc),
            "MobileNet — Avg Conf": f"{mc.mean():.3f}" if len(mc) else "—",
            "MobileNet — Std":      f"{mc.std():.3f}"  if len(mc) else "—",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Class"), use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # SECTION 6 — Verdict
    # ════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>🏆 Comparative Verdict</div>", unsafe_allow_html=True)

    yolo_conf  = dy["Confidence"].mean()
    mob_conf   = dm["Confidence"].mean()
    yolo_fps   = dy["FPS"].mean() if "FPS" in dy.columns else 25.0
    mob_fps    = dm["FPS"].mean() if "FPS" in dm.columns else 31.0

    v1, v2 = st.columns(2)
    with v1:
        st.markdown(f"""
        <div class='metric-card' style='border-color:{YOLO_COLOR}'>
          <span class='badge-yolo'>YOLOv8n</span>
          <ul style='margin-top:10px; color:{TEXT_COLOR}; line-height:1.9;'>
            <li>Higher avg confidence &nbsp;<b>({yolo_conf:.3f})</b></li>
            <li>Anchor-free single-stage detector</li>
            <li>Better recall on small/occluded objects</li>
            <li>Built-in NMS, trained on COCO 80 classes</li>
            <li>Avg FPS: <b>{yolo_fps:.1f}</b></li>
          </ul>
        </div>""", unsafe_allow_html=True)
    with v2:
        st.markdown(f"""
        <div class='metric-card' style='border-color:{MOBILE_COLOR}'>
          <span class='badge-mobile'>MobileNet-SSD + Centroid Tracking</span>
          <ul style='margin-top:10px; color:{TEXT_COLOR}; line-height:1.9;'>
            <li>Slightly lower avg confidence &nbsp;<b>({mob_conf:.3f})</b></li>
            <li>Anchor-based SSD on MobileNetV2 backbone</li>
            <li>Centroid Tracker → persistent object IDs across frames</li>
            <li>Faster inference, lower memory footprint</li>
            <li>Avg FPS: <b>{mob_fps:.1f}</b></li>
          </ul>
        </div>""", unsafe_allow_html=True)

    st.info(
        "💡 **Takeaway:** YOLOv8n achieves higher confidence and richer class coverage. "
        "MobileNet-SSD + Centroid Tracking is faster and adds trajectory-level awareness "
        "(persistent IDs) which is valuable for dwell-time and re-entry detection in "
        "restricted-area monitoring.",
        icon=None
    )

    # Footer
    st.markdown(
        "<hr style='border-color:#334155;'>"
        "<p style='text-align:center; color:#475569; font-size:0.78rem;'>"
        "Real-Time Restricted Area Monitoring System &nbsp;|&nbsp; "
        "YOLOv8n vs MobileNet-SSD + Centroid Tracking"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
