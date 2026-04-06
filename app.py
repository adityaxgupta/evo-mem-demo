# app.py  —  EVO-MEM Streamlit Demo
# Run:  streamlit run app.py

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

from evolution_memory import (
    EvolutionSatelliteDataset,
    EvolutionMemoryModel,
)

# ──────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="EVO-MEM",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────
# CUSTOM CSS  —  clean, professional, no emojis
# ──────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0f0f0f;
        color: #d4d4d4;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 2.5rem 3rem 2rem 3rem; max-width: 1200px; }

    /* ── HEADER ── */
    .evo-header { border-bottom: 1px solid #2a2a2a; padding-bottom: 1.2rem; margin-bottom: 2rem; }
    .evo-label  { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.15em; color: #7a6a50; text-transform: uppercase; }
    .evo-title  { font-size: 3rem; font-weight: 600; letter-spacing: -0.03em; color: #e8e2d8; margin: 0.2rem 0 0.4rem 0; }
    .evo-sub    { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #666; }

    /* ── SECTION LABELS ── */
    .section-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.12em;
        color: #7a6a50;
        text-transform: uppercase;
        border-left: 2px solid #7a6a50;
        padding-left: 0.6rem;
        margin-bottom: 0.8rem;
    }

    /* ── STAT PILLS ── */
    .stat-row { display: flex; gap: 0.8rem; margin: 1rem 0 1.4rem 0; flex-wrap: wrap; }
    .stat-pill {
        background: #181818;
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        padding: 0.5rem 0.9rem;
        min-width: 120px;
    }
    .stat-pill .stat-key {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 0.1em;
        color: #555;
        text-transform: uppercase;
    }
    .stat-pill .stat-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: #c8975a;
        font-weight: 500;
        margin-top: 0.1rem;
    }

    /* ── CAPTION OUTPUT BOX ── */
    .caption-box {
        background: #141414;
        border: 1px solid #2a2a2a;
        border-left: 3px solid #c8975a;
        border-radius: 4px;
        padding: 1.2rem 1.4rem;
        margin: 1rem 0;
    }
    .caption-box .cap-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 0.1em;
        color: #555;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .caption-box .cap-text {
        font-size: 1.15rem;
        font-weight: 500;
        color: #e8e2d8;
        line-height: 1.4;
    }
    .caption-box .cap-match {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #5a9a6a;
        margin-top: 0.5rem;
    }

    /* ── GROUND TRUTH BOX ── */
    .truth-box {
        background: #141414;
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
    }
    .truth-box .truth-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 0.1em;
        color: #555;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .truth-box .truth-text {
        font-size: 0.9rem;
        color: #999;
    }
    .truth-box .truth-theme {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #666;
        margin-top: 0.3rem;
    }

    /* ── TRAINING STATUS ── */
    .train-status {
        background: #141414;
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        padding: 1rem 1.2rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #888;
        margin-bottom: 1rem;
    }

    /* ── RADIO BUTTONS (theme selector) ── */
    div[data-testid="stRadio"] label {
        font-size: 0.9rem !important;
        color: #bbb !important;
    }
    div[data-testid="stRadio"] label:hover {
        color: #e8e2d8 !important;
    }

    /* ── PRIMARY BUTTON ── */
    div.stButton > button {
        background-color: #c8975a;
        color: #0f0f0f;
        border: none;
        border-radius: 3px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        font-weight: 600;
        padding: 0.6rem 1.8rem;
        text-transform: uppercase;
        cursor: pointer;
        transition: background 0.15s;
    }
    div.stButton > button:hover {
        background-color: #daa870;
        color: #0f0f0f;
    }

    /* Matplotlib figure bg transparent */
    .stPlotlyChart, .stPyplot { background: transparent !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────
THEMES = EvolutionSatelliteDataset.THEMES
DEVICE = "cpu"   # Streamlit Cloud has no GPU; CPU is reliable here


# ──────────────────────────────────────────────────────
# TRAINING  —  cached so it only runs once per session
# ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_trained_model():
    """
    Trains EVO-MEM for 10 epochs on synthetic data, exactly replicating
    the notebook pipeline. Cached so this only runs once on startup.
    Returns the trained model and loss history.
    """
    model = EvolutionMemoryModel(num_classes=4).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    dataset = EvolutionSatelliteDataset(size=1000)
    train_set, val_set = random_split(dataset, [800, 200])
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=8, shuffle=False)

    t_hist, v_hist = [], []

    for epoch in range(10):
        # ── Train
        model.train()
        t_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        # ── Validate
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                v_loss += criterion(model(images), labels).item()

        t_hist.append(t_loss / len(train_loader))
        v_hist.append(v_loss / len(val_loader))

    model.eval()
    return model, t_hist, v_hist


# ──────────────────────────────────────────────────────
# INFERENCE HELPER
# ──────────────────────────────────────────────────────
def run_pipeline(model, theme_id: int):
    """
    Generates a 5-frame stream for the chosen theme, passes it through
    the trained EVO-MEM pipeline, and returns prediction + confidence scores.
    """
    stream = EvolutionSatelliteDataset._make_stream(theme_id)   # (5, 3, 224, 224)
    stream_gpu = stream.unsqueeze(0).to(DEVICE)                 # (1, 5, 3, 224, 224)

    with torch.no_grad():
        logits = model(stream_gpu)                              # (1, 4)
        probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_id    = int(np.argmax(probs))
    pred_cap   = THEMES[pred_id]["caption"]
    truth_cap  = THEMES[theme_id]["caption"]
    correct    = pred_id == theme_id

    return stream, pred_id, pred_cap, truth_cap, probs, correct


# ──────────────────────────────────────────────────────
# PLOT HELPERS
# ──────────────────────────────────────────────────────
def plot_stream(stream: torch.Tensor, theme_name: str):
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    fig.patch.set_facecolor("#141414")
    for t in range(5):
        ax = axes[t]
        img = stream[t].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(f"T+{t}", fontsize=8, color="#888",
                     fontfamily="monospace", pad=4)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a2a")
    fig.suptitle(f"Theme: {theme_name}", fontsize=9, color="#7a6a50",
                 fontfamily="monospace", y=0.02)
    plt.tight_layout(pad=0.5)
    return fig


def plot_confidence(probs: np.ndarray, pred_id: int):
    fig, ax = plt.subplots(figsize=(5, 2.6))
    fig.patch.set_facecolor("#141414")
    ax.set_facecolor("#141414")

    labels = [THEMES[i]["name"] for i in range(4)]
    colors = ["#c8975a" if i == pred_id else "#333" for i in range(4)]
    bars   = ax.barh(labels, probs, color=colors, height=0.5)

    for bar, p in zip(bars, probs):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{p:.3f}", va="center", ha="left",
            fontsize=8, color="#aaa", fontfamily="monospace",
        )

    ax.set_xlim(0, 1.15)
    ax.tick_params(colors="#888", labelsize=8)
    ax.xaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", colors="#888")
    plt.tight_layout(pad=0.6)
    return fig


def plot_loss(t_hist, v_hist):
    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor("#141414")
    ax.set_facecolor("#141414")

    epochs = range(1, len(t_hist) + 1)
    ax.plot(epochs, t_hist, color="#c8975a", marker="o", markersize=3,
            linewidth=1.5, label="Train")
    ax.plot(epochs, v_hist, color="#5a9a6a", marker="x", markersize=4,
            linewidth=1.5, label="Val")

    ax.set_xlabel("Epoch", fontsize=8, color="#666", fontfamily="monospace")
    ax.set_ylabel("Loss", fontsize=8, color="#666", fontfamily="monospace")
    ax.tick_params(colors="#666", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a2a")

    legend = ax.legend(fontsize=7, framealpha=0)
    for text in legend.get_texts():
        text.set_color("#888")

    plt.tight_layout(pad=0.6)
    return fig


# ──────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────
st.markdown(
    """
    <div class="evo-header">
        <div class="evo-label">NLP / Remote Sensing Research</div>
        <div class="evo-title">EVO-MEM</div>
        <div class="evo-sub">
            Evolutionary Memory-Augmented Satellite Image Captioning System
            &nbsp;|&nbsp; ViT + Episodic Memory Bank + Evolutionary Selector
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────
# LOAD / TRAIN MODEL
# ──────────────────────────────────────────────────────
train_placeholder = st.empty()
train_placeholder.markdown(
    '<div class="train-status">Initialising model — training in progress, please wait...</div>',
    unsafe_allow_html=True,
)

model, t_hist, v_hist = load_trained_model()
train_placeholder.empty()

# ──────────────────────────────────────────────────────
# STAT PILLS
# ──────────────────────────────────────────────────────
final_train = t_hist[-1]
final_val   = v_hist[-1]

st.markdown(
    f"""
    <div class="stat-row">
        <div class="stat-pill">
            <div class="stat-key">Encoder</div>
            <div class="stat-val">ViT-B/16</div>
        </div>
        <div class="stat-pill">
            <div class="stat-key">Memory Slots</div>
            <div class="stat-val">100</div>
        </div>
        <div class="stat-pill">
            <div class="stat-key">Mutation Rate</div>
            <div class="stat-val">0.10</div>
        </div>
        <div class="stat-pill">
            <div class="stat-key">Temporal Frames</div>
            <div class="stat-val">T = 5</div>
        </div>
        <div class="stat-pill">
            <div class="stat-key">Final Train Loss</div>
            <div class="stat-val">{final_train:.4f}</div>
        </div>
        <div class="stat-pill">
            <div class="stat-key">Final Val Loss</div>
            <div class="stat-val">{final_val:.4f}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────
# MAIN LAYOUT
# ──────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.markdown('<div class="section-label">Select Scene Theme</div>', unsafe_allow_html=True)

    theme_names = [THEMES[i]["name"] for i in range(4)]
    selected_name = st.radio(
        label="theme",
        options=theme_names,
        index=0,
        label_visibility="collapsed",
    )
    selected_id = theme_names.index(selected_name)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Learning Convergence</div>', unsafe_allow_html=True)
    st.pyplot(plot_loss(t_hist, v_hist), use_container_width=True)

with right_col:
    st.markdown('<div class="section-label">System Demo</div>', unsafe_allow_html=True)
    st.markdown(
        "Select a land-cover theme on the left. The system generates a 5-frame "
        "temporal satellite image stream, passes it through the full EVO-MEM pipeline "
        "(ViT encoder → Episodic Memory Bank → Evolutionary Selector → Decoder), "
        "and produces a natural-language caption.",
        unsafe_allow_html=False,
    )

    run = st.button("RUN EVO-MEM PIPELINE")

    if run:
        with st.spinner("Running pipeline..."):
            stream, pred_id, pred_cap, truth_cap, probs, correct = run_pipeline(
                model, selected_id
            )

        # ── Temporal image stream
        st.markdown('<div class="section-label">Temporal Image Stream</div>', unsafe_allow_html=True)
        st.pyplot(plot_stream(stream, selected_name), use_container_width=True)

        # ── Caption output
        match_line = (
            '<div class="cap-match">PREDICTION CORRECT</div>'
            if correct else
            '<div class="cap-match" style="color:#9a5a5a;">PREDICTION MISMATCH — classifier returned alternate theme</div>'
        )
        st.markdown(
            f"""
            <div class="caption-box">
                <div class="cap-label">Model Output Caption</div>
                <div class="cap-text">{pred_cap}</div>
                {match_line}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Ground truth + confidence
        gt_col, conf_col = st.columns(2, gap="medium")

        with gt_col:
            st.markdown('<div class="section-label">Ground Truth</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="truth-box">
                    <div class="truth-label">Expected Caption</div>
                    <div class="truth-text">{truth_cap}</div>
                    <div class="truth-theme">Theme: {selected_name.upper()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with conf_col:
            st.markdown('<div class="section-label">Class Confidence</div>', unsafe_allow_html=True)
            st.pyplot(plot_confidence(probs, pred_id), use_container_width=True)
