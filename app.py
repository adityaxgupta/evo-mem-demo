# app.py  —  EVO-MEM Streamlit Demo

import os
import urllib.request
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

from evolution_memory import EvolutionSatelliteDataset, EvolutionMemoryModel, make_stream, THEMES

st.set_page_config(
    page_title="EVO-MEM",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f0f0f;
    color: #d4d4d4;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 3rem 2rem 3rem; max-width: 1200px; }

.evo-header  { border-bottom: 1px solid #2a2a2a; padding-bottom: 1.2rem; margin-bottom: 2rem; }
.evo-label   { font-family:'JetBrains Mono',monospace; font-size:0.7rem; letter-spacing:0.15em; color:#7a6a50; text-transform:uppercase; }
.evo-title   { font-size:3rem; font-weight:600; letter-spacing:-0.03em; color:#e8e2d8; margin:0.2rem 0 0.4rem 0; }
.evo-sub     { font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#666; }

.sec-label {
    font-family:'JetBrains Mono',monospace; font-size:0.68rem; letter-spacing:0.12em;
    color:#7a6a50; text-transform:uppercase;
    border-left:2px solid #7a6a50; padding-left:0.6rem; margin-bottom:0.8rem;
}

.stat-row { display:flex; gap:0.8rem; margin:1rem 0 1.6rem 0; flex-wrap:wrap; }
.stat-pill { background:#181818; border:1px solid #2a2a2a; border-radius:4px; padding:0.5rem 0.9rem; min-width:120px; }
.stat-pill .sk { font-family:'JetBrains Mono',monospace; font-size:0.6rem; letter-spacing:0.1em; color:#555; text-transform:uppercase; }
.stat-pill .sv { font-family:'JetBrains Mono',monospace; font-size:0.9rem; color:#c8975a; font-weight:500; margin-top:0.1rem; }

.cap-box {
    background:#141414; border:1px solid #2a2a2a; border-left:3px solid #c8975a;
    border-radius:4px; padding:1.2rem 1.4rem; margin:1rem 0;
}
.cap-box .ck { font-family:'JetBrains Mono',monospace; font-size:0.6rem; letter-spacing:0.1em; color:#555; text-transform:uppercase; margin-bottom:0.5rem; }
.cap-box .cv { font-size:1.15rem; font-weight:500; color:#e8e2d8; line-height:1.4; }
.cap-box .cm { font-family:'JetBrains Mono',monospace; font-size:0.7rem; margin-top:0.5rem; }
.match    { color:#5a9a6a; }
.mismatch { color:#9a5a5a; }

.truth-box { background:#141414; border:1px solid #2a2a2a; border-radius:4px; padding:1rem 1.2rem; margin-top:1rem; }
.truth-box .tk { font-family:'JetBrains Mono',monospace; font-size:0.6rem; letter-spacing:0.1em; color:#555; text-transform:uppercase; margin-bottom:0.4rem; }
.truth-box .tv { font-size:0.9rem; color:#999; }
.truth-box .tt { font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#666; margin-top:0.3rem; }

div.stButton > button {
    background-color:#c8975a; color:#0f0f0f; border:none; border-radius:3px;
    font-family:'JetBrains Mono',monospace; font-size:0.78rem; letter-spacing:0.08em;
    font-weight:600; padding:0.6rem 1.8rem; text-transform:uppercase;
}
div.stButton > button:hover { background-color:#daa870; color:#0f0f0f; }
div[data-testid="stRadio"] label { font-size:0.9rem !important; color:#bbb !important; }
div[data-testid="stRadio"] label:hover { color:#e8e2d8 !important; }
</style>
""", unsafe_allow_html=True)

DEVICE = "cpu"

# ──────────────────────────────────────────────────────
# LOAD MODEL  —  download weights, load, then warm up memory bank
# ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_history():
    weights_path = "evo_mem_weights.pth"

    if not os.path.exists(weights_path):
        with st.spinner("Downloading model weights... this only happens once on server wake-up."):
            url = "https://github.com/adityaxgupta/evo-mem-demo/releases/download/v1.0/evo_mem_weights.pth"
            urllib.request.urlretrieve(url, weights_path)

    model = EvolutionMemoryModel(num_classes=4).to(DEVICE)
    model.load_state_dict(
        torch.load(weights_path, map_location=DEVICE, weights_only=True)
    )

    # ── CRITICAL: warm up the Episodic Memory Bank before any inference.
    # When weights are loaded fresh, the memory buffer is all zeros.
    # The decoder was trained with a *populated* bank; giving it zeros
    # causes extreme logit imbalance (100% confidence on a single class).
    # Warmup runs 25 synthetic images (all 4 themes) through the encoder
    # to populate the bank with realistic, diverse embeddings.
    model.warmup_memory()
    model.eval()

    with open("training_history.json", "r") as f:
        history = json.load(f)

    return model, history["t_hist"], history["v_hist"]


# ──────────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────────
def run_pipeline(model, theme_id: int):
    stream    = make_stream(theme_id)
    stream_in = stream.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(stream_in)
        probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_id   = int(np.argmax(probs))
    correct   = pred_id == theme_id
    return stream, pred_id, THEMES[pred_id]["caption"], THEMES[theme_id]["caption"], probs, correct


# ──────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────
def plot_stream(stream, theme_name):
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    fig.patch.set_facecolor("#141414")
    for t in range(5):
        ax = axes[t]
        ax.imshow(np.clip(stream[t].permute(1, 2, 0).numpy(), 0, 1))
        ax.set_title(f"T+{t}", fontsize=8, color="#888", fontfamily="monospace", pad=4)
        ax.axis("off")
    fig.suptitle(f"Theme: {theme_name}", fontsize=9, color="#7a6a50",
                 fontfamily="monospace", y=0.02)
    plt.tight_layout(pad=0.5)
    return fig


def plot_confidence(probs, pred_id):
    fig, ax = plt.subplots(figsize=(5, 2.6))
    fig.patch.set_facecolor("#141414")
    ax.set_facecolor("#141414")
    labels = [THEMES[i]["name"] for i in range(4)]
    colors = ["#c8975a" if i == pred_id else "#333" for i in range(4)]
    bars   = ax.barh(labels, probs, color=colors, height=0.5)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{p:.3f}", va="center", ha="left",
                fontsize=8, color="#aaa", fontfamily="monospace")
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", colors="#888", labelsize=8)
    plt.tight_layout(pad=0.6)
    return fig


def plot_loss(t_hist, v_hist):
    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor("#141414")
    ax.set_facecolor("#141414")
    epochs = range(1, len(t_hist) + 1)
    ax.plot(epochs, t_hist, color="#c8975a", marker="o", markersize=3, linewidth=1.5, label="Train")
    ax.plot(epochs, v_hist, color="#5a9a6a", marker="x", markersize=4, linewidth=1.5, label="Val")
    ax.set_xlabel("Epoch", fontsize=8, color="#666", fontfamily="monospace")
    ax.set_ylabel("Loss",  fontsize=8, color="#666", fontfamily="monospace")
    ax.tick_params(colors="#666", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a2a")
    legend = ax.legend(fontsize=7, framealpha=0)
    for txt in legend.get_texts():
        txt.set_color("#888")
    plt.tight_layout(pad=0.6)
    return fig


# ──────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────
st.markdown("""
<div class="evo-header">
    <div class="evo-label">NLP / Remote Sensing Research</div>
    <div class="evo-title">EVO-MEM</div>
    <div class="evo-sub">
        Evolutionary Memory-Augmented Satellite Image Captioning System
        &nbsp;|&nbsp; ViT + Episodic Memory Bank + Evolutionary Selector
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────────────
with st.spinner("Loading model..."):
    model, t_hist, v_hist = load_model_and_history()

# ──────────────────────────────────────────────────────
# STAT PILLS
# ──────────────────────────────────────────────────────
st.markdown(f"""
<div class="stat-row">
    <div class="stat-pill"><div class="sk">Encoder</div><div class="sv">ViT-B/16</div></div>
    <div class="stat-pill"><div class="sk">Memory Slots</div><div class="sv">100</div></div>
    <div class="stat-pill"><div class="sk">Mutation Rate</div><div class="sv">0.10</div></div>
    <div class="stat-pill"><div class="sk">Temporal Frames</div><div class="sv">T = 5</div></div>
    <div class="stat-pill"><div class="sk">Final Train Loss</div><div class="sv">{t_hist[-1]:.4f}</div></div>
    <div class="stat-pill"><div class="sk">Final Val Loss</div><div class="sv">{v_hist[-1]:.4f}</div></div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────
# LAYOUT
# ──────────────────────────────────────────────────────
left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown('<div class="sec-label">Select Scene Theme</div>', unsafe_allow_html=True)
    theme_names   = [THEMES[i]["name"] for i in range(4)]
    selected_name = st.radio("theme", theme_names, index=0, label_visibility="collapsed")
    selected_id   = theme_names.index(selected_name)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Learning Convergence</div>', unsafe_allow_html=True)
    st.pyplot(plot_loss(t_hist, v_hist), use_container_width=True)

with right:
    st.markdown('<div class="sec-label">System Demo</div>', unsafe_allow_html=True)
    st.markdown(
        "Select a land-cover theme on the left. The system generates a 5-frame temporal "
        "satellite image stream, passes it through the full EVO-MEM pipeline "
        "(ViT encoder → Episodic Memory Bank → Evolutionary Selector → Decoder), "
        "and produces a natural-language caption."
    )

    if st.button("RUN EVO-MEM PIPELINE"):
        with st.spinner("Running pipeline..."):
            stream, pred_id, pred_cap, truth_cap, probs, correct = run_pipeline(model, selected_id)

        st.markdown('<div class="sec-label">Temporal Image Stream</div>', unsafe_allow_html=True)
        st.pyplot(plot_stream(stream, selected_name), use_container_width=True)

        match_cls  = "match" if correct else "mismatch"
        match_text = "PREDICTION CORRECT" if correct else "PREDICTION MISMATCH — classifier returned alternate theme"
        st.markdown(f"""
        <div class="cap-box">
            <div class="ck">Model Output Caption</div>
            <div class="cv">{pred_cap}</div>
            <div class="cm {match_cls}">{match_text}</div>
        </div>
        """, unsafe_allow_html=True)

        gc, cc = st.columns(2, gap="medium")
        with gc:
            st.markdown('<div class="sec-label">Ground Truth</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="truth-box">
                <div class="tk">Expected Caption</div>
                <div class="tv">{truth_cap}</div>
                <div class="tt">Theme: {selected_name.upper()}</div>
            </div>
            """, unsafe_allow_html=True)
        with cc:
            st.markdown('<div class="sec-label">Class Confidence</div>', unsafe_allow_html=True)
            st.pyplot(plot_confidence(probs, pred_id), use_container_width=True)
