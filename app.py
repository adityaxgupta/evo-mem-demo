import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import time
from evolution_memory import (
    EvolutionSatelliteDataset,
    EvolutionMemoryModel,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EVO-MEM | Satellite Caption System",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Research-grade dark aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=IBM+Plex+Sans:wght@300;400;500;600&family=Bebas+Neue&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    background-color: #0e0e0e;
    color: #d4cfc8;
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ── Top header bar ── */
.site-header {
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 1.4rem;
    margin-bottom: 2.4rem;
}
.site-header .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    color: #c97d3a;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.site-header h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.8rem;
    letter-spacing: 0.06em;
    color: #f0ebe3;
    margin: 0;
    line-height: 1;
}
.site-header .sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #6b6560;
    margin-top: 0.5rem;
    letter-spacing: 0.04em;
}

/* ── Section titles ── */
.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: #c97d3a;
    text-transform: uppercase;
    border-left: 2px solid #c97d3a;
    padding-left: 0.75rem;
    margin-bottom: 1.2rem;
}

/* ── Theme card selector ── */
.theme-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin-bottom: 1.6rem;
}
.theme-card {
    background: #161616;
    border: 1px solid #272727;
    padding: 1rem 1.2rem;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
}
.theme-card:hover { border-color: #c97d3a; background: #1c1c1c; }
.theme-card.active { border-color: #c97d3a; background: #1e1710; }
.theme-card .tc-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: #c97d3a;
    text-transform: uppercase;
}
.theme-card .tc-name {
    font-size: 1rem;
    font-weight: 600;
    color: #f0ebe3;
    margin-top: 0.2rem;
}
.theme-card .tc-desc {
    font-size: 0.78rem;
    color: #6b6560;
    margin-top: 0.3rem;
    line-height: 1.4;
}

/* ── Architecture panel ── */
.arch-panel {
    background: #111111;
    border: 1px solid #232323;
    padding: 1.6rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #7a7570;
    line-height: 2;
}
.arch-panel .arch-active { color: #c97d3a; }
.arch-panel .arch-node {
    padding: 0.2rem 0.6rem;
    border: 1px solid #2a2a2a;
    display: inline-block;
    margin: 0.1rem 0;
    color: #b0a89e;
}

/* ── Result box ── */
.result-box {
    background: #111111;
    border: 1px solid #2e2e2e;
    border-left: 3px solid #c97d3a;
    padding: 1.4rem 1.6rem;
    margin-top: 1.2rem;
}
.result-box .result-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    color: #5a5550;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-box .result-caption {
    font-size: 1.25rem;
    font-weight: 500;
    color: #f0ebe3;
    line-height: 1.4;
}
.result-box .result-match {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    margin-top: 0.8rem;
    color: #5a9e6f;
    letter-spacing: 0.05em;
}
.result-box .result-mismatch {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    margin-top: 0.8rem;
    color: #9e5a5a;
    letter-spacing: 0.05em;
}

/* ── Stat chips ── */
.stat-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.6rem;
    flex-wrap: wrap;
}
.stat-chip {
    background: #161616;
    border: 1px solid #272727;
    padding: 0.6rem 1rem;
    flex: 1;
}
.stat-chip .sc-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    color: #5a5550;
    text-transform: uppercase;
}
.stat-chip .sc-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    color: #c97d3a;
    margin-top: 0.2rem;
}

/* ── Divider ── */
.thin-divider {
    border: none;
    border-top: 1px solid #1e1e1e;
    margin: 2rem 0;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label { color: #7a7570 !important; font-size: 0.8rem !important; }

.stButton > button {
    background: #c97d3a !important;
    color: #0e0e0e !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    font-weight: 500 !important;
}
.stButton > button:hover {
    background: #d98f50 !important;
}

.stSelectbox > div > div {
    background: #161616 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 0 !important;
    color: #d4cfc8 !important;
}

div[data-testid="stStatusWidget"] { display: none; }

/* progress bar */
.stProgress > div > div > div > div {
    background: #c97d3a !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL CACHE — load once
# ─────────────────────────────────────────────
DEVICE = "cpu"  # Streamlit Cloud has no GPU

THEMES = {
    0: {
        "name": "Urban",
        "tag": "Land Use",
        "desc": "Urban sprawl and infrastructure growth",
        "caption": "Urban expansion seen in the northern sector",
    },
    1: {
        "name": "Forest",
        "tag": "Vegetation",
        "desc": "Canopy loss and deforestation activity",
        "caption": "Deforestation occurring in the rainforest",
    },
    2: {
        "name": "Water",
        "tag": "Hydrology",
        "desc": "River overflow and seasonal flooding",
        "caption": "River levels rising due to seasonal floods",
    },
    3: {
        "name": "Desert",
        "tag": "Arid Zone",
        "desc": "Construction activity on sandy terrain",
        "caption": "New residential complex built on sandy terrain",
    },
}

@st.cache_resource(show_spinner=False)
def load_model():
    model = EvolutionMemoryModel(num_classes=4).to(DEVICE)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_dataset():
    return EvolutionSatelliteDataset()


def generate_stream_for_theme(theme_id, dataset_obj):
    theme_info = dataset_obj.themes_data[theme_id]
    stream = []
    for t in range(5):
        img = np.full((224, 224, 3), theme_info["base"], dtype=np.float32)
        num_blobs = 2 + (t * 3)
        for _ in range(num_blobs):
            x, y = random.randint(0, 180), random.randint(0, 180)
            size = random.randint(15, 40)
            img[x:x + size, y:y + size] = theme_info["growth"]
        img += np.random.uniform(-0.05, 0.05, (224, 224, 3))
        stream.append(torch.from_numpy(np.clip(img, 0, 1)).permute(2, 0, 1))
    return torch.stack(stream)


def run_prediction(model, test_stream):
    with torch.no_grad():
        inp = test_stream.unsqueeze(0).to(DEVICE)
        output = model(inp)
        pred_id = torch.argmax(output, dim=1).item()
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
    return pred_id, probs


def plot_stream(test_stream, theme_id):
    theme = THEMES[theme_id]
    colors = {0: "#c97d3a", 1: "#5a9e6f", 2: "#3a7ec9", 3: "#c9b83a"}
    accent = colors[theme_id]

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.2))
    fig.patch.set_facecolor("#0e0e0e")

    for i, ax in enumerate(axes):
        img = test_stream[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_facecolor("#0e0e0e")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(accent)
            spine.set_linewidth(1.5)
        ax.set_xlabel(
            f"T+{i}",
            color="#6b6560",
            fontsize=9,
            fontfamily="monospace",
            labelpad=6,
        )

    fig.text(
        0.5, 1.01,
        f"TEMPORAL IMAGE STREAM  //  THEME: {theme['name'].upper()}",
        ha="center", va="bottom",
        color="#5a5550", fontsize=8, fontfamily="monospace",
    )
    plt.tight_layout(pad=0.4)
    return fig


def plot_confidence(probs):
    fig, ax = plt.subplots(figsize=(5, 2.6))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    labels = [THEMES[i]["name"] for i in range(4)]
    colors = ["#c97d3a" if p == max(probs) else "#2a2a2a" for p in probs]
    bars = ax.barh(labels, probs, color=colors, height=0.5)

    for bar, prob in zip(bars, probs):
        ax.text(
            min(prob + 0.01, 0.98), bar.get_y() + bar.get_height() / 2,
            f"{prob:.3f}",
            va="center", ha="left",
            color="#6b6560", fontsize=8, fontfamily="monospace",
        )

    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Confidence", color="#4a4540", fontsize=8, fontfamily="monospace")
    ax.tick_params(colors="#5a5550", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a2a")
    ax.xaxis.label.set_color("#4a4540")
    plt.tight_layout(pad=0.6)
    return fig


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="site-header">
    <div class="label">NLP / Remote Sensing Research</div>
    <h1>EVO-MEM</h1>
    <div class="sub">Evolutionary Memory-Augmented Satellite Image Captioning System &nbsp;|&nbsp; ViT + Episodic Memory Bank + Evolutionary Selector</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LAYOUT: two columns
# ─────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.6], gap="large")

# ══════════════════════════════════════════════
# LEFT — Controls
# ══════════════════════════════════════════════
with left_col:

    st.markdown('<div class="section-title">Select Scene Theme</div>', unsafe_allow_html=True)

    # Theme radio disguised as cards
    theme_choice = st.radio(
        label="theme_select",
        options=[0, 1, 2, 3],
        format_func=lambda x: THEMES[x]["name"],
        label_visibility="collapsed",
        horizontal=False,
    )

    # Render visual theme cards below radio (decorative info)
    for tid, t in THEMES.items():
        active_cls = "active" if tid == theme_choice else ""
        st.markdown(f"""
        <div class="theme-card {active_cls}" style="margin-bottom:0.5rem">
            <div class="tc-label">{t['tag']}</div>
            <div class="tc-name">{t['name']}</div>
            <div class="tc-desc">{t['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Architecture Flow</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="arch-panel">
        <div class="arch-node arch-active">Input: Temporal Image Stream (T=5)</div><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&darr;<br>
        <div class="arch-node">Vision Transformer Encoder (ViT-B/16)</div><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&darr;<br>
        <div class="arch-node">Feature Embeddings (768-dim)</div><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&darr;<br>
        <div class="arch-node">Episodic Memory Bank (EMB)</div><br>
        &nbsp;Write &nbsp;&nbsp;&darr;&nbsp;&nbsp; Read &nbsp;&nbsp;&darr;&nbsp;&nbsp; Rewrite<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&darr;<br>
        <div class="arch-node">Evolutionary Selector (mutation + selection)</div><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&darr;<br>
        <div class="arch-node">Memory-Augmented Decoder (MLP)</div><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&darr;<br>
        <div class="arch-node arch-active">Caption Output</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# RIGHT — Demo + Results
# ══════════════════════════════════════════════
with right_col:

    st.markdown('<div class="section-title">System Demo</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.8rem; color:#5a5550; line-height:1.6; margin-bottom:1.4rem; font-family:'IBM Plex Mono',monospace;">
    Select a land-cover theme on the left. The system will generate a 5-frame temporal
    satellite image stream for that theme, pass it through the full EVO-MEM pipeline,
    and produce a natural-language caption.
    </div>
    """, unsafe_allow_html=True)

    # Stat row
    st.markdown("""
    <div class="stat-row">
        <div class="stat-chip">
            <div class="sc-label">Encoder</div>
            <div class="sc-val">ViT-B/16</div>
        </div>
        <div class="stat-chip">
            <div class="sc-label">Memory Capacity</div>
            <div class="sc-val">100 slots</div>
        </div>
        <div class="stat-chip">
            <div class="sc-label">Mutation Rate</div>
            <div class="sc-val">0.10</div>
        </div>
        <div class="stat-chip">
            <div class="sc-label">Temporal Frames</div>
            <div class="sc-val">T = 5</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    run_btn = st.button("Run EVO-MEM Pipeline")

    if run_btn:
        with st.spinner("Loading model weights..."):
            model = load_model()
            dataset_obj = load_dataset()

        progress_bar = st.progress(0)

        with st.spinner("Generating temporal image stream..."):
            time.sleep(0.3)
            progress_bar.progress(20)
            test_stream = generate_stream_for_theme(theme_choice, dataset_obj)

        with st.spinner("Encoding with ViT..."):
            time.sleep(0.3)
            progress_bar.progress(50)

        with st.spinner("Running Evolutionary Selector + Memory Bank..."):
            time.sleep(0.3)
            progress_bar.progress(75)
            pred_id, probs = run_prediction(model, test_stream)

        progress_bar.progress(100)
        time.sleep(0.2)
        progress_bar.empty()

        # ── Image Stream ──
        st.markdown('<div class="section-title" style="margin-top:1.2rem">Temporal Image Stream</div>', unsafe_allow_html=True)
        fig_stream = plot_stream(test_stream, theme_choice)
        st.pyplot(fig_stream, use_container_width=True)
        plt.close(fig_stream)

        # ── Result ──
        target_caption = THEMES[theme_choice]["caption"]
        predicted_caption = THEMES[pred_id]["caption"]
        match = pred_id == theme_choice
        match_html = (
            '<div class="result-match">PREDICTION CORRECT — theme classification matched ground truth</div>'
            if match else
            '<div class="result-mismatch">PREDICTION MISMATCH — classifier returned alternate theme</div>'
        )

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Model Output Caption</div>
            <div class="result-caption">{predicted_caption}</div>
            {match_html}
        </div>
        """, unsafe_allow_html=True)

        # ── Two columns: ground truth + confidence ──
        rc1, rc2 = st.columns([1, 1], gap="medium")

        with rc1:
            st.markdown('<div class="section-title" style="margin-top:1.4rem">Ground Truth</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#111; border:1px solid #232323; padding:1rem 1.2rem;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.6rem;
                            letter-spacing:0.15em; color:#5a5550; text-transform:uppercase;
                            margin-bottom:0.4rem;">Expected Caption</div>
                <div style="font-size:0.88rem; color:#b0a89e; line-height:1.5;">{target_caption}</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                            color:#5a5550; margin-top:0.8rem;">Theme: {THEMES[theme_choice]['name'].upper()}</div>
            </div>
            """, unsafe_allow_html=True)

        with rc2:
            st.markdown('<div class="section-title" style="margin-top:1.4rem">Class Confidence</div>', unsafe_allow_html=True)
            fig_conf = plot_confidence(probs)
            st.pyplot(fig_conf, use_container_width=True)
            plt.close(fig_conf)

    else:
        # Placeholder before run
        st.markdown("""
        <div style="border:1px dashed #232323; padding:3rem 2rem; text-align:center; margin-top:1rem;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
                        letter-spacing:0.15em; color:#3a3530; text-transform:uppercase;">
                Select a theme and press Run to begin
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)
st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem;
            color:#2e2e2e; text-align:center; letter-spacing:0.1em; padding-bottom:1rem;">
    EVO-MEM &nbsp;|&nbsp; Evolutionary Memory-Augmented Image Captioning
    &nbsp;|&nbsp; ViT + EMB + Evolutionary Selector &nbsp;|&nbsp; Research Prototype
</div>
""", unsafe_allow_html=True)
