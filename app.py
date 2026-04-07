import os
import urllib.request
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import json
import pandas as pd

from evolution_memory import EvolutionSatelliteDataset, EvolutionMemoryModel, make_stream, THEMES

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EVO-MEM",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Minimal CSS strictly for maximizing screen real estate
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
</style>
""", unsafe_allow_html=True)

DEVICE = "cpu"

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_history():
    weights_path = "evo_mem_weights.pth"

    if not os.path.exists(weights_path):
        with st.spinner("Downloading model weights..."):
            url = "https://github.com/adityaxgupta/evo-mem-demo/releases/download/v1.0/evo_mem_weights.pth"
            urllib.request.urlretrieve(url, weights_path)

    model = EvolutionMemoryModel(num_classes=4).to(DEVICE)
    model.load_state_dict(
        torch.load(weights_path, map_location=DEVICE, weights_only=True)
    )

    model.warmup_memory()
    model.eval()

    with open("training_history.json", "r") as f:
        history = json.load(f)

    return model, history["t_hist"], history["v_hist"]

# ─────────────────────────────────────────────
# INFERENCE PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(model, theme_id: int):
    stream = make_stream(theme_id)
    stream_in = stream.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(stream_in)
        probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_id = int(np.argmax(probs))
    correct = pred_id == theme_id
    
    return stream, pred_id, THEMES[pred_id]["caption"], THEMES[theme_id]["caption"], probs, correct

# ─────────────────────────────────────────────
# UI LAYOUT
# ─────────────────────────────────────────────

# Header
st.title("EVO-MEM Architecture")
st.markdown("Evolutionary Memory-Augmented Satellite Image Captioning System")
st.divider()

with st.spinner("Initializing system..."):
    model, t_hist, v_hist = load_model_and_history()

# Core Metrics Dashboard
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Encoder", "ViT-B/16")
col2.metric("Memory Slots", "100")
col3.metric("Mutation Rate", "0.10")
col4.metric("Temporal Frames", "T = 5")
col5.metric("Final Train Loss", f"{t_hist[-1]:.4f}")
col6.metric("Final Val Loss", f"{v_hist[-1]:.4f}")

st.write("")

# Main Interface
left_col, right_col = st.columns([1, 2.5], gap="large")

with left_col:
    st.subheader("Configuration")
    theme_names = [THEMES[i]["name"] for i in range(4)]
    selected_name = st.radio("Select Scene Theme", theme_names)
    selected_id = theme_names.index(selected_name)

    st.write("")
    st.write("")

    st.subheader("Learning Convergence")
    loss_df = pd.DataFrame({
        "Train Loss": t_hist,
        "Val Loss": v_hist
    }, index=range(1, len(t_hist) + 1))
    
    st.line_chart(loss_df)

with right_col:
    st.subheader("Inference Demo")
    st.markdown("Generates a temporal satellite stream, processes via the ViT + Episodic Memory pipeline, and predicts the caption.")

    if st.button("Run EVO-MEM Pipeline", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            stream, pred_id, pred_cap, truth_cap, probs, correct = run_pipeline(model, selected_id)

        st.write("### Temporal Image Stream")
        img_cols = st.columns(5)
        for t in range(5):
            with img_cols[t]:
                img_np = np.clip(stream[t].permute(1, 2, 0).numpy(), 0, 1)
                st.image(img_np, caption=f"Frame T+{t}", use_column_width=True)

        st.write("---")

        if correct:
            st.success("Prediction Match: The model successfully identified the theme context.")
        else:
            st.error("Prediction Mismatch: The model selected an alternate theme context.")

        res_col1, res_col2 = st.columns(2, gap="large")

        with res_col1:
            st.markdown("#### Output Analysis")
            with st.container(border=True):
                st.markdown("**Expected Caption (Ground Truth)**")
                st.write(truth_cap)

            with st.container(border=True):
                st.markdown("**Model Output Caption**")
                st.write(pred_cap)

        with res_col2:
            st.markdown("#### Class Confidence")
            conf_df = pd.DataFrame({
                "Probability": probs
            }, index=[THEMES[i]["name"] for i in range(4)])
            
            st.bar_chart(conf_df)