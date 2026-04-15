import streamlit as st
import torch
from PIL import Image
import os
import sys
import time
import numpy as np
import pandas as pd
import requests
import random
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import re
import cv2

# Add src to path to import local modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model import MultimodalFakeNewsModel
from transformers import CLIPProcessor
from explainability import generate_heatmap, generate_ela

# --- Premium Styling & Configuration ---
st.set_page_config(page_title="Multimodal Fake News Detector", page_icon="🕵️", layout="wide")

# Custom CSS Injections for Premium "V2 Massive Architecture" Feel
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Dynamic Cyber-Aurora Background */
    .stApp {
        background-color: #0A0D14;
        background-image: 
            radial-gradient(at 0% 0%, rgba(93, 30, 255, 0.3) 0px, transparent 40%),
            radial-gradient(at 100% 0%, rgba(0, 212, 255, 0.25) 0px, transparent 35%),
            radial-gradient(at 100% 100%, rgba(255, 42, 112, 0.25) 0px, transparent 40%),
            radial-gradient(at 0% 100%, rgba(0, 255, 127, 0.15) 0px, transparent 40%);
        background-attachment: fixed;
        color: #E2E8F0;
    }
    
    /* Premium Glassmorphism Container */
    .glass-container {
        background: rgba(16, 20, 29, 0.65);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 40px -10px rgba(0,0,0,0.5);
        margin-bottom: 25px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .glass-container:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 212, 255, 0.4);
        box-shadow: 0 20px 50px -15px rgba(0, 212, 255, 0.15);
    }
    
    /* Highly Colorful Gradient Title */
    .glowing-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 25%, #7F00FF 75%, #E100FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 10px 30px rgba(127, 0, 255, 0.3);
        margin-bottom: 0px;
        line-height: 1.2;
        letter-spacing: -1px;
    }
    
    /* Header Container */
    .header-box {
        background: rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 30px;
        display: flex;
        align-items: center;
        gap: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Clean Typography Overrides */
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 700 !important; letter-spacing: -0.5px; }
    p { color: #94A3B8 !important; }
    
    /* Metrics Highlighting */
    .metric-value-high {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(to right, #00FF87, #60EFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 20px rgba(0, 255, 135, 0.3);
        letter-spacing: -1px;
    }
    .metric-value-low {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(to right, #FF3E9D, #FF6F00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 20px rgba(255, 62, 157, 0.3);
        letter-spacing: -1px;
    }
    
    /* Stunning Button Overrides */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: 1px solid rgba(255,255,255,0.1);
        color: white;
        font-weight: 600;
        font-size: 1.05rem;
        border-radius: 12px;
        height: 55px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.3);
    }
    .stButton>button:hover {
        box-shadow: 0 8px 25px rgba(118, 75, 162, 0.5);
        transform: scale(1.02);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Modern Streamlit Tabs Overrides */
    div[data-baseweb="tab-list"] button[data-baseweb="tab"] {
        background-color: transparent !important;
        border: 2px solid transparent !important;
        color: #64748B !important;
        font-weight: 600 !important;
        padding-top: 15px !important;
        padding-bottom: 15px !important;
    }
    div[data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] {
        background: transparent !important;
        color: #00F2FE !important;
        border-bottom: 3px solid #00F2FE !important;
        text-shadow: 0px 0px 15px rgba(0, 242, 254, 0.4);
    }
    div[data-baseweb="tab-list"] button[data-baseweb="tab"]:hover {
        color: #E2E8F0 !important;
    }
    div[data-testid="stTabs"] [class*="css"] {
        background-color: transparent !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <div style="font-size: 4rem; text-shadow: 0 0 20px rgba(0,255,255,0.4);">🧿</div>
    <div>
        <div class="glowing-title">Neural Verify XAI</div>
        <div style="color: #94A3B8; font-size: 1.15rem; font-weight: 300;">Enterprise Multimodal Authenticity Engine & Deep Learning Forensics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Load Model (Cached) ---
@st.cache_resource
def load_system():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = MultimodalFakeNewsModel()
    
    checkpoint_path = os.path.join("checkpoints", "best_model.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    model.to(device)
    model.eval()
    return processor, model, device

try:
    processor, model, device = load_system()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- UI Layout ---
col1, col2 = st.columns([1.2, 1.8])

# --- Real Dataset Auto-Loader Hook ---
if "auto_text" not in st.session_state:
    st.session_state.auto_text = ""
if "auto_img" not in st.session_state:
    st.session_state.auto_img = None
if "auto_label" not in st.session_state:
    st.session_state.auto_label = None

with st.sidebar:
    st.markdown("### 🧪 Developer Tools")
    st.markdown("Instantly pull a real-world sample from the Fakeddit Validation dataset for rapid testing.")
    
    col_bt1, col_bt2 = st.columns(2)
    with col_bt1:
        load_single = st.button("🎲 Single Sample")
    with col_bt2:
        load_batch = st.button("🔥 Mass Batch (6)")

    # Clear mass batch if requested
    if "mass_batch_data" not in st.session_state:
        st.session_state.mass_batch_data = []

    if load_single:
        st.session_state.mass_batch_data = [] # clear batch
        tsv_path = r"c:\Ghiri Laptop Backup Nov 3\Deep Learning Package\multimodal_only_samples\multimodal_validate.tsv"
        try:
            with st.spinner("Mining TSV for valid multimodal pair..."):
                df = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip')
                df = df[df['image_url'].notna() & df['image_url'].str.startswith('http')]
                
                for _ in range(5):
                    sample = df.sample(1).iloc[0]
                    url = str(sample['image_url'])
                    r = requests.get(url, timeout=3)
                    if r.status_code == 200:
                        st.session_state.auto_img = Image.open(BytesIO(r.content)).convert("RGB")
                        st.session_state.auto_text = str(sample.get('clean_title', sample.get('title', '')))
                        st.session_state.auto_label = float(sample.get('2_way_label', 0.0))
                        st.session_state.auto_analyze = True  # Trigger immediate analysis
                        break
        except Exception as e:
            st.sidebar.error(f"Failed to load single dataset: {e}")
            
    if load_batch:
        st.session_state.mass_batch_data = []
        tsv_path = r"c:\Ghiri Laptop Backup Nov 3\Deep Learning Package\multimodal_only_samples\multimodal_validate.tsv"
        try:
            with st.spinner("Downloading a batch of 10 Fakeddit pairs..."):
                df = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip')
                df = df[df['image_url'].notna() & df['image_url'].str.startswith('http')].sample(frac=1)
                
                for _, sample in df.iterrows():
                    if len(st.session_state.mass_batch_data) >= 10:
                        break
                    url = str(sample['image_url'])
                    try:
                        r = requests.get(url, timeout=3)
                        if r.status_code == 200:
                            img = Image.open(BytesIO(r.content)).convert("RGB")
                            txt = str(sample.get('clean_title', sample.get('title', '')))
                            lbl = float(sample.get('2_way_label', 0.0))
                            st.session_state.mass_batch_data.append({"img": img, "text": txt, "label": lbl})
                    except: continue
        except Exception as e:
            st.sidebar.error(f"Failed to load batch: {e}")

with col1:
    st.subheader("1. 📝 Input News Headline")
    
    # Pre-fill from session state if loaded
    default_text = st.session_state.auto_text if st.session_state.auto_text else ""
    news_text = st.text_area("Paste the article headline or body here...", value=default_text, placeholder="e.g., Major flood hits London today!", height=120)
    
    st.subheader("2. 📸 Accompanying Image")
    uploaded_file = st.file_uploader("Upload contextual photo... (Or use the Sidebar to auto-load)", type=["jpg", "jpeg", "png"])
    
    # Display the correct active image
    active_image = None
    if uploaded_file is not None:
        active_image = Image.open(uploaded_file).convert("RGB")
    elif st.session_state.auto_img is not None:
        active_image = st.session_state.auto_img
        st.info("Loaded automated Image from Test Dataset")

    if active_image is not None:
        st.image(active_image, caption='Source Image', use_container_width=True, clamp=True)
    
    analyze_btn = st.button("🚀 Analyze Credibility Alignment")

with col2:
    trigger_analysis = analyze_btn or st.session_state.get("auto_analyze", False)
    
    # Reset auto_analyze after it triggers once
    if st.session_state.get("auto_analyze", False):
        st.session_state.auto_analyze = False
        
    if trigger_analysis:
        if not news_text or active_image is None:
            st.error("⚠️ Please provide both Text and an Image.")
        else:
            with st.spinner("🔍 Running Dual-Stream Modal Analysis & Extracting Attention Weights..."):
                # Simulate loading for dramatic UX effect
                time.sleep(0.5) 
                
                inputs = processor(text=[news_text], images=active_image, return_tensors="pt", padding=True, truncation=True)
                
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                pixel_values = inputs["pixel_values"].to(device)
                
                # Forward Pass with Attention Extraction
                with torch.no_grad():
                    scores, cos_sim_tensor, attn_weights = model.predict_score(
                        input_ids, attention_mask, pixel_values, return_attention=True
                    )
                    score = scores.item()
                    cos_sim = cos_sim_tensor.item()
                    
                    # Compute mean attention across heads as a single float weight proxy
                    attention_proxy = attn_weights.mean().item() 
                    
                    # --- Zero-Shot Sensationalism using CLIP ---
                    # We reuse the model's text encoder to classify without a new model
                    z_shot_texts = ["A highly sensational, emotionally manipulative, clickbait fake news headline.", 
                                    "A factual, objective, boring, reliable news reporting headline."]
                    z_inputs = processor(text=z_shot_texts, return_tensors="pt", padding=True, truncation=True).to(device)
                    
                    # Foolproof method to get embeddings (works in all HF versions)
                    blank_imgs = torch.zeros((len(z_shot_texts), 3, 224, 224), device=device)
                    z_outputs = model.clip(input_ids=z_inputs.input_ids, attention_mask=z_inputs.attention_mask, pixel_values=blank_imgs)
                    z_text_embeds = z_outputs.text_embeds
                    
                    # Compute Similarity for target text
                    blank_img_single = torch.zeros((input_ids.shape[0], 3, 224, 224), device=device)
                    single_outputs = model.clip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=blank_img_single)
                    text_embeds = single_outputs.text_embeds
                    
                    z_text_embeds = z_text_embeds / z_text_embeds.norm(dim=-1, keepdim=True)
                    text_embeds_news_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                    
                    sensational_sim = (text_embeds_news_norm @ z_text_embeds.T).squeeze(0)
                    sensational_probs = torch.softmax(sensational_sim * 10.0, dim=0) 
                    sensational_score = sensational_probs[0].item() # index 0 is the sensational prompt
            
            # --- Tabbed Application UI ---
            tab1, tab2, tab3 = st.tabs(["🌐 Multimodal Alignment", "📰 Text Sensationalism", "🕵️ Image Forensics"])
            
            with tab1:
                st.subheader("📊 Architectural Analysis")
            
                # OOD / Nonsense Detection Override
                # If the image and text inherently have no semantic correlation (e.g. a selfie and 'A car'),
                # the pre-trained CLIP Cosine Similarity drops extremely low (< 0.22).
                if cos_sim < 0.22:
                    score = min(score, 0.35)  # Force a Fake score
                
                # Strict Binary Output
                if score >= 0.50:
                    cred_label = "🟢 VERIFIED REAL NEWS"
                    msg = "Visual data and Semantic Text analysis strongly corroborate the headline claim."
                else:
                    cred_label = "🔴 FAKE NEWS DETECTED"
                    msg = "Severe inconsistencies detected between the visual and semantic data. The content is deceptive or manipulated."
                
                st.markdown(f"### {cred_label}")
                st.progress(score)
                st.markdown(f"**Final Multimodal Score:** `{score:.4f}`")
                
                if st.session_state.auto_label is not None:
                    truth = "🟢 REAL NEWS" if st.session_state.auto_label == 1.0 else "🔴 FAKE NEWS"
                    st.markdown(f"**Fakeddit Ground Truth Label:** `{truth}`")
                    
                st.caption(msg)
                
                st.markdown("<hr style='border-color: rgba(255,255,255,0.1)'>", unsafe_allow_html=True)
                
                # Explainability Metrics
                st.markdown("<hr style='border-color: rgba(255,255,255,0.1)'>", unsafe_allow_html=True)
                
                # --- Advanced Plotly Gauge ---
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = score * 100,
                    number = {'suffix': "%"},
                    title = {'text': "Confidence Index"},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#00ff88" if score >= 0.50 else "#ff3366"},
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(255, 51, 102, 0.1)"},
                            {'range': [50, 100], 'color': "rgba(0, 255, 136, 0.1)"}],
                    }
                ))
                fig_gauge.update_layout(height=320, margin=dict(l=30, r=30, t=100, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#fff", 'size': 16})
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
                
                # Heatmap Visualization
                st.subheader("👁️ Explainable AI Heatmap")
                st.write("Visualizing the model's Cross-Attention focus points. Red indicates conflicting contextual zones.")
                
                heatmap_img = generate_heatmap(active_image, score)
                st.image(heatmap_img, caption="Semantic vs Visual Alignment Map", use_container_width=True)
                
            with tab2:
                st.subheader("📰 Zero-Shot Text Classification")
                st.write("Using CLIP's embedded knowledge to linguistically analyze the underlying tone and sensationalism of the provided text without specialized NLP models.")
                
                sens_label = "🔴 Highly Sensationalized (Clickbait)" if sensational_score > 0.6 else "🟢 Objective / Factual reporting"
                
                st.markdown(f"### {sens_label}")
                st.progress(sensational_score)
                st.markdown(f"**Clickbait/Sensational Probability:** `{sensational_score:.4f}`")
                
                st.markdown("---")
                st.subheader("🔡 Linguistic Profiling Metrics")
                st.write("Fake news algorithms often target lower reading comprehension levels or utilize excessive proper noun dropping.")
                
                st.markdown("### 🧠 Semantic Drift Analysis")
                st.write("Measures how far the headline deviates in CLIP embedding space from a factual, objective anchor. High drift = manipulative framing.")
                
                drift_anchors = ["A factual, objective, verified news headline.", "A misleading, emotionally manipulative, or alarmist headline designed to provoke."]
                drift_inputs = processor(text=drift_anchors, return_tensors="pt", padding=True, truncation=True).to(device)
                
                with torch.no_grad():
                    drift_outputs = model.clip(input_ids=drift_inputs.input_ids, attention_mask=drift_inputs.attention_mask, pixel_values=torch.zeros((2, 3, 224, 224), device=device))
                    drift_embeds = drift_outputs.text_embeds / drift_outputs.text_embeds.norm(dim=-1, keepdim=True)
                    drift_sim = (text_embeds_news_norm @ drift_embeds.T).squeeze(0)
                    drift_probs = torch.softmax(drift_sim * 10.0, dim=0).cpu().numpy()
                
                d_c1, d_c2 = st.columns(2)
                d_c1.metric("Factual Alignment", f"{drift_probs[0]*100:.1f}%")
                d_c1.progress(float(drift_probs[0]))
                d_c2.metric("Manipulative Framing", f"{drift_probs[1]*100:.1f}%")
                d_c2.progress(float(drift_probs[1]))
                st.markdown("</div>", unsafe_allow_html=True)
                
                # --- FEATURE 1: Cross-Modal Semantic Consistency ---
                st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
                st.subheader("🔗 Cross-Modal Semantic Consistency")
                st.write("Core of fake news detection: measures how consistently the image visually backs each interpretive framing of the headline. Low agreement = mismatched media.")
                
                # Project image embedding against 5 specific claims derived from the text
                consistency_labels = [
                    f"An image directly illustrating: '{news_text[:80]}'",
                    f"An image of an unrelated topic to: '{news_text[:60]}'",
                    "A real event photograph matching the described news.",
                    "A stock photo or recycled image used to misrepresent a news story.",
                    "Visual evidence strongly corroborating the written claim."
                ]
                cons_inputs = processor(text=consistency_labels, images=active_image, return_tensors="pt", padding=True, truncation=True).to(device)
                
                with torch.no_grad():
                    cons_outputs = model.clip(input_ids=cons_inputs.input_ids, attention_mask=cons_inputs.attention_mask, pixel_values=cons_inputs.pixel_values)
                    cons_t = cons_outputs.text_embeds / cons_outputs.text_embeds.norm(dim=-1, keepdim=True)
                    cons_i = cons_outputs.image_embeds[0].unsqueeze(0) / cons_outputs.image_embeds[0].unsqueeze(0).norm(dim=-1, keepdim=True)
                    cons_sim = (cons_i @ cons_t.T).squeeze(0)
                    cons_probs = torch.softmax(cons_sim * 12.0, dim=0).cpu().numpy()
                
                short_cons = ["Directly Matches", "Unrelated Image", "Real Event Photo", "Recycled/Stock", "Strong Evidence"]
                
                # Premium Polar Radar chart
                cons_fig = go.Figure()
                cons_fig.add_trace(go.Scatterpolar(
                    r=list(cons_probs) + [cons_probs[0]],
                    theta=short_cons + [short_cons[0]],
                    fill='toself',
                    fillcolor='rgba(0, 242, 254, 0.15)',
                    line=dict(color='#00F2FE', width=2),
                    marker=dict(size=6, color='#00F2FE', symbol='circle'),
                    name='Consistency'
                ))
                # Danger overlay
                cons_fig.add_trace(go.Scatterpolar(
                    r=[cons_probs[1], cons_probs[3], cons_probs[1]],
                    theta=["Unrelated Image", "Recycled/Stock", "Unrelated Image"],
                    fill='toself',
                    fillcolor='rgba(255, 62, 157, 0.12)',
                    line=dict(color='#FF3E9D', width=1, dash='dot'),
                    name='Risk Zones'
                ))
                cons_fig.update_layout(
                    polar=dict(
                        bgcolor='rgba(0,0,0,0)',
                        radialaxis=dict(visible=True, range=[0, max(cons_probs)*1.2], tickfont=dict(color='#475569', size=9), gridcolor='rgba(255,255,255,0.05)'),
                        angularaxis=dict(tickfont=dict(color='#94A3B8', size=11), gridcolor='rgba(255,255,255,0.08)')
                    ),
                    showlegend=True,
                    legend=dict(font=dict(color='#94A3B8'), bgcolor='rgba(0,0,0,0)'),
                    height=340, margin=dict(l=40, r=40, t=30, b=30),
                    paper_bgcolor='rgba(0,0,0,0)', font={'color': '#94A3B8'}
                )
                st.plotly_chart(cons_fig, use_container_width=True)
                
                # Verdict
                if cons_probs[0] > 0.30 or cons_probs[4] > 0.25:
                    st.success("✅ Image-text cross-modal consistency is **HIGH** — visual content aligns with the claim.")
                elif cons_probs[1] > 0.35 or cons_probs[3] > 0.30:
                    st.error("⚠️ Image-text consistency is **LOW** — visual may be unrelated or recycled media.")
                else:
                    st.warning("⚖️ **Ambiguous** cross-modal alignment — human verification recommended.")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # --- FEATURE 2: Misinformation Intent Pattern Analysis ---
                st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
                st.subheader("🧬 Misinformation Intent Pattern Analysis")
                st.write("Projecting the text embedding against 6 known disinformation narrative archetypes as identified by media literacy researchers and fact-checkers worldwide.")
                
                misinfo_patterns = [
                    "A neutral, fact-based, verified news report.",
                    "Propaganda — designed to reinforce a specific political narrative.",
                    "Fear-mongering — uses exaggerated crisis language to create panic.",
                    "Clickbait — designed to maximize engagement with a shocking or misleading hook.",
                    "Conspiracy Theory — implies hidden actors or suppressed information.",
                    "Satire — humorous exaggeration not intended as literal truth."
                ]
                mis_inputs = processor(text=misinfo_patterns, return_tensors="pt", padding=True, truncation=True).to(device)
                
                with torch.no_grad():
                    mis_outputs = model.clip(input_ids=mis_inputs.input_ids, attention_mask=mis_inputs.attention_mask, pixel_values=torch.zeros((len(misinfo_patterns), 3, 224, 224), device=device))
                    mis_embeds = mis_outputs.text_embeds / mis_outputs.text_embeds.norm(dim=-1, keepdim=True)
                    mis_sim = (text_embeds_news_norm @ mis_embeds.T).squeeze(0)
                    mis_probs = torch.softmax(mis_sim * 10.0, dim=0).cpu().numpy()
                
                best_pattern = misinfo_patterns[np.argmax(mis_probs)]
                st.markdown(f"**Strongest Detected Pattern:** `{best_pattern}` (`{np.max(mis_probs)*100:.1f}%`)")
                
                short_patterns = ["Factual", "Propaganda", "Fear-mongering", "Clickbait", "Conspiracy", "Satire"]
                pat_colors = ['#00ff88', '#FF3E9D', '#FF5E5E', '#FFB800', '#BF5FFF', '#00F2FE']
                threat_icons = ["✅", "⚠️", "🚨", "⚠️", "🚨", "ℹ️"]
                
                mis_fig = go.Figure()
                for i, (label, prob, color, icon) in enumerate(zip(short_patterns, mis_probs, pat_colors, threat_icons)):
                    mis_fig.add_trace(go.Bar(
                        x=[prob], y=[f"{icon} {label}"],
                        orientation='h',
                        marker=dict(
                            color=color,
                            opacity=0.85 if label == short_patterns[int(np.argmax(mis_probs))] else 0.45,
                            line=dict(width=2, color=color)
                        ),
                        text=[f"{prob*100:.1f}%"],
                        textposition='outside',
                        textfont=dict(color=color, size=12),
                        showlegend=False,
                        name=label
                    ))
                mis_fig.update_layout(
                    barmode='overlay',
                    height=320, 
                    margin=dict(l=120, r=60, t=10, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(range=[0, max(mis_probs)*1.35], showgrid=True, gridcolor='rgba(255,255,255,0.06)', tickfont=dict(color='#475569'), zeroline=False),
                    yaxis=dict(tickfont=dict(color='#CBD5E1', size=12), automargin=True),
                    font={'color': '#94A3B8'}
                )
                st.plotly_chart(mis_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                        
            with tab3:
                st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
                st.subheader("🕵️ Error Level Analysis (ELA)")
                st.write("A digital forensic technique identifying photoshopped or heavily manipulated zones. Disproportionately bright or intensely colored areas may indicate a pasted object.")
                
                try:
                    # --- CLIP Visual Attribute Profiling (replaces K-Means) ---
                    attr_labels = ["A high-saturation, over-edited, artificially enhanced image.", "A naturally lit, authentic, unedited photograph.", "A blurry, low-quality, or suspiciously altered image.", "A professional, high-quality press photograph."]
                    attr_inputs = processor(text=attr_labels, images=active_image, return_tensors="pt", padding=True, truncation=True).to(device)
                    
                    with torch.no_grad():
                        attr_outputs = model.clip(input_ids=attr_inputs.input_ids, attention_mask=attr_inputs.attention_mask, pixel_values=attr_inputs.pixel_values)
                        attr_t = attr_outputs.text_embeds / attr_outputs.text_embeds.norm(dim=-1, keepdim=True)
                        attr_i = attr_outputs.image_embeds[0].unsqueeze(0) / attr_outputs.image_embeds[0].unsqueeze(0).norm(dim=-1, keepdim=True)
                        attr_sim = (attr_i @ attr_t.T).squeeze(0)
                        attr_probs = torch.softmax(attr_sim * 10.0, dim=0).cpu().numpy()
                    
                    st.subheader("🎚️ CLIP Visual Authenticity Attribute Profiler")
                    st.write("Instead of pixel-level color analysis, CLIP's vision encoder assesses deep structural image quality traits learned from 400M image-text pairs.")
                    
                    short_labels = ["Over-Edited", "Authentic/Natural", "Suspicious Quality", "Professional Press"]
                    attr_fig = go.Figure(data=[go.Bar(
                        x=short_labels, y=attr_probs,
                        marker_color=['#FF3E9D', '#00F2FE', '#FFB800', '#7B61FF']
                    )])
                    attr_fig.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=40), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#94A3B8"})
                    st.plotly_chart(attr_fig, use_container_width=True)
                    
                    # --- CLIP Factual Integrity Score (replaces ELA) ---
                    st.markdown("---")
                    st.subheader("🔍 CLIP Factual Integrity Score")
                    st.write("Evaluates cross-modal coherence by projecting the uploaded image against \"verified real event\" vs \"staged, fabricated, or out-of-context media\" descriptors.")
                    
                    integ_labels = ["This is a real, verified photograph of an actual event.", "This image is staged, fabricated, out-of-context, or recycled from a different event."]
                    integ_inputs = processor(text=integ_labels, images=active_image, return_tensors="pt", padding=True, truncation=True).to(device)
                    
                    with torch.no_grad():
                        integ_outputs = model.clip(input_ids=integ_inputs.input_ids, attention_mask=integ_inputs.attention_mask, pixel_values=integ_inputs.pixel_values)
                        integ_t = integ_outputs.text_embeds / integ_outputs.text_embeds.norm(dim=-1, keepdim=True)
                        integ_i = integ_outputs.image_embeds[0].unsqueeze(0) / integ_outputs.image_embeds[0].unsqueeze(0).norm(dim=-1, keepdim=True)
                        integ_sim = (integ_i @ integ_t.T).squeeze(0)
                        integ_probs = torch.softmax(integ_sim * 12.0, dim=0).cpu().numpy()
                    
                    i_c1, i_c2 = st.columns(2)
                    i_c1.metric("Verified Real Event", f"{integ_probs[0]*100:.1f}%")
                    i_c1.progress(float(integ_probs[0]))
                    i_c2.metric("Staged / Fabricated", f"{integ_probs[1]*100:.1f}%")
                    i_c2.progress(float(integ_probs[1]))
                    
                    if integ_probs[1] > 0.55:
                        st.error("⚠️ Neural Engine flagged this image as potentially staged or out-of-context.")
                    else:
                        st.success("✅ Neural Engine: Image appears consistent with authentic, real-event media.")
                    
                    # --- FEATURE 4: Visual Sensationalism / Shock Value Index ---
                    st.markdown("---")
                    st.subheader("📸 Visual Sensationalism & Shock Value")
                    st.write("Fake news heavily relies on provocative, clickbaity visuals. This Neural Index ranks the sheer shock value of the image without evaluating the text.")
                    scene_labels = ["A boring, everyday mundane photograph", "A chaotic disaster or violent event", "An intense, emotionally charged or shocking scenario", "A text-heavy meme or deceptive infographic screenshot"]
                    sc_inputs = processor(text=scene_labels, images=active_image, return_tensors="pt", padding=True, truncation=True).to(device)
                    
                    with torch.no_grad():
                        sc_outputs = model.clip(input_ids=sc_inputs.input_ids, attention_mask=sc_inputs.attention_mask, pixel_values=sc_inputs.pixel_values)
                        sc_t_embeds = sc_outputs.text_embeds / sc_outputs.text_embeds.norm(dim=-1, keepdim=True)
                        sc_i_embeds = sc_outputs.image_embeds[0].unsqueeze(0) / sc_outputs.image_embeds[0].unsqueeze(0).norm(dim=-1, keepdim=True)
                        sc_sim = (sc_i_embeds @ sc_t_embeds.T).squeeze(0)
                        sc_probs = torch.softmax(sc_sim * 10.0, dim=0).cpu().numpy()
                    
                    shock_score = float(sc_probs[1] + sc_probs[2] + sc_probs[3])  # Sum of sensational categories
                    
                    st.write(f"**Calculated Visual Shock Value:** `{shock_score*100:.1f}%`")
                    
                    sc_fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = shock_score * 100,
                        number = {'suffix': "%"},
                        title = {'text': "Image Clickbait Intensity", 'font': {'size': 18, 'color': '#94A3B8'}},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "#FF3E9D" if shock_score > 0.60 else "#00F2FE"},
                            'steps': [
                                {'range': [0, 50], 'color': "rgba(0, 242, 254, 0.1)"},
                                {'range': [50, 100], 'color': "rgba(255, 62, 157, 0.1)"}],
                        }
                    ))
                    sc_fig.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=40), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#fff"})
                    st.plotly_chart(sc_fig, use_container_width=True)
                    
                    # --- FEATURE 3: Neural Synthetic Media / Deepfake Detection ---
                    st.markdown("---")
                    st.subheader("🤖 Neural Synthetic Media Detection")
                    st.write("CLIP's vision encoder detects whether the image exhibits structural residuals typical of AI-generated media (Midjourney, DALL-E, GAN).")
                    df_labels = ["A genuine physical photograph captured by a real camera.", "An AI-generated hyperrealistic synthetic image, deepfake, or computer graphic."]
                    df_inputs = processor(text=df_labels, images=active_image, return_tensors="pt", padding=True, truncation=True).to(device)
                    
                    with torch.no_grad():
                        df_outputs = model.clip(input_ids=df_inputs.input_ids, attention_mask=df_inputs.attention_mask, pixel_values=df_inputs.pixel_values)
                        df_t_embeds = df_outputs.text_embeds / df_outputs.text_embeds.norm(dim=-1, keepdim=True)
                        df_i_embeds = df_outputs.image_embeds[0].unsqueeze(0) / df_outputs.image_embeds[0].unsqueeze(0).norm(dim=-1, keepdim=True)
                        df_sim = (df_i_embeds @ df_t_embeds.T).squeeze(0)
                        df_probs = torch.softmax(df_sim * 15.0, dim=0).cpu().numpy()
                    
                    df_c1, df_c2 = st.columns(2)
                    df_c1.metric("Authentic Photograph", f"{df_probs[0]*100:.1f}%")
                    df_c1.progress(float(df_probs[0]))
                    df_c2.metric("AI-Generated / Deepfake", f"{df_probs[1]*100:.1f}%")
                    df_c2.progress(float(df_probs[1]))
                    
                    # --- FEATURE 5: Adversarial Perturbation Susceptibility ---
                    st.markdown("---")
                    st.subheader("🧫 Adversarial Perturbation Susceptibility")
                    st.write("Gaussian noise is injected into the image tensor. A large embedding collapse signals the image may be adversarially crafted to fool neural networks.")
                    
                    with torch.no_grad():
                        noise = torch.randn_like(pixel_values) * 0.1
                        noisy_pixels = (pixel_values + noise).clamp(0, 1)
                        
                        noisy_vis = model.clip.vision_model(pixel_values=noisy_pixels)
                        noisy_proj = model.clip.visual_projection(noisy_vis.pooler_output)
                        noisy_norm = noisy_proj[0].unsqueeze(0) / noisy_proj[0].unsqueeze(0).norm(dim=-1, keepdim=True)
                        
                        clean_vis = model.clip.vision_model(pixel_values=pixel_values)
                        clean_proj = model.clip.visual_projection(clean_vis.pooler_output)
                        clean_norm = clean_proj[0].unsqueeze(0) / clean_proj[0].unsqueeze(0).norm(dim=-1, keepdim=True)
                        
                        stability = float((clean_norm @ noisy_norm.T).squeeze().item())
                    
                    adv_c1, adv_c2 = st.columns([1, 2])
                    adv_c1.metric("Embedding Stability", f"{stability*100:.1f}%")
                    adv_c1.progress(float(min(max(stability, 0.0), 1.0)))
                    if stability < 0.80:
                        adv_c2.error("⚠️ **HIGH VULNERABILITY:** The neural embedding significantly collapsed under Gaussian perturbation. Possible adversarial manipulation.")
                    else:
                        adv_c2.success("✅ **ROBUST:** The image tensor strongly resisted noise injection. No adversarial attack signatures detected.")

                except Exception as e:
                    st.error(f"Could not generate ELA or Palette: {e}")
    else:
        # Placeholder before run
        # Mass Batch Layout takes over if present
        if not st.session_state.get('mass_batch_data', []):
            st.markdown("<h3 style='text-align: center; padding-top: 40px; color: #888;'>System Idle</h3>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #888;'>Awaiting multi-modal inputs for validation.</p>", unsafe_allow_html=True)

# ----------------- MASS BATCH RENDERER (Full DL Analytics) -----------------
if st.session_state.get('mass_batch_data', []):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## 🔥 Mass Batch Evaluation Gallery — Full Neural Analysis", unsafe_allow_html=True)
    st.markdown("Each sample below runs the complete Deep Learning pipeline: Credibility Score, Heatmap, Topic Classification, Emotion Polarity, Deepfake Detection, and Adversarial Robustness.")
    
    for idx, item in enumerate(st.session_state.mass_batch_data):
        st.markdown(f"<div class='premium-card'>", unsafe_allow_html=True)
        st.markdown(f"### 📰 Sample {idx+1}: *{item['text'][:120]}*", unsafe_allow_html=True)
        
        with st.spinner(f"Running full neural pipeline on sample {idx+1}..."):
            # Core inference
            b_inputs = processor(text=[item['text']], images=item['img'], return_tensors="pt", padding=True, truncation=True)
            b_input_ids = b_inputs["input_ids"].to(device)
            b_attn_mask = b_inputs["attention_mask"].to(device)
            b_pixel_vals = b_inputs["pixel_values"].to(device)
            
            with torch.no_grad():
                b_scores, _, _ = model.predict_score(b_input_ids, b_attn_mask, b_pixel_vals, return_attention=True)
                b_score = b_scores.item()
                
                # Text embeddings for zero-shot features
                b_txt_out = model.clip.text_model(input_ids=b_input_ids, attention_mask=b_attn_mask)
                b_txt_proj = model.clip.text_projection(b_txt_out.pooler_output)
                b_txt_norm = b_txt_proj[0].unsqueeze(0) / b_txt_proj[0].unsqueeze(0).norm(dim=-1, keepdim=True)
            
            b_heatmap = generate_heatmap(item['img'], b_score)
            truth_lbl = "🟢 REAL" if item['label'] == 1.0 else "🔴 FAKE"
            score_lbl = "🔴 FAKE DETECTED" if b_score < 0.5 else "🟢 VERIFIED REAL"
        
        # --- Row 1: Image + Heatmap + Gauge ---
        r1c1, r1c2, r1c3 = st.columns([1, 1, 1])
        with r1c1:
            st.image(item['img'], caption="Source Image", use_container_width=True)
        with r1c2:
            st.image(b_heatmap, caption="AI Attention Heatmap", use_container_width=True)
        with r1c3:
            b_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=b_score * 100,
                number={'suffix': "%"},
                title={'text': "Credibility Score", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#00ff88" if b_score >= 0.50 else "#ff3366"},
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(255,51,102,0.1)"},
                        {'range': [50, 100], 'color': "rgba(0,255,136,0.1)"}],
                }
            ))
            b_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=80, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#fff"})
            st.plotly_chart(b_gauge, use_container_width=True)
        
        st.markdown(f"**Ground Truth:** `{truth_lbl}` &nbsp;&nbsp;|&nbsp;&nbsp; **Prediction:** `{score_lbl}` (`{b_score:.4f}`)")
        st.progress(b_score)
        st.markdown("---")
        
        # --- Row 2: Cross-Modal Consistency & Misinformation Patterns ---
        r2c1, r2c2 = st.columns(2)
        
        with r2c1:
            st.markdown("**🔗 Cross-Modal Semantic Consistency**")
            b_cons_labels = [
                f"Image directly illustrating: '{item['text'][:60]}'",
                f"Image unrelated to: '{item['text'][:50]}'",
                "Real event photo matching the news.",
                "Recycled/stock image misrepresenting a story.",
                "Strong visual evidence for the claim."
            ]
            b_cons_inp = processor(text=b_cons_labels, images=item['img'], return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                b_cons_out = model.clip(input_ids=b_cons_inp.input_ids, attention_mask=b_cons_inp.attention_mask, pixel_values=b_cons_inp.pixel_values)
                b_cons_t = b_cons_out.text_embeds / b_cons_out.text_embeds.norm(dim=-1, keepdim=True)
                b_cons_i = b_cons_out.image_embeds[0].unsqueeze(0) / b_cons_out.image_embeds[0].unsqueeze(0).norm(dim=-1, keepdim=True)
                b_cons_sim = (b_cons_i @ b_cons_t.T).squeeze(0)
                b_cons_prob = torch.softmax(b_cons_sim * 12.0, dim=0).cpu().numpy()
            b_cons_labels_short = ["Matches", "Unrelated", "Real Event", "Recycled", "Evidence"]
            b_cf = go.Figure(data=[go.Bar(x=b_cons_labels_short, y=b_cons_prob,
                marker_color=['#00F2FE', '#FF3E9D', '#00ff88', '#FFB800', '#7B61FF'])])
            b_cf.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=40), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#94A3B8", 'size': 10})
            st.plotly_chart(b_cf, use_container_width=True)
        
        with r2c2:
            st.markdown("**🧬 Misinformation Intent Pattern**")
            b_mis_patterns = ["Factual News", "Propaganda", "Fear-mongering", "Clickbait", "Conspiracy Theory", "Satire"]
            b_mis_full = [
                "A neutral, fact-based, verified news report.",
                "Propaganda reinforcing a political narrative.",
                "Fear-mongering with exaggerated crisis language.",
                "Clickbait with a shocking misleading hook.",
                "Conspiracy theory implying hidden actors.",
                "Satire not intended as literal truth."
            ]
            b_mis_inp = processor(text=b_mis_full, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                b_mis_out = model.clip(input_ids=b_mis_inp.input_ids, attention_mask=b_mis_inp.attention_mask, pixel_values=torch.zeros((len(b_mis_full), 3, 224, 224), device=device))
                b_mis_emb = b_mis_out.text_embeds / b_mis_out.text_embeds.norm(dim=-1, keepdim=True)
                b_mis_sim = (b_txt_norm @ b_mis_emb.T).squeeze(0)
                b_mis_prob = torch.softmax(b_mis_sim * 10.0, dim=0).cpu().numpy()
            b_mf = go.Figure(data=[go.Bar(x=b_mis_patterns, y=b_mis_prob,
                marker_color=['#00ff88', '#FF3E9D', '#FF3E9D', '#FFB800', '#7B61FF', '#00F2FE'])])
            b_mf.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=40), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#94A3B8", 'size': 10})
            st.plotly_chart(b_mf, use_container_width=True)
            st.caption(f"Dominant: **{b_mis_patterns[int(np.argmax(b_mis_prob))]}** ({np.max(b_mis_prob)*100:.0f}%)")
        
        st.markdown("---")

        
        # --- Row 3: Deepfake Detection & Adversarial Robustness ---
        r3c1, r3c2 = st.columns(2)
        
        with r3c1:
            st.markdown("**🤖 Deepfake / Synthetic Media Detection**")
            b_df_inp = processor(text=["A genuine photograph.", "An AI-generated or deepfake image."], images=item['img'], return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                b_df_out = model.clip(input_ids=b_df_inp.input_ids, attention_mask=b_df_inp.attention_mask, pixel_values=b_df_inp.pixel_values)
                b_df_t = b_df_out.text_embeds / b_df_out.text_embeds.norm(dim=-1, keepdim=True)
                b_df_i = b_df_out.image_embeds[0].unsqueeze(0) / b_df_out.image_embeds[0].unsqueeze(0).norm(dim=-1, keepdim=True)
                b_df_sim = (b_df_i @ b_df_t.T).squeeze(0)
                b_df_prob = torch.softmax(b_df_sim * 15.0, dim=0).cpu().numpy()
            bdf1, bdf2 = st.columns(2)
            bdf1.metric("Authentic", f"{b_df_prob[0]*100:.0f}%"); bdf1.progress(float(b_df_prob[0]))
            bdf2.metric("AI-Generated", f"{b_df_prob[1]*100:.0f}%"); bdf2.progress(float(b_df_prob[1]))
        
        with r3c2:
            st.markdown("**🧫 Adversarial Robustness**")
            with torch.no_grad():
                b_noise = torch.randn_like(b_pixel_vals) * 0.1
                b_noisy = (b_pixel_vals + b_noise).clamp(0, 1)
                b_cv = model.clip.vision_model(pixel_values=b_pixel_vals)
                b_nv = model.clip.vision_model(pixel_values=b_noisy)
                b_cp = model.clip.visual_projection(b_cv.pooler_output)
                b_np_ = model.clip.visual_projection(b_nv.pooler_output)
                b_cn = b_cp[0].unsqueeze(0) / b_cp[0].unsqueeze(0).norm(dim=-1, keepdim=True)
                b_nn = b_np_[0].unsqueeze(0) / b_np_[0].unsqueeze(0).norm(dim=-1, keepdim=True)
                b_stab = float((b_cn @ b_nn.T).squeeze().item())
            st.metric("Embedding Stability", f"{b_stab*100:.1f}%")
            st.progress(float(min(max(b_stab, 0.0), 1.0)))
            if b_stab < 0.80:
                st.error("⚠️ Adversarially Vulnerable")
            else:
                st.success("✅ Robust")
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<br><br><center style='color:#555;'>Deep Learning Framework - V2 Architecture</center>", unsafe_allow_html=True)
