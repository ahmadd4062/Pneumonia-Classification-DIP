import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="PneumoScan · AI X-Ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #08090c !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f1117 !important;
    border-right: 1px solid #1e2530 !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* ── Brand block ── */
.brand-block {
    padding: 24px 20px 18px;
    border-bottom: 1px solid #1e2530;
    margin-bottom: 20px;
}
.brand-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 3px;
}
.brand-icon {
    width: 34px; height: 34px;
    background: rgba(0,212,170,0.12);
    border: 1px solid #00d4aa;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}
.brand-name {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 16px;
    color: #e8eaf0;
    letter-spacing: 0.06em;
}
.brand-tagline {
    font-size: 11px;
    color: #4a5568;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 4px;
    padding-left: 44px;
}
.nav-section-label {
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a5568;
    padding: 0 8px;
    margin-bottom: 6px;
    margin-top: 4px;
}
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 12px;
    border-radius: 8px;
    color: #7e8fa6;
    font-size: 13px;
    cursor: pointer;
    margin-bottom: 2px;
}
.nav-item.active {
    background: rgba(0,212,170,0.1);
    color: #00d4aa;
    border: 1px solid rgba(0,212,170,0.2);
}
.nav-divider { border: none; border-top: 1px solid #1e2530; margin: 14px 0; }

/* ── Model info card ── */
.model-info-card {
    background: #161b22;
    border: 1px solid #1e2530;
    border-radius: 10px;
    padding: 14px 16px;
    margin: 16px 0 8px;
}
.model-card-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4a5568;
    margin-bottom: 5px;
    font-family: 'DM Mono', monospace;
}
.model-card-name {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #00d4aa;
    margin-bottom: 10px;
}
.model-acc-bar-bg {
    height: 3px;
    background: #1e2530;
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 6px;
}
.model-acc-bar-fill {
    height: 100%;
    width: 94%;
    background: #00d4aa;
    border-radius: 2px;
}
.model-stat-row {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #7e8fa6;
    margin-top: 4px;
    font-family: 'DM Mono', monospace;
}
.model-stat-val { color: #00d4aa; }
.dev-note {
    padding: 0 4px;
    margin-top: 8px;
    font-size: 10px;
    color: #4a5568;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.08em;
    line-height: 1.8;
}
.dev-note span { color: #7e8fa6; }

/* ── Top bar ── */
.topbar {
    padding: 18px 32px;
    background: #0f1117;
    border-bottom: 1px solid #1e2530;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.page-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 22px;
    color: #e8eaf0;
    letter-spacing: -0.01em;
}
.page-title .accent { color: #00d4aa; }
.topbar-badges { display: flex; gap: 10px; align-items: center; }
.badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.04em;
    font-family: 'DM Mono', monospace;
    display: inline-block;
}
.badge-green { background: rgba(61,220,132,0.1); color: #3ddc84; border: 1px solid rgba(61,220,132,0.2); }
.badge-amber { background: rgba(247,201,72,0.1); color: #f7c948; border: 1px solid rgba(247,201,72,0.2); }
.badge-red   { background: rgba(255,85,85,0.1);  color: #ff5555; border: 1px solid rgba(255,85,85,0.2); }

/* ── Section header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 16px;
}
.section-bar {
    width: 3px;
    height: 18px;
    background: #00d4aa;
    border-radius: 2px;
    flex-shrink: 0;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 15px;
    color: #e8eaf0;
}
.section-sub {
    font-size: 11px;
    color: #4a5568;
    font-family: 'DM Mono', monospace;
    margin-left: auto;
}

/* ── Upload zone ── */
.upload-zone-wrap {
    border: 1px dashed #2a3441;
    border-radius: 16px;
    padding: 40px 32px;
    text-align: center;
    background: #161b22;
    position: relative;
    margin-bottom: 8px;
}
.upload-label-corner {
    position: absolute;
    top: 16px; right: 20px;
    font-size: 10px;
    font-family: 'DM Mono', monospace;
    color: #4a5568;
    letter-spacing: 0.15em;
}
.upload-icon-wrap {
    width: 56px; height: 56px;
    background: rgba(0,212,170,0.1);
    border: 1px solid rgba(0,212,170,0.3);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 16px;
    font-size: 24px;
}
.upload-title {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 18px;
    color: #e8eaf0;
    margin-bottom: 8px;
}
.upload-sub {
    color: #7e8fa6;
    font-size: 13px;
    line-height: 1.7;
    margin-bottom: 14px;
}
.fmt-pills { display: flex; gap: 6px; justify-content: center; flex-wrap: wrap; }
.fmt-pill {
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-family: 'DM Mono', monospace;
    background: #1e2530;
    color: #7e8fa6;
    border: 1px solid #2a3441;
    display: inline-block;
}

/* ── Step cards ── */
.step-card {
    background: #161b22;
    border: 1px solid #1e2530;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 4px;
}
.step-card.active-card { border-color: rgba(0,212,170,0.5); }
.step-meta-wrap { padding: 12px 14px; }
.step-num {
    font-size: 10px;
    font-family: 'DM Mono', monospace;
    color: #4a5568;
    letter-spacing: 0.1em;
    margin-bottom: 3px;
}
.step-name {
    font-size: 13px;
    font-weight: 500;
    color: #e8eaf0;
    margin-bottom: 4px;
}
.step-desc {
    font-size: 11px;
    color: #7e8fa6;
    line-height: 1.5;
}
.step-tag {
    display: inline-block;
    margin-top: 8px;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 10px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
}
.tag-core     { background: rgba(0,212,170,0.08); color: #00d4aa; border: 1px solid rgba(0,212,170,0.2); }
.tag-analysis { background: #1e2530; color: #7e8fa6; border: 1px solid #2a3441; }

/* ── Diagnosis result ── */
.diag-card {
    background: #161b22;
    border: 1px solid #1e2530;
    border-radius: 14px;
    padding: 28px 24px;
    text-align: center;
}
.diag-ring {
    width: 90px; height: 90px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 16px;
    font-size: 36px;
}
.ring-normal   { background: rgba(61,220,132,0.08); border: 2px solid #3ddc84; }
.ring-pneumonia{ background: rgba(255,85,85,0.08); border: 2px solid #ff5555; }
.diag-label {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 22px;
    margin-bottom: 6px;
}
.label-normal    { color: #3ddc84; }
.label-pneumonia { color: #ff5555; }
.diag-sub { font-size: 12px; color: #7e8fa6; line-height: 1.6; }

/* ── Stat bars ── */
.stat-bar-card {
    background: #161b22;
    border: 1px solid #1e2530;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 10px;
}
.stat-bar-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4a5568;
    font-family: 'DM Mono', monospace;
    margin-bottom: 6px;
}
.stat-bar-val {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 24px;
    margin-bottom: 8px;
}
.stat-bar-bg { height: 4px; background: #1e2530; border-radius: 2px; overflow: hidden; }
.stat-bar-fill-normal    { height: 100%; background: #3ddc84; border-radius: 2px; }
.stat-bar-fill-pneumonia { height: 100%; background: #ff5555; border-radius: 2px; }

/* ── Raw output box ── */
.raw-output-box {
    background: #161b22;
    border: 1px solid #1e2530;
    border-radius: 12px;
    padding: 14px 18px;
    margin-top: 0;
}
.raw-output-label {
    font-size: 10px;
    font-family: 'DM Mono', monospace;
    color: #4a5568;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.raw-output-val {
    font-family: 'DM Mono', monospace;
    font-size: 14px;
    color: #00d4aa;
}
.raw-output-meta {
    font-size: 11px;
    color: #7e8fa6;
    margin-top: 4px;
}

/* ── Edge card ── */
.edge-card {
    background: #161b22;
    border: 1px solid #1e2530;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 6px;
}
.edge-meta {
    padding: 10px 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.edge-name {
    font-size: 12px;
    font-weight: 500;
    color: #e8eaf0;
    margin-bottom: 2px;
}
.edge-type {
    font-size: 10px;
    font-family: 'DM Mono', monospace;
    color: #4a5568;
    letter-spacing: 0.05em;
}

/* ── Status bar ── */
.status-bar {
    border-top: 1px solid #1e2530;
    padding: 10px 32px;
    background: #0f1117;
    display: flex;
    align-items: center;
    gap: 16px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #4a5568;
    flex-wrap: wrap;
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #3ddc84;
    display: inline-block;
    animation: pulse 2s infinite;
    flex-shrink: 0;
}
@keyframes pulse { 0%,100%{opacity:1}50%{opacity:0.3} }
.status-sep { width: 1px; height: 12px; background: #1e2530; flex-shrink: 0; }
.status-val { color: #7e8fa6; }

/* ── Disclaimer ── */
.disclaimer-bar {
    text-align: center;
    padding: 14px 32px;
    font-size: 11px;
    color: #4a5568;
    border-top: 1px solid #1e2530;
    line-height: 1.8;
}
.disclaimer-bar .warn { color: #f7c948; }

/* ── Error box ── */
.error-box {
    margin: 20px 32px;
    padding: 16px 20px;
    background: rgba(255,85,85,0.08);
    border: 1px solid rgba(255,85,85,0.3);
    border-radius: 10px;
    font-size: 13px;
    color: #ff5555;
}

/* ── Landing grid ── */
.landing-wrap {
    padding: 40px 32px;
    max-width: 900px;
    margin: 0 auto;
}
.landing-hero {
    text-align: center;
    margin-bottom: 40px;
}
.landing-emoji { font-size: 56px; margin-bottom: 16px; }
.landing-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 32px;
    color: #e8eaf0;
    letter-spacing: -0.02em;
    margin-bottom: 10px;
}
.landing-sub {
    color: #7e8fa6;
    font-size: 14px;
    line-height: 1.8;
    max-width: 520px;
    margin: 0 auto;
}
.feat-card {
    background: #161b22;
    border: 1px solid #1e2530;
    border-radius: 12px;
    padding: 20px 18px;
    height: 100%;
}
.feat-card.highlight { border-color: rgba(0,212,170,0.25); }
.feat-icon { font-size: 24px; margin-bottom: 10px; }
.feat-title {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 14px;
    color: #e8eaf0;
    margin-bottom: 6px;
}
.feat-desc { font-size: 12px; color: #7e8fa6; line-height: 1.6; }
.disclaimer-card {
    background: #161b22;
    border: 1px solid #1e2530;
    border-radius: 12px;
    padding: 16px 20px;
    font-size: 12px;
    color: #7e8fa6;
    line-height: 1.8;
    margin-top: 16px;
}
.disclaimer-card .warn-label {
    color: #f7c948;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    display: block;
    margin-bottom: 4px;
}
.disclaimer-card strong { color: #e8eaf0; }

/* ── Streamlit widget theming ── */
[data-testid="stFileUploader"] { background: transparent !important; }
[data-testid="stFileUploader"] section {
    background: #161b22 !important;
    border: 1px dashed #2a3441 !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
[data-testid="stFileUploader"] label { color: #7e8fa6 !important; font-size: 13px !important; }
.stButton > button {
    background: rgba(0,212,170,0.1) !important;
    color: #00d4aa !important;
    border: 1px solid rgba(0,212,170,0.3) !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.05em !important;
}
.stButton > button:hover {
    background: rgba(0,212,170,0.2) !important;
    border-color: #00d4aa !important;
}
.stSpinner > div { border-top-color: #00d4aa !important; }

/* Force Streamlit p tags inside markdown not to collapse flex children */
[data-testid="stMarkdownContainer"] p { margin: 0; }
</style>
""", unsafe_allow_html=True)


# ============== HELPER FUNCTIONS (FIXED) ==============

def preprocess_image(image_path, img_size=(224, 224)):
    """
    Load and preprocess image for DIP pipeline
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, img_size)
    return img


def enhance_image(img):
    """
    Apply enhancement: gamma correction + histogram equalization
    """
    # Gamma correction
    def gamma_correction(image, gamma=1.5):
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    gamma_corrected = gamma_correction(img, gamma=1.5)
    
    # Histogram equalization
    enhanced = cv2.equalizeHist(gamma_corrected)
    
    return enhanced


def full_dip_pipeline(image_path):
    """
    Full DIP pipeline returning all intermediate results
    """
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        return None
    
    resized = cv2.resize(original, (224, 224))
    
    # Gamma correction
    def gamma_correction(img, gamma=1.5):
        table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img, table)
    
    gamma_corrected = gamma_correction(resized, gamma=1.5)
    
    # Histogram Equalization
    hist_equalized = cv2.equalizeHist(gamma_corrected)
    
    # Laplacian edges
    laplacian = np.uint8(np.abs(cv2.Laplacian(hist_equalized, cv2.CV_64F)))
    
    # Binary threshold
    _, binary_thresh = cv2.threshold(hist_equalized, 127, 255, cv2.THRESH_BINARY)
    
    # Otsu threshold
    otsu_val, otsu_thresh = cv2.threshold(
        hist_equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # Bit Plane Slicing (MSB plane 7)
    bit_plane_7 = cv2.bitwise_and(hist_equalized, np.full_like(hist_equalized, 128))
    bit_plane_7 = np.where(bit_plane_7 > 0, 255, 0).astype(np.uint8)
    
    # Histogram Stretching
    p_low  = np.percentile(hist_equalized, 2)
    p_high = np.percentile(hist_equalized, 98)
    if p_high > p_low:
        stretched = np.clip((hist_equalized.astype(np.float32) - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8)
    else:
        stretched = hist_equalized.copy()
    
    return {
        'original':   original,
        'resized':    resized,
        'gamma':      gamma_corrected,
        'histogram':  hist_equalized,
        'edges':      laplacian,
        'segmented':  binary_thresh,
        'enhanced':   hist_equalized,  # Alias for compatibility
        'otsu':       otsu_thresh,
        'otsu_val':   int(otsu_val),
        'bitplane':   bit_plane_7,
        'stretched':  stretched,
    }


def visualize_results(results):
    """
    Display DIP pipeline results using matplotlib
    """
    if results is None:
        return
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    steps = [
        ('Original', results['original']),
        ('Resized', results['resized']),
        ('Gamma', results['gamma']),
        ('Histogram Eq', results['histogram']),
        ('Edges', results['edges']),
        ('Binary', results['segmented']),
        ('Otsu', results['otsu']),
        ('Bit-Plane', results['bitplane'])
    ]
    
    for idx, (title, img) in enumerate(steps):
        row, col = idx // 4, idx % 4
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(title, fontsize=12)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig


def compare_edge_detection(img):
    """
    Compare Sobel vs Laplacian edge detection
    """
    # Sobel
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.uint8(np.clip(np.sqrt(sobelx**2 + sobely**2), 0, 255))
    
    # Laplacian
    laplacian = np.uint8(np.abs(cv2.Laplacian(img, cv2.CV_64F)))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(sobel, cmap='gray')
    axes[0].set_title('Sobel Edge Detection (First Order)', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(laplacian, cmap='gray')
    axes[1].set_title('Laplacian Edge Detection (Second Order)', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig, sobel, laplacian


# ============== LOAD MODEL ==============
@st.cache_resource
def load_model():
    model_paths = [
        'models/pneumonia_model.h5',
        'models/pneumonia_model_augmented.h5',
        '../models/pneumonia_model.h5',
    ]
    for path in model_paths:
        if os.path.exists(path):
            try:
                return tf.keras.models.load_model(path)
            except:
                continue
    return None


# ============== DIP PIPELINE ==============
def process_image_dip_steps(image_path):
    """
    Process image through DIP steps
    """
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        return None
    resized = cv2.resize(original, (224, 224))

    # ── Step 3: Gamma correction ──
    def gamma_correction(img, gamma=1.5):
        table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img, table)

    gamma_corrected = gamma_correction(resized, gamma=1.5)

    # ── Step 4: Histogram Equalization (CNN input) ──
    hist_equalized = cv2.equalizeHist(gamma_corrected)

    # ── Step 5: Laplacian edge detection ──
    laplacian = np.uint8(np.abs(cv2.Laplacian(hist_equalized, cv2.CV_64F)))

    # ── Step 6: Binary threshold ──
    _, binary_thresh = cv2.threshold(hist_equalized, 127, 255, cv2.THRESH_BINARY)

    # ── NEW Step 7: Otsu Thresholding ──
    otsu_val, otsu_thresh = cv2.threshold(
        hist_equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ── NEW Step 8: Bit Plane Slicing (MSB plane 7) ──
    bit_plane_7 = cv2.bitwise_and(hist_equalized, np.full_like(hist_equalized, 128))
    bit_plane_7 = np.where(bit_plane_7 > 0, 255, 0).astype(np.uint8)

    # ── NEW Step 9: Histogram Stretching (contrast stretching) ──
    p_low  = np.percentile(hist_equalized, 2)
    p_high = np.percentile(hist_equalized, 98)
    if p_high > p_low:
        stretched = np.clip((hist_equalized.astype(np.float32) - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8)
    else:
        stretched = hist_equalized.copy()

    return {
        'original':   original,
        'resized':    resized,
        'gamma':      gamma_corrected,
        'histogram':  hist_equalized,
        'edges':      laplacian,
        'segmented':  binary_thresh,
        'otsu':       otsu_thresh,
        'otsu_val':   int(otsu_val),
        'bitplane':   bit_plane_7,
        'stretched':  stretched,
    }


# ============== PREDICTION ==============
def predict_image(results, model):
    img = cv2.resize(results['histogram'], (224, 224)) / 255.0
    img = img.reshape(1, 224, 224, 1)
    return float(model.predict(img, verbose=0)[0][0])


# ============== SIDEBAR ==============
def render_sidebar(model_loaded: bool):
    with st.sidebar:
        st.markdown("""
        <div class="brand-block">
            <div class="brand-row">
                <div class="brand-icon">🫁</div>
                <div class="brand-name">PNEUMOSCAN</div>
            </div>
            <div class="brand-tagline">AI Chest X-Ray Analysis</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="nav-section-label">Tools</div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-item active">📊 &nbsp;Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-item">🔬 &nbsp;DIP Pipeline</div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-item">📁 &nbsp;History</div>', unsafe_allow_html=True)
        st.markdown('<hr class="nav-divider">', unsafe_allow_html=True)
        st.markdown('<div class="nav-section-label">System</div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-item">⚙️ &nbsp;Model Config</div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-item">ℹ️ &nbsp;About</div>', unsafe_allow_html=True)
        st.markdown('<hr class="nav-divider">', unsafe_allow_html=True)

        status_color = "#3ddc84" if model_loaded else "#ff5555"
        status_text  = "Loaded" if model_loaded else "Not Found"

        st.markdown(f"""
        <div class="model-info-card">
            <div class="model-card-label">Active Model</div>
            <div class="model-card-name">pneumonia_cnn.h5</div>
            <div class="model-acc-bar-bg"><div class="model-acc-bar-fill"></div></div>
            <div class="model-stat-row">
                <span>Accuracy</span>
                <span class="model-stat-val">94.3%</span>
            </div>
            <div class="model-stat-row">
                <span>Status</span>
                <span style="color:{status_color};">● {status_text}</span>
            </div>
            <div class="model-stat-row">
                <span>Input</span>
                <span>224×224 · Gray</span>
            </div>
        </div>
        <div class="dev-note">
            DEVELOPED BY<br><span>THE AMIGOS</span>
        </div>
        """, unsafe_allow_html=True)


# ============== TOP BAR ==============
def render_topbar(model_loaded: bool):
    model_badge = (
        '<span class="badge badge-green">● Model Loaded</span>'
        if model_loaded else
        '<span class="badge badge-red">✕ Model Missing</span>'
    )
    st.markdown(f"""
    <div class="topbar">
        <div class="page-title">Chest X-Ray <span class="accent">Diagnostic</span></div>
        <div class="topbar-badges">
            {model_badge}
            <span class="badge badge-amber">DIP · 9 Steps</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============== UPLOAD SECTION ==============
def render_upload_zone():
    st.markdown("""
    <div class="section-header">
        <div class="section-bar"></div>
        <div class="section-title">Input Image</div>
        <div class="section-sub">JPEG / PNG · Max 10 MB</div>
    </div>
    <div class="upload-zone-wrap">
        <div class="upload-label-corner">UPLOAD</div>
        <div class="upload-icon-wrap">⬆️</div>
        <div class="upload-title">Drop your chest X-ray here</div>
        <div class="upload-sub">
            Supported formats: JPG, JPEG, PNG<br>
            Image will be processed through a 9-step DIP pipeline before AI classification
        </div>
        <div class="fmt-pills">
            <span class="fmt-pill">JPG</span>
            <span class="fmt-pill">JPEG</span>
            <span class="fmt-pill">PNG</span>
            <span class="fmt-pill">Max 10MB</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    return st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")


# ============== DIP PIPELINE DISPLAY ==============
def render_pipeline(results):
    """Renders all 9 DIP steps in a 3-column grid."""

    steps = [
        ("01", "Source Image",            "Raw chest X-ray loaded as grayscale",                        "core",     "original",  ""),
        ("02", "Resize 224×224",          "Standardized to 224×224 pixels for CNN input",               "core",     "resized",   ""),
        ("03", "Gamma Correction",        "γ = 1.5 · boosts dark lung tissue details",                  "core",     "gamma",     ""),
        ("04", "Histogram Equalization",  "Global contrast enhancement · used as CNN input",            "active",   "histogram", ""),
        ("05", "Laplacian Edges",         "Second-order derivative · highlights fine structures",        "analysis", "edges",     ""),
        ("06", "Binary Threshold",        "Fixed threshold T=127 · baseline segmentation",              "analysis", "segmented", ""),
        ("07", "Otsu Threshold",          f"Auto threshold T={results['otsu_val']} · optimal class sep","analysis", "otsu",      "Otsu"),
        ("08", "Bit-Plane Slice (MSB)",   "Plane 7 isolation · reveals dominant intensity structure",   "analysis", "bitplane",  "Bit-Plane"),
        ("09", "Histogram Stretching",    "2–98 percentile stretch · maximises dynamic range",          "analysis", "stretched", "Stretch"),
    ]

    st.markdown("""
    <div class="section-header">
        <div class="section-bar"></div>
        <div class="section-title">DIP Processing Pipeline</div>
        <div class="section-sub">STEPS 01 — 09</div>
    </div>
    """, unsafe_allow_html=True)

    for row_start in range(0, len(steps), 3):
        row_steps = steps[row_start:row_start + 3]
        cols = st.columns(3, gap="small")
        for col_idx, step in enumerate(row_steps):
            num, name, desc, kind, img_key, label = step
            tag_html = (
                '<span class="step-tag tag-core">CORE · CNN INPUT</span>' if kind == "active"
                else '<span class="step-tag tag-core">CORE</span>' if kind == "core"
                else f'<span class="step-tag tag-analysis">{"NEW · " if label else ""}DISPLAY ONLY</span>'
            )
            border_cls = "active-card" if kind == "active" else ""
            with cols[col_idx]:
                st.markdown(f"""
                <div class="step-card {border_cls}">
                    <div class="step-meta-wrap">
                        <div class="step-num">STEP {num}</div>
                        <div class="step-name">{name}</div>
                        <div class="step-desc">{desc}</div>
                        {tag_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.image(results[img_key], use_container_width=True, clamp=True)


# ============== DIAGNOSIS DISPLAY ==============
def render_diagnosis(prediction: float):
    is_pneumonia = prediction > 0.5
    conf     = prediction if is_pneumonia else 1.0 - prediction
    pneu_pct = prediction
    norm_pct = 1.0 - prediction

    st.markdown("""
    <div class="section-header">
        <div class="section-bar"></div>
        <div class="section-title">Diagnosis Output</div>
    </div>
    """, unsafe_allow_html=True)

    col_result, col_stats = st.columns([1, 1], gap="medium")

    with col_result:
        ring_cls  = "ring-pneumonia" if is_pneumonia else "ring-normal"
        icon      = "⚠️" if is_pneumonia else "✅"
        label     = "PNEUMONIA" if is_pneumonia else "NORMAL"
        label_cls = "label-pneumonia" if is_pneumonia else "label-normal"
        sub_line  = (
            f"Confidence: {conf:.1%} · Pneumonia markers detected"
            if is_pneumonia else
            f"Confidence: {conf:.1%} · No pneumonia markers detected"
        )
        st.markdown(f"""
        <div class="diag-card">
            <div class="diag-ring {ring_cls}">{icon}</div>
            <div class="diag-label {label_cls}">{label}</div>
            <div class="diag-sub">
                {sub_line}<br><br>
                <span style="font-size:11px;color:#f7c948;">
                    ⚠ Always consult a medical professional for actual diagnosis.
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_stats:
        st.markdown(f"""
        <div class="stat-bar-card">
            <div class="stat-bar-label">Normal Probability</div>
            <div class="stat-bar-val" style="color:#3ddc84">{norm_pct:.1%}</div>
            <div class="stat-bar-bg">
                <div class="stat-bar-fill-normal" style="width:{norm_pct*100:.1f}%"></div>
            </div>
        </div>
        <div class="stat-bar-card">
            <div class="stat-bar-label">Pneumonia Probability</div>
            <div class="stat-bar-val" style="color:#ff5555">{pneu_pct:.1%}</div>
            <div class="stat-bar-bg">
                <div class="stat-bar-fill-pneumonia" style="width:{pneu_pct*100:.1f}%"></div>
            </div>
        </div>
        <div class="raw-output-box">
            <div class="raw-output-label">RAW MODEL OUTPUT</div>
            <div class="raw-output-val">{prediction:.6f}</div>
            <div class="raw-output-meta">Threshold: 0.500000 · Input: histogram-equalized</div>
        </div>
        """, unsafe_allow_html=True)


# ============== EDGE COMPARISON ==============
def render_edge_comparison(img_gray: np.ndarray):
    sobelx    = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely    = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel     = np.uint8(np.clip(np.sqrt(sobelx**2 + sobely**2), 0, 255))
    laplacian = np.uint8(np.abs(cv2.Laplacian(img_gray, cv2.CV_64F)))

    st.markdown("""
    <div class="section-header">
        <div class="section-bar"></div>
        <div class="section-title">Edge Detection Comparison</div>
        <div class="section-sub">SOBEL vs LAPLACIAN</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("""
        <div class="edge-card">
            <div class="edge-meta">
                <div>
                    <div class="edge-name">Sobel Edge Detection</div>
                    <div class="edge-type">FIRST-ORDER DERIVATIVE · ∇f</div>
                </div>
                <span class="badge badge-amber">∇f</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.image(sobel, use_container_width=True, clamp=True)

    with c2:
        st.markdown("""
        <div class="edge-card">
            <div class="edge-meta">
                <div>
                    <div class="edge-name">Laplacian Edge Detection</div>
                    <div class="edge-type">SECOND-ORDER DERIVATIVE · ∇²f</div>
                </div>
                <span class="badge badge-amber">∇²f</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.image(laplacian, use_container_width=True, clamp=True)


# ============== STATUS BAR ==============
def render_status_bar(model_loaded: bool, extra: str = ""):
    status = "System ready" if model_loaded else "Model not loaded"
    parts = [
        '<span class="status-dot"></span>',
        f'<span class="status-val">{status}</span>',
        '<span class="status-sep"></span>',
        '<span>Model</span>',
        '<span class="status-val">pneumonia_cnn.h5</span>',
        '<span class="status-sep"></span>',
        '<span>Input</span>',
        '<span class="status-val">224\u00d7224 \u00b7 Grayscale</span>',
    ]
    if extra:
        parts += [
            '<span class="status-sep"></span>',
            f'<span class="status-val">{extra}</span>',
        ]
    parts += [
        '<span class="status-sep"></span>',
        '<span>Developed by The Amigos</span>',
    ]
    inner = "".join(parts)
    st.markdown(
        f'<div class="status-bar">{inner}</div>'
        '<div class="disclaimer-bar">'
        '<span class="warn">\u26a0 Clinical Disclaimer:</span> '
        'This tool is intended for educational and research purposes only.<br>'
        'It does not replace professional medical judgment. Always consult a qualified physician for diagnosis and treatment.'
        '</div>',
        unsafe_allow_html=True,
    )


# ============== LANDING PAGE ==============
def render_landing():
    st.markdown('<div class="landing-wrap">', unsafe_allow_html=True)
    st.markdown("""
    <div class="landing-hero">
        <div class="landing-emoji">🫁</div>
        <div class="landing-title">AI-Powered Pneumonia Detection</div>
        <div class="landing-sub">
            Upload a chest X-ray and let the 9-step DIP pipeline classify it in seconds.
            Every processing stage is shown individually for full transparency.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        st.markdown("""
        <div class="feat-card">
            <div class="feat-icon">🔬</div>
            <div class="feat-title">9-Step DIP Pipeline</div>
            <div class="feat-desc">
                Resize, gamma correction, histogram equalization, edge detection,
                Otsu thresholding, bit-plane slicing, and histogram stretching —
                all visualised side by side.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feat-card highlight">
            <div class="feat-icon">🤖</div>
            <div class="feat-title">CNN Classification</div>
            <div class="feat-desc">
                Deep convolutional neural network trained on the Kaggle
                chest X-ray dataset, achieving 94%+ accuracy on the test set.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="feat-card">
            <div class="feat-icon">⚡</div>
            <div class="feat-title">Real-Time Results</div>
            <div class="feat-desc">
                Instant confidence scores with probability bars,
                plus dual edge-detection comparison (Sobel vs Laplacian).
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-card">
        <span class="warn-label">⚠ DISCLAIMER</span>
        This system is for educational and research use only.
        It is <strong>not a medical device</strong> and must not be used
        as a substitute for professional clinical diagnosis.
    </div>
    </div>
    """, unsafe_allow_html=True)


# ============== MAIN ==============
def main():
    model = load_model()
    model_loaded = model is not None

    render_sidebar(model_loaded)
    render_topbar(model_loaded)

    with st.container():
        st.markdown('<div style="padding: 0 32px;">', unsafe_allow_html=True)
        uploaded_file = render_upload_zone()
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        temp_path = "temp_uploaded_xray.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if not model_loaded:
            st.markdown("""
            <div class="error-box">
                ✕ &nbsp;Model not found. Train the CNN and place it at
                <code style="background:#1e2530;padding:2px 6px;border-radius:4px;font-size:12px">
                models/pneumonia_model.h5</code>
            </div>
            """, unsafe_allow_html=True)
            return

        with st.spinner("Processing image through DIP pipeline…"):
            results = process_image_dip_steps(temp_path)

        if results:
            st.markdown('<div style="padding: 0 32px;">', unsafe_allow_html=True)
            render_pipeline(results)

            with st.spinner("Running CNN inference…"):
                prediction = predict_image(results, model)

            render_diagnosis(prediction)
            render_edge_comparison(results['histogram'])
            st.markdown('</div>', unsafe_allow_html=True)

            if os.path.exists(temp_path):
                os.remove(temp_path)
        else:
            st.markdown("""
            <div class="error-box">
                ✕ &nbsp;Could not process image. Please try a different file.
            </div>
            """, unsafe_allow_html=True)
    else:
        render_landing()

    render_status_bar(model_loaded)


if __name__ == "__main__":
    main()