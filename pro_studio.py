import sys
import streamlit as st
import io

# --- AI COMPATIBILITY PATCH ---
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    try:
        from torchvision.transforms import functional as F
        sys.modules['torchvision.transforms.functional_tensor'] = F
    except: pass

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from rembg import remove
from gfpgan import GFPGANer

# --- CONFIGURATION ---
PX_PER_MM = 23.622  # 600 DPI Accuracy
CANVAS_SIZES = {
    "A4 Sheet": (210, 297),
    "4x6 Inch (Photo Paper)": (101.6, 152.4),
    "A3 Sheet": (297, 420),
}
PHOTO_TYPES = {
    "Standard Passport (35x45mm)": (35, 45),
    "US Visa (2x2 inch)": (50.8, 50.8),
    "Stamp Size (20x25mm)": (20, 25),
    "Custom Size": None
}

# --- UI ENHANCEMENTS (CSS) ---
st.set_page_config(page_title="Lekhmala Photo Studio", layout="wide", page_icon="📸")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007BFF; color: white; }
    .stDownloadButton>button { width: 100%; border-radius: 5px; background-color: #28a745; color: white; }
    h1 { color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif; text-align: center; margin-bottom: 0px;}
    .subtitle { text-align: center; color: #555; margin-bottom: 30px; }
    .footer { text-align: center; margin-top: 50px; color: #888; font-size: 0.8em; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    import os
    import urllib.request
    model_path = 'GFPGANv1.4.pth'
    # Auto-download the model if it's missing (needed for Cloud)
    if not os.path.exists(model_path):
        with st.spinner("Downloading AI Model for the first time... please wait."):
            url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
            urllib.request.urlretrieve(url, model_path)
    
    return GFPGANer(model_path=model_path, upscale=2, arch='clean', channel_multiplier=2)

# --- HEADER ---
st.markdown("<h1>📸 Lekhmala Photo Studio</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'><b> by <b>Bishal Mehta</b></p>", unsafe_allow_html=True)

# --- TABBED INTERFACE ---
tab1, tab2, tab3 = st.tabs(["🚀 1. Setup & Upload", "🎨 2. Crop & Adjustment", "📋 3. Layout & Export"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.info("📄 Paper Configuration")
        paper_choice = st.selectbox("Select Paper Size", list(CANVAS_SIZES.keys()))
        orient = st.radio("Page Orientation", ["Portrait", "Landscape"], index=1, horizontal=True)
    with col2:
        st.info("🖼️ Photo Configuration")
        photo_choice = st.selectbox("Select Photo Type", list(PHOTO_TYPES.keys()))
        if photo_choice == "Custom Size":
            target_dim = (st.number_input("Width (mm)", 35), st.number_input("Height (mm)", 45))
        else:
            target_dim = PHOTO_TYPES[photo_choice]

    uploaded_file = st.file_uploader("📤 Upload Image (Closeup Portrait)", type=['jpg', 'jpeg', 'png'])

# --- LOGIC ENGINE ---
if uploaded_file:
    if 'processed_img' not in st.session_state or st.session_state.get('last_uploaded') != uploaded_file.name:
        with st.status("Running Professional AI Enhancement...", expanded=True) as status:
            st.write("Loading Face Restoration Model...")
            enhancer = load_models()
            data = np.frombuffer(uploaded_file.read(), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            
            st.write("Restoring Biometric Details...")
            _, _, enhanced = enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            pil_img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            
            st.write("Generating Clean White Background...")
            no_bg = remove(pil_img)
            final_sub = Image.new("RGBA", no_bg.size, (255, 255, 255, 255))
            final_sub.paste(no_bg, (0, 0), mask=no_bg)
            
            st.session_state['processed_img'] = final_sub.convert("RGB")
            st.session_state['last_uploaded'] = uploaded_file.name
            status.update(label="AI Processing Complete! Goto Next Tab", state="complete")

    subject = st.session_state['processed_img']

    with tab2:
        col_ctrl, col_prev = st.columns([1, 1])
        with col_ctrl:
            st.write("### ✂️ Manual Crop Control")
            zoom = st.slider("Zoom Level", 0.5, 4.0, 1.25)
            move_y = st.slider("Vertical Shift", -1000, 1000, 0)
            move_x = st.slider("Horizontal Shift", -1000, 1000, 0)
            show_border = st.checkbox("Apply Photo Border (Black)", value=True)
        
        # Real-time Photo Crop
        sw_px, sh_px = int(target_dim[0] * PX_PER_MM), int(target_dim[1] * PX_PER_MM)
        img_w, img_h = subject.size
        crop_w = img_w / zoom
        crop_h = (crop_w * sh_px) / sw_px
        left = (img_w/2 + move_x) - crop_w/2
        top = (img_h/2 + move_y) - crop_h/2
        
        single_photo = subject.crop((left, top, left + crop_w, top + crop_h))
        single_photo = single_photo.resize((sw_px, sh_px), Image.Resampling.LANCZOS)
        if show_border:
            draw = ImageDraw.Draw(single_photo)
            draw.rectangle([0, 0, sw_px-1, sh_px-1], outline="black", width=3)
            
        with col_prev:
            st.write("### 🖼️ Frame Preview")
            st.image(single_photo, width=220)

    with tab3:
        col_lay, col_sheet = st.columns([1, 2])
        with col_lay:
            st.write("### 📐 Layout Control")
            num_copies = st.number_input("Copies to print", 1, 250, 12)
            gap_h = st.slider("Horizontal Gap (mm)", 0.0, 15.0, 2.0)
            gap_v = st.slider("Vertical Gap (mm)", 0.0, 15.0, 2.0)
            margin = st.slider("Page Margin (mm)", 2.0, 25.0, 5.0)
            pdf_mode = st.selectbox("Export Color Mode", ["RGB", "CMYK"])
            
        # Canvas Generation
        p_w_mm, p_h_mm = CANVAS_SIZES[paper_choice]
        cw_px, ch_px = (int(p_h_mm * PX_PER_MM), int(p_w_mm * PX_PER_MM)) if orient == "Landscape" else (int(p_w_mm * PX_PER_MM), int(p_h_mm * PX_PER_MM))
        
        canvas = Image.new("RGB", (cw_px, ch_px), "white")
        m_px, gh_px, gv_px = int(margin * PX_PER_MM), int(gap_h * PX_PER_MM), int(gap_v * PX_PER_MM)
        curr_x, curr_y, placed = m_px, m_px, 0
        
        for i in range(num_copies):
            if curr_x + sw_px > cw_px - m_px:
                curr_x = m_px
                curr_y += sh_px + gv_px
            if curr_y + sh_px > ch_px - m_px: break
            canvas.paste(single_photo, (curr_x, curr_y))
            curr_x += sw_px + gh_px
            placed += 1

        with col_sheet:
            st.write(f"### 📄 Final Canvas ({placed} Photos)")
            st.image(canvas, use_container_width=True)

        # Download Actions
        st.markdown("---")
        d1, d2, d3 = st.columns(3)
        buf_j = io.BytesIO(); canvas.save(buf_j, format="JPEG", quality=100); d1.download_button("📥 Save as JPG", buf_j.getvalue(), "studio.jpg")
        buf_p = io.BytesIO(); canvas.save(buf_p, format="PNG"); d2.download_button("📥 Save as PNG", buf_p.getvalue(), "studio.png")
        buf_pdf = io.BytesIO(); final_pdf = canvas.copy()
        if pdf_mode == "CMYK": final_pdf = final_pdf.convert("CMYK")
        final_pdf.save(buf_pdf, format="PDF", resolution=600.0); d3.download_button(f"📥 Save as {pdf_mode} PDF", buf_pdf.getvalue(), "studio.pdf")

else:
    st.warning("👋 Welcome! Please upload a portrait photo in Tab 1 to start your professional session.")

# --- FOOTER ---
st.markdown("<div class='footer'>Lekhmala Photo Studio v2.0 | Professional Photo Print Solution | © 2026 Bishal Mehta</div>", unsafe_allow_html=True)