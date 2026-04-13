import os
import sys
import io
import urllib.request

# --- 1. CLOUD & GRAPHICS SAFETY NET ---
# Prevents OpenCV and Torch errors on Cloud Servers
os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    import torchvision.transforms.functional_tensor
except ImportError:
    try:
        from torchvision.transforms import functional as F
        sys.modules['torchvision.transforms.functional_tensor'] = F
    except:
        pass

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from rembg import remove
from gfpgan import GFPGANer

# --- 2. CONFIGURATION & STANDARDS ---
PX_PER_MM = 23.622  # High-Res 600 DPI Accuracy
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

# --- 3. PAGE SETUP & CSS ---
st.set_page_config(page_title="Lekhmala Photo Studio", layout="wide", page_icon="📸")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007BFF; color: white; font-weight: bold; }
    .stDownloadButton>button { width: 100%; border-radius: 5px; background-color: #28a745; color: white; font-weight: bold; }
    h1 { color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif; text-align: center; margin-bottom: 0px;}
    .subtitle { text-align: center; color: #555; margin-bottom: 30px; }
    .footer { text-align: center; margin-top: 50px; color: #888; font-size: 0.8em; border-top: 1px solid #ddd; padding-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. AI MODEL LOADER ---
@st.cache_resource
def load_models():
    model_path = 'GFPGANv1.4.pth'
    if not os.path.exists(model_path):
        with st.spinner("Downloading AI Face Model (First time only)..."):
            url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
            urllib.request.urlretrieve(url, model_path)
    return GFPGANer(model_path=model_path, upscale=2, arch='clean', channel_multiplier=2)

# --- 5. HEADER ---
st.markdown("<h1>📸 Lekhmala Photo Studio</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Professional Biometric Photo Solution | Developed by <b>Bishal Mehta</b></p>", unsafe_allow_html=True)

# --- 6. WORKFLOW TABS ---
tab1, tab2, tab3 = st.tabs(["🚀 Step 1: Upload", "🎨 Step 2: Adjust", "📥 Step 3: Export"])

with tab1:
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        st.info("📄 Paper Setup")
        paper_choice = st.selectbox("Select Paper Size", list(CANVAS_SIZES.keys()))
        orient = st.radio("Page Orientation", ["Portrait", "Landscape"], index=1, horizontal=True)
    with col_cfg2:
        st.info("🖼️ Photo Standard")
        photo_choice = st.selectbox("Select Photo Type", list(PHOTO_TYPES.keys()))
        if photo_choice == "Custom Size":
            target_dim = (st.number_input("Width (mm)", 35), st.number_input("Height (mm)", 45))
        else:
            target_dim = PHOTO_TYPES[photo_choice]

    uploaded_file = st.file_uploader("📤 Upload Portrait Photo", type=['jpg', 'jpeg', 'png'])

# --- 7. CORE PROCESSING LOGIC ---
if uploaded_file:
    # Use session state to prevent AI from re-running on every slider change
    if 'processed_img' not in st.session_state or st.session_state.get('last_uploaded') != uploaded_file.name:
        with st.status("Lekhmala AI is working...", expanded=True) as status:
            st.write("Initializing Face Restoration...")
            enhancer = load_models()
            
            # Read image
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            st.write("Enhancing Biometric Quality...")
            _, _, enhanced = enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            pil_img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            
            st.write("Generating Studio White Background...")
            no_bg = remove(pil_img)
            
            # Create white background
            final_sub = Image.new("RGBA", no_bg.size, (255, 255, 255, 255))
            final_sub.paste(no_bg, (0, 0), mask=no_bg)
            
            st.session_state['processed_img'] = final_sub.convert("RGB")
            st.session_state['last_uploaded'] = uploaded_file.name
            status.update(label="AI Processing Complete!", state="complete")

    subject_img = st.session_state['processed_img']

    with tab2:
        st.write("### ✂️ Frame & Biometric Alignment")
        col_ctrl, col_prev = st.columns([1, 1])
        
        with col_ctrl:
            st.info("Nepal Standard: Head should fill 70-80% of the frame.")
            zoom = st.slider("Zoom / Face Size", 0.5, 4.0, 1.25, 0.05)
            move_y = st.slider("Vertical Position", -1500, 1500, 0, 10)
            move_x = st.slider("Horizontal Position", -1500, 1500, 0, 10)
            show_border = st.checkbox("Add Photo Border", value=True)
        
        # Calculate Dimensions and Crop
        sw_px, sh_px = int(target_dim[0] * PX_PER_MM), int(target_dim[1] * PX_PER_MM)
        img_w, img_h = subject_img.size
        crop_w = img_w / zoom
        crop_h = (crop_w * sh_px) / sw_px
        left = (img_w/2 + move_x) - crop_w/2
        top = (img_h/2 + move_y) - crop_h/2
        
        # Perform Crop and Final Resize
        single_photo = subject_img.crop((left, top, left + crop_w, top + crop_h))
        single_photo = single_photo.resize((sw_px, sh_px), Image.Resampling.LANCZOS)
        
        if show_border:
            draw = ImageDraw.Draw(single_photo)
            draw.rectangle([0, 0, sw_px-1, sh_px-1], outline="black", width=4)
            
        with col_prev:
            st.image(single_photo, width=220, caption="Live Biometric Preview")

    with tab3:
        st.write("### 📋 Layout & Final Output")
        col_lay, col_sheet = st.columns([1, 2])
        
        with col_lay:
            num_copies = st.number_input("Number of copies", 1, 300, 12)
            gap_h = st.slider("Horizontal Gap (mm)", 0.0, 20.0, 2.0)
            gap_v = st.slider("Vertical Gap (mm)", 0.0, 20.0, 2.0)
            margin = st.slider("Page Margin (mm)", 2.0, 30.0, 5.0)
            pdf_mode = st.selectbox("Export Color Profile", ["RGB", "CMYK"])
            
        # Canvas Generation
        p_w_mm, p_h_mm = CANVAS_SIZES[paper_choice]
        if orient == "Landscape":
            cw_px, ch_px = int(p_h_mm * PX_PER_MM), int(p_w_mm * PX_PER_MM)
        else:
            cw_px, ch_px = int(p_w_mm * PX_PER_MM), int(p_h_mm * PX_PER_MM)
        
        canvas = Image.new("RGB", (cw_px, ch_px), "white")
        m_px, gh_px, gv_px = int(margin * PX_PER_MM), int(gap_h * PX_PER_MM), int(gap_v * PX_PER_MM)
        
        curr_x, curr_y, placed = m_px, m_px, 0
        for i in range(num_copies):
            if curr_x + sw_px > cw_px - m_px:
                curr_x = m_px
                curr_y += sh_px + gv_px
            if curr_y + sh_px > ch_px - m_px: break
            canvas.paste(single_photo, (int(curr_x), int(curr_y)))
            curr_x += sw_px + gh_px
            placed += 1

        with col_sheet:
            st.image(canvas, use_container_width=True, caption=f"Final Print Sheet ({placed} Photos)")

        # --- EXPORT ACTIONS ---
        st.markdown("---")
        d1, d2, d3 = st.columns(3)
        
        buf_j = io.BytesIO()
        canvas.save(buf_j, format="JPEG", quality=100)
        d1.download_button("📥 Save as Ultra-HD JPG", buf_j.getvalue(), "Lekhmala_Studio.jpg")
        
        buf_p = io.BytesIO()
        canvas.save(buf_p, format="PNG")
        d2.download_button("📥 Save as Lossless PNG", buf_p.getvalue(), "Lekhmala_Studio.png")
        
        buf_pdf = io.BytesIO()
        pdf_canvas = canvas.copy()
        if pdf_mode == "CMYK": pdf_canvas = pdf_canvas.convert("CMYK")
        pdf_canvas.save(buf_pdf, format="PDF", resolution=600.0)
        d3.download_button(f"📥 Save as {pdf_mode} PDF", buf_pdf.getvalue(), "Lekhmala_Studio.pdf")

else:
    st.warning("👋 Namaste! Please upload a photo in **Tab 1** to start your professional studio session.")

# --- FOOTER ---
st.markdown("<div class='footer'>Lekhmala Photo Studio v2.5 | Optimized for Nepal Gov Standards | Developed by Bishal Mehta</div>", unsafe_allow_html=True)
