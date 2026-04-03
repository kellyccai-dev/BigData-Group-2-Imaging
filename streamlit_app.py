import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu

# ==========================================
# 1. Page Config & Professional UI
# ==========================================
st.set_page_config(layout="wide", page_title="Food AI Pro Dashboard", initial_sidebar_state="collapsed")

# Restore sleek fade-in and premium styling
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .main .block-container { animation: fadeIn 0.6s ease-out; padding-top: 1.5rem; }

    .nav-link {
        border: 1px solid rgba(0,0,0,0.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. THE ANIMATED EXPANDING MENU
# ==========================================
app_mode = option_menu(
    menu_title=None, 
    options=["Analysis Dashboard", "Food Browser", "Hybrid Classifier"], 
    icons=["bar-chart-line", "grid", "cpu"], 
    orientation="horizontal",
    styles={
        "container": {
            "padding": "10px!important", 
            "background-color": "#ffffff", 
            "border-radius": "12px", 
            "box-shadow": "0px 4px 12px rgba(0,0,0,0.05)",
            "margin-bottom": "25px"
        },
        "icon": {"color": "#ff4b4b", "font-size": "20px"}, 
        "nav-link": {
            "font-size": "0px", # Hides text by default for the "expansion" effect
            "color": "transparent",
            "text-align": "center", 
            "margin": "0px 8px", 
            "padding": "12px", 
            "border-radius": "10px", 
            "flex": "1", # Fills empty space
            "transition": "all 0.4s cubic-bezier(0.4, 0, 0.2, 1)", 
            "display": "flex",
            "justify-content": "center",
            "align-items": "center",
            "gap": "10px"
        },
        "nav-link-selected": {
            "font-size": "16px", # Reveals text on selection
            "color": "white", 
            "background-color": "#ff4b4b",
            "flex": "2.5", # Expands the tab
            "font-weight": "600"
        },
    }
)

# ==========================================
# 3. Data & Hybrid Model Loading
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Embedded-images.csv', header=0, skiprows=[1, 2])
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

@st.cache_resource
def load_ai_brains():
    # Base extracts features (vibe-check), Full gives generic names
    base = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    full = tf.keras.applications.EfficientNetB0(weights='imagenet')
    return base, full

df = load_data()
base_model, full_model = load_ai_brains()

# ==========================================
# 4. Mode 1: Dashboard
# ==========================================
if app_mode == "Analysis Dashboard":
    st.title("📊 System Analytics")
    if df is not None:
        import plotly.express as px
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.sunburst(df, path=['category', 'Cluster'], title="Category vs AI Clusters")
            st.plotly_chart(fig1, width="stretch")
        with c2:
            fig2 = px.histogram(df, x="category", color="category", title="Dataset Distribution")
            st.plotly_chart(fig2, width="stretch")

# ==========================================
# 5. Mode 2: Food Browser (4-Column Grid)
# ==========================================
elif app_mode == "Food Browser":
    st.title("🍔 Menu Gallery")
    if df is not None:
        categories = sorted(df['category'].unique())
        choice = st.selectbox("Filter by Category:", categories)
        filtered = df[df['category'] == choice].head(16)
        
        cols_per_row = 4
        for i in range(0, len(filtered), cols_per_row):
            columns = st.columns(cols_per_row)
            for j, col in enumerate(columns):
                if i + j < len(filtered):
                    item = filtered.iloc[i + j]
                    with col:
                        # Streamlit automatically handles webp in st.image
                        img_path = item['image']
                        if os.path.exists(img_path):
                            st.image(img_path, width="stretch")
                        else:
                            st.image("https://via.placeholder.com/300", width="stretch")
                        st.caption(f"**{item['image name']}**")

# ==========================================
# 6. Mode 3: Hybrid AI (The "Potato" Fix + WEBP)
# ==========================================
elif app_mode == "Hybrid Classifier":
    st.title("🤖 Pro Hybrid Classifier")
    st.markdown("Supports **JPG, PNG, and WEBP**. Cross-references your dataset for precision.")
    
    # ADDED: webp support in the file uploader
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "webp"])
    
    if file and df is not None:
        col_img, col_res = st.columns([1, 1])
        img = Image.open(file).convert('RGB')
        col_img.image(img, width="stretch", caption="Uploaded Image")
        
        with st.spinner("Analyzing visual fingerprints..."):
            # Prepare image
            input_img = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img.resize((224, 224))), axis=0)
            input_pre = tf.keras.applications.efficientnet.preprocess_input(input_img)
            
            # 1. Feature Extraction (The Vibe Check)
            uploaded_features = base_model.predict(input_pre)
            
            # 2. Similarity Search against CSV embeddings (n0, n1, n2...)
            n_cols = [c for c in df.columns if c.startswith('n') and c[1:].isdigit()]
            dataset_features = df[n_cols].values
            
            similarities = cosine_similarity(uploaded_features, dataset_features)
            match_idx = np.argmax(similarities)
            match_data = df.iloc[match_idx]
            match_score = similarities[0][match_idx]
            
            with col_res:
                st.subheader("Classification Result")
                
                # High confidence uses your CSV label (Fixed Baked Potato logic)
                if match_score > 0.65:
                    st.success(f"**Category: {match_data['category']}**")
                    st.info(f"Verified against: {match_data['image name']}")
                else:
                    # Fallback to general AI labels
                    preds = full_model.predict(input_pre)
                    label = tf.keras.applications.efficientnet.decode_predictions(preds, top=1)[0][0][1]
                    st.warning(f"Uncertain Match. AI Guess: {label.replace('_', ' ').title()}")
                
                st.metric("Visual Similarity Score", f"{match_score*100:.1f}%")
                st.progress(float(match_score))
                
                with st.expander("Show Closest Match from Dataset"):
                    st.write("This is the item in your CSV that the AI thinks looks most similar:")
                    if os.path.exists(match_data['image']):
                        st.image(match_data['image'], width=200)
                    else:
                        st.write("(Path not found in local storage)")
