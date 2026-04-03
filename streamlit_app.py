import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================
st.set_page_config(layout="wide", page_title="Hybrid Food AI", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    @keyframes fadeIn { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }
    .main .block-container { animation: fadeIn 0.5s ease-out; padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Animated Navigation Menu
# ==========================================
app_mode = option_menu(
    menu_title=None, 
    options=["Analysis Dashboard", "Food Browser", "Hybrid Classifier"], 
    icons=["bar-chart-line-fill", "grid-fill", "cpu-fill"], 
    orientation="horizontal",
    styles={
        "container": {"padding": "8px!important", "background-color": "#f8f9fa", "border-radius": "10px", "width": "100%"},
        "nav-link": {"font-size": "0px", "flex": "1", "transition": "all 0.4s ease", "display": "flex", "justify-content": "center", "align-items": "center", "border-radius": "8px"},
        "nav-link-selected": {"font-size": "15px", "background-color": "#ff4b4b", "flex": "2.5", "color": "white"}
    }
)

# ==========================================
# 3. Data & Model Loading (Cached)
# ==========================================
@st.cache_data
def load_data():
    try:
        # Load data, skipping metadata rows if they exist in your CSV
        df = pd.read_csv('Embedded-images.csv', header=0, skiprows=[1, 2])
        return df
    except: return None

@st.cache_resource
def load_hybrid_models():
    # Model 1: Headless (extracts numbers/features)
    base = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    # Model 2: Full (gives generic labels)
    full = tf.keras.applications.EfficientNetB0(weights='imagenet')
    return base, full

df = load_data()
base_model, full_model = load_hybrid_models()

# ==========================================
# 4. Mode 1: Analysis Dashboard
# ==========================================
if app_mode == "Analysis Dashboard":
    st.subheader("Model Performance & Metrics")
    if df is not None:
        import plotly.express as px
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.strip(df, x='Cluster', y='category', color='category', title="Cluster Membership")
            st.plotly_chart(fig1, width="stretch")
        with col2:
            comp = df.groupby(['Cluster', 'category']).size().reset_index(name='count')
            fig2 = px.bar(comp, x='Cluster', y='count', color='category', title="Category Breakdown")
            st.plotly_chart(fig2, width="stretch")
    else:
        st.error("CSV not found.")

# ==========================================
# 5. Mode 2: Food Browser (Grid Fix)
# ==========================================
elif app_mode == "Food Browser":
    st.header("Search Food Menu")
    if df is not None:
        search_cat = st.selectbox("Category:", sorted(df['category'].unique()))
        cat_df = df[df['category'] == search_cat].head(12)
        
        cols_per_row = 4 # Perfectly sized for PC
        for i in range(0, len(cat_df), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(cat_df):
                    row = cat_df.iloc[i + j]
                    with col:
                        if os.path.exists(row['image']): st.image(row['image'], width="stretch")
                        else: st.image("https://via.placeholder.com/150", width="stretch")
                        st.caption(f"**{row['image name']}**")

# ==========================================
# 6. Mode 3: Hybrid Classifier (The "Potato" Fix)
# ==========================================
elif app_mode == "Hybrid Classifier":
    st.header("Hybrid AI Classifier")
    st.info("This AI combines ImageNet logic with your specific CSV data for better accuracy.")
    
    uploaded_file = st.file_uploader("Upload food photo...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and df is not None:
        col1, col2 = st.columns(2)
        img = Image.open(uploaded_file).convert('RGB')
        col1.image(img, width="stretch", caption="Your Upload")
        
        with st.spinner("Analyzing and cross-referencing your dataset..."):
            # Prepare image
            img_resized = img.resize((224, 224))
            img_arr = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_batch = np.expand_dims(img_arr, axis=0)
            img_pre = tf.keras.applications.efficientnet.preprocess_input(img_batch)
            
            # 1. Get Generic Prediction
            preds = full_model.predict(img_pre)
            generic_label = tf.keras.applications.efficientnet.decode_predictions(preds, top=1)[0][0][1]
            
            # 2. Get Visual Fingerprint (Embedding)
            features = base_model.predict(img_pre)
            
            # 3. Similarity Search against CSV
            feat_cols = [c for c in df.columns if c.startswith('n') and c[1:].isdigit()]
            csv_feats = df[feat_cols].values
            sims = cosine_similarity(features, csv_feats)
            best_idx = np.argmax(sims)
            match = df.iloc[best_idx]
            score = sims[0][best_idx]
            
            with col2:
                st.subheader("Final Verdict")
                # Logic: If it looks like something in our CSV, trust that. 
                if score > 0.65:
                    st.success(f"**Result: {match['category']}**")
                    st.write(f"Matched with: *{match['image name']}*")
                else:
                    st.warning(f"AI Suggestion: {generic_label.replace('_', ' ').title()}")
                
                st.write(f"**Data Match Confidence:** {score*100:.1f}%")
                st.progress(float(score))
                
                with st.expander("Why this result?"):
                    st.write("The AI found this reference in your dataset had the closest visual features:")
                    if os.path.exists(match['image']): st.image(match['image'], width=150)
