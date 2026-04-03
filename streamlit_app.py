import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu

# ==========================================
# 1. Page Config & Professional Styling
# ==========================================
st.set_page_config(layout="wide", page_title="Food AI Pro", initial_sidebar_state="collapsed")

# Restore the sleek fade-in and the "expanding" menu logic
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Smooth Fade-in for the whole app */
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .main .block-container { animation: fadeIn 0.6s ease-out; padding-top: 1.5rem; }

    /* Custom CSS to help the menu feel more 'centered' and premium */
    .nav-link {
        border: 1px solid rgba(0,0,0,0.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. THE ANIMATED MENU (Fixes Space & Adds Expansion)
# ==========================================
# This uses the 'flex' logic to ensure the selected item expands and text appears
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
            "font-size": "0px", # Hides text by default
            "color": "transparent",
            "text-align": "center", 
            "margin": "0px 8px", 
            "padding": "12px", 
            "border-radius": "10px", 
            "flex": "1", # Fills the empty space
            "transition": "all 0.4s cubic-bezier(0.4, 0, 0.2, 1)", # THE EXPANSION ANIMATION
            "display": "flex",
            "justify-content": "center",
            "align-items": "center",
            "gap": "10px"
        },
        "nav-link-selected": {
            "font-size": "16px", # Reveals text when clicked
            "color": "white", 
            "background-color": "#ff4b4b",
            "flex": "2.5", # Makes the active tab much wider
            "font-weight": "600"
        },
    }
)

# ==========================================
# 3. Smart Data Loading
# ==========================================
@st.cache_data
def load_data():
    try:
        # Load your specific CSV, cleaning out the metadata rows
        df = pd.read_csv('Embedded-images.csv', header=0, skiprows=[1, 2])
        return df
    except Exception as e:
        st.error(f"Data Error: {e}")
        return None

@st.cache_resource
def load_ai_brains():
    # Load two versions: one for 'looking' (features) and one for 'naming' (labels)
    base = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    full = tf.keras.applications.EfficientNetB0(weights='imagenet')
    return base, full

df = load_data()
base_model, full_model = load_ai_brains()

# ==========================================
# 4. Mode 1: Dashboard (Sleek Visuals)
# ==========================================
if app_mode == "Analysis Dashboard":
    st.title("📊 System Analytics")
    if df is not None:
        import plotly.express as px
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.sunburst(df, path=['category', 'Cluster'], title="Category vs AI Clusters")
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.histogram(df, x="category", color="category", title="Dataset Distribution")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Please upload your Embedded-images.csv to see the dashboard.")

# ==========================================
# 5. Mode 2: Food Browser (The 4-Column Grid)
# ==========================================
elif app_mode == "Food Browser":
    st.title("🍔 Menu Gallery")
    if df is not None:
        categories = sorted(df['category'].unique())
        choice = st.selectbox("Filter by Category:", categories)
        
        filtered = df[df['category'] == choice].head(16)
        
        # 4-column layout fixes the "Giant Image" problem on desktop
        cols_per_row = 4
        for i in range(0, len(filtered), cols_per_row):
            columns = st.columns(cols_per_row)
            for j, col in enumerate(columns):
                if i + j < len(filtered):
                    item = filtered.iloc[i + j]
                    with col:
                        if os.path.exists(item['image']):
                            st.image(item['image'], use_container_width=True)
                        else:
                            st.image("https://via.placeholder.com/300", use_container_width=True)
                        st.caption(f"**{item['image name']}**")
    else:
        st.error("Dataset not found.")

# ==========================================
# 6. Mode 3: Hybrid AI (Fixes the Baked Potato)
# ==========================================
elif app_mode == "Hybrid Classifier":
    st.title("🤖 Intelligent Classifier")
    st.markdown("This model cross-references your CSV data to ensure maximum accuracy.")
    
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    if file and df is not None:
        col_img, col_res = st.columns([1, 1])
        img = Image.open(file).convert('RGB')
        col_img.image(img, use_container_width=True, caption="Target Image")
        
        with st.spinner("Hybrid Logic processing..."):
            # Prepare image for AI
            input_img = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img.resize((224, 224))), axis=0)
            input_pre = tf.keras.applications.efficientnet.preprocess_input(input_img)
            
            # 1. Ask the AI what it 'sees' (Features)
            uploaded_features = base_model.predict(input_pre)
            
            # 2. Compare it to your CSV columns (n0, n1, n2...)
            n_cols = [c for c in df.columns if c.startswith('n') and c[1:].isdigit()]
            dataset_features = df[n_cols].values
            
            # 3. Find the most similar image in your dataset
            similarities = cosine_similarity(uploaded_features, dataset_features)
            match_idx = np.argmax(similarities)
            match_data = df.iloc[match_idx]
            match_score = similarities[0][match_idx]
            
            with col_res:
                st.subheader("AI Verdict")
                # If the image looks 70%+ like your baked potatoes, we call it a baked potato!
                if match_score > 0.70:
                    st.success(f"Confirmed: **{match_data['category']}**")
                    st.info(f"Verified against item: {match_data['image name']}")
                else:
                    # Fallback to generic AI if no match is found in your data
                    preds = full_model.predict(input_pre)
                    label = tf.keras.applications.efficientnet.decode_predictions(preds, top=1)[0][0][1]
                    st.warning(f"Unfamiliar item. AI thinks: {label.replace('_', ' ')}")
                
                st.metric("Visual Match Confidence", f"{match_score*100:.1f}%")
                st.progress(float(match_score))
                
                with st.expander("Show Closest Dataset Reference"):
                    st.write("The AI matched your upload with this reference from your CSV:")
                    if os.path.exists(match_data['image']):
                        st.image(match_data['image'], width=200)
