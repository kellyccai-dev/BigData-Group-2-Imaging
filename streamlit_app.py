import os
# MUST BE AT THE VERY TOP: Silences the TensorFlow GPU & CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from streamlit_option_menu import option_menu

# ==========================================
# 1. Page Config & Professional UI
# ==========================================
st.set_page_config(layout="wide", page_title="Food AI Pro Dashboard", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .main .block-container { animation: fadeIn 0.6s ease-out; padding-top: 1.5rem; }
    .nav-link { border: 1px solid rgba(0,0,0,0.05) !important; }
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
        "container": {"padding": "10px!important", "background-color": "#ffffff", "border-radius": "12px", "box-shadow": "0px 4px 12px rgba(0,0,0,0.05)", "margin-bottom": "25px"},
        "icon": {"color": "#ff4b4b", "font-size": "20px"}, 
        "nav-link": {"font-size": "0px", "color": "transparent", "text-align": "center", "margin": "0px 8px", "padding": "12px", "border-radius": "10px", "flex": "1", "transition": "all 0.4s cubic-bezier(0.4, 0, 0.2, 1)", "display": "flex", "justify-content": "center", "align-items": "center", "gap": "10px"},
        "nav-link-selected": {"font-size": "16px", "color": "white", "background-color": "#ff4b4b", "flex": "2.5", "font-weight": "600"},
    }
)

# ==========================================
# 3. Data Loading & ResNet50 Brains
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Embedded-images.csv', header=0, skiprows=[1, 2])
        if 'Cluster' in df.columns:
            df['Cluster_num'] = df['Cluster'].str.extract(r'(\d+)').astype(float)
            df = df.sort_values('Cluster_num').drop(columns=['Cluster_num'])
        return df
    except Exception as e:
        return None

@st.cache_data
def get_pca_data(df):
    feature_cols = [col for col in df.columns if col.startswith('n') and col[1:].isdigit()]
    if not feature_cols:
        return None
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[feature_cols])
    df_pca = df[['image name', 'category', 'Cluster', 'width', 'height']].copy()
    df_pca['PCA1'] = pca_result[:, 0]
    df_pca['PCA2'] = pca_result[:, 1]
    return df_pca

@st.cache_resource
def load_ai_brains():
    # SWAPPED TO RESNET50: Outputs exactly 2048 features to perfectly match your CSV dimensions
    base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
    full = tf.keras.applications.ResNet50(weights='imagenet')
    return base, full

df = load_data()
base_model, full_model = load_ai_brains()

# ==========================================
# 4. Mode 1: Dashboard (ALL 5 KPIs RESTORED)
# ==========================================
if app_mode == "Analysis Dashboard":
    if df is None:
        st.error("⚠️ Data not found! Please upload `Embedded-images.csv`.")
    else:
        import plotly.express as px
        
        st.markdown("### Model Performance & Metrics")
        st.write("---")
        
        st.subheader("Section A: Cluster Overview & Composition")
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.strip(df, x='Cluster', y='category', color='category', hover_data=['image name'], stripmode='overlay', title="KPI 1: Cluster Membership")
            st.plotly_chart(fig1, width="stretch")
            
        with col2:
            composition = df.groupby(['Cluster', 'category']).size().reset_index(name='count')
            composition['Percentage'] = composition.groupby('Cluster')['count'].transform(lambda x: x / x.sum() * 100)
            fig2 = px.bar(composition, x='Cluster', y='Percentage', color='category', title="KPI 2: Category Breakdown")
            st.plotly_chart(fig2, width="stretch")

        st.write("---")
        st.subheader("Section B: Metric Analysis Deep Dive")
        col3, col4 = st.columns(2)
        with col3:
            heatmap_data = df.groupby(['category', 'Cluster']).size().unstack(fill_value=0)
            fig3 = px.imshow(heatmap_data, text_auto=True, aspect="auto", color_continuous_scale='Blues', title="KPI 3: Heatmap")
            st.plotly_chart(fig3, width="stretch")
            
        with col4:
            fig4 = px.scatter(df, x='width', y='height', color='category', size='size', hover_data=['image name'], title="KPI 4: Image Dimensions vs. Size")
            st.plotly_chart(fig4, width="stretch")

        st.write("---")
        st.subheader("Section C: Semantic Similarity Map")
        df_pca = get_pca_data(df)
        if df_pca is not None:
            fig5 = px.scatter(df_pca, x='PCA1', y='PCA2', color='Cluster', symbol='category', hover_data=['image name', 'category'], title="KPI 5: 2D PCA Projection Map")
            fig5.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(fig5, width="stretch")

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
                        # STRICT ERROR HANDLING: Prevents MediaFileStorageError
                        try:
                            img_path = str(item['image'])
                            if os.path.exists(img_path):
                                st.image(img_path, use_container_width=True)
                            else:
                                st.image("https://via.placeholder.com/300?text=Image+Missing", use_container_width=True)
                        except:
                            st.image("https://via.placeholder.com/300?text=Load+Error", use_container_width=True)
                            
                        st.caption(f"**{item['image name']}**")

# ==========================================
# 6. Mode 3: Hybrid AI (Top-3 Voting System)
# ==========================================
elif app_mode == "Hybrid Classifier":
    st.title("🤖 Pro Hybrid Classifier")
    st.markdown("Supports **JPG, PNG, and WEBP**. Now using Top-3 Consensus Voting.")
    
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "webp"])
    
    if file and df is not None:
        col_img, col_res = st.columns([1, 1])
        img = Image.open(file).convert('RGB')
        col_img.image(img, use_container_width=True, caption="Uploaded Image")
        
        with st.spinner("Analyzing 2,048 visual parameters..."):
            img_resized = img.resize((224, 224))
            input_img = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img_resized), axis=0)
            input_pre = tf.keras.applications.resnet50.preprocess_input(input_img)
            
            # 1. Feature Extraction
            uploaded_features = base_model.predict(input_pre)
            
            # 2. Similarity Search against CSV
            n_cols = [c for c in df.columns if c.startswith('n') and c[1:].isdigit()]
            dataset_features = df[n_cols].values
            
            # Get similarities for all images
            similarities = cosine_similarity(uploaded_features, dataset_features)[0]
            
            # 3. GET THE TOP 3 MATCHES INSTEAD OF JUST 1
            top_3_idx = similarities.argsort()[-3:][::-1]
            top_3_matches = df.iloc[top_3_idx]
            top_3_scores = similarities[top_3_idx]
            
            best_match = top_3_matches.iloc[0]
            best_score = top_3_scores[0]
            
            # Find the most common category among the Top 3
            consensus_category = top_3_matches['category'].mode()[0]
            
            with col_res:
                st.subheader("Classification Result")
                
                # LOWERED THRESHOLD & ADDED CONSENSUS LOGIC
                if best_score > 0.45 or consensus_category == best_match['category']:
                    st.success(f"**Category: {consensus_category}**")
                    st.info(f"Verified against top dataset matches.")
                else:
                    preds = full_model.predict(input_pre)
                    label = tf.keras.applications.resnet50.decode_predictions(preds, top=1)[0][0][1]
                    st.warning(f"Uncertain Match. AI Guess: {label.replace('_', ' ').title()}")
                
                st.metric("Top Visual Similarity Score", f"{best_score*100:.1f}%")
                st.progress(float(min(best_score, 1.0)))
                
                with st.expander("See the Top 3 Matches from your Dataset"):
                    st.write("The AI found these three images visually closest to your upload:")
                    
                    # Show the top 3 images side-by-side
                    match_cols = st.columns(3)
                    for i in range(3):
                        with match_cols[i]:
                            try:
                                match_img_path = str(top_3_matches.iloc[i]['image'])
                                if os.path.exists(match_img_path):
                                    st.image(match_img_path, use_container_width=True)
                                else:
                                    st.image("https://via.placeholder.com/150", use_container_width=True)
                                st.caption(f"{top_3_matches.iloc[i]['category']} ({top_3_scores[i]*100:.0f}%)")
                            except:
                                st.write("Error loading image")
