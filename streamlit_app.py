import streamlit as st
import pandas as pd
import os
import time 

# ==========================================
# 1. Page Configuration & Navigation CSS
# ==========================================
st.set_page_config(layout="wide", page_title="Food Image Classifier Dashboard")

# Inject custom CSS specifically targeting the Sidebar Expanders
st.markdown("""
<style>
    /* Make sidebar expanders look like interactive modern tiles */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background-color: #f1f3f6;
        border: 1px solid #dcdcdc;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 1.05em;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    [data-testid="stSidebar"] .streamlit-expanderHeader:hover {
        background-color: #e2e6eb;
    }
</style>
""", unsafe_allow_html=True)

st.title("🍔 Interactive Food Image Classification Dashboard")

# ==========================================
# 2. Data Loading & Heavy Math (Cached)
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Embedded-images.csv', header=0, skiprows=[1, 2])
        df = df.copy() 
        if 'Cluster' in df.columns:
            df['Cluster_num'] = df['Cluster'].str.extract(r'(\d+)').astype(float)
            df = df.sort_values('Cluster_num').drop(columns=['Cluster_num'])
            df = df.copy() 
        return df
    except Exception as e:
        return None

@st.cache_data
def get_pca_data(df):
    from sklearn.decomposition import PCA
    feature_cols = [col for col in df.columns if col.startswith('n') and col[1:].isdigit()]
    if not feature_cols:
        return None
    features = df[feature_cols]
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    
    df_pca = df[['image name', 'category', 'Cluster', 'width', 'height']].copy()
    df_pca['PCA1'] = pca_result[:, 0]
    df_pca['PCA2'] = pca_result[:, 1]
    return df_pca

df = load_data()

# ==========================================
# 3. Animated Tile Sidebar Navigation
# ==========================================
# We use session state so the buttons inside the tiles can drive the page changes
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Analysis"

def set_page(page_name):
    st.session_state.app_mode = page_name

with st.sidebar:
    st.title("Main Navigation")
    st.caption("Click the 🔽 arrow on a tile to reveal its full section name and select it!")

    # Tile 1: Analysis
    is_analysis = (st.session_state.app_mode == "Analysis")
    with st.expander("📊 Analysis Dashboard", expanded=is_analysis):
        st.markdown("#### Interactive Model Analysis")
        st.write("View the 5 key mathematical metrics and the 2D visual similarity map.")
        st.button("Launch Analysis", on_click=set_page, args=("Analysis",), use_container_width=True, disabled=is_analysis)

    # Tile 2: Browser
    is_menu = (st.session_state.app_mode == "Browser")
    with st.expander("🍽️ Food Browser", expanded=is_menu):
        st.markdown("#### Food Image Browser (The Menu)")
        st.write("Search and visualize your dataset images grouped by category.")
        st.button("Launch Browser", on_click=set_page, args=("Browser",), use_container_width=True, disabled=is_menu)

    # Tile 3: Live AI Classifier
    is_ai = (st.session_state.app_mode == "AI")
    with st.expander("📸 Live AI Model", expanded=is_ai):
        st.markdown("#### Live AI Image Classifier")
        st.write("Upload custom WebP/PNG/JPG images for real-time deep learning classification.")
        st.button("Launch AI Classifier", on_click=set_page, args=("AI",), use_container_width=True, disabled=is_ai)

    st.markdown("---")
    with st.expander("⚙️ Repository Instructions", expanded=False):
        st.markdown("""
        Ensure your GitHub repository looks like this:
        - `streamlit_app.py` 
        - `requirements.txt` 
        - `Embedded-images.csv` 
        - `Burger/` (Images folder)
        - `Pizza/` (Images folder)
        """)

# ==========================================
# Animation Buffer: Smooth page transition
# ==========================================
app_mode = st.session_state.app_mode
with st.spinner("Loading module..."):
    time.sleep(0.3) 

# ==========================================
# 4. Mode 1: Interactive Model Analysis 
# ==========================================
if app_mode == "Analysis":
    if df is None:
        st.error("⚠️ **Data not found!** Please ensure `Embedded-images.csv` is uploaded to your GitHub repository.")
    else:
        import plotly.express as px
        
        st.markdown("### Interactive Model Performance & Interpretation Metrics")
        st.markdown("These charts are generated dynamically from your Orange output data. **Hover your mouse over the plots to interact with them!**")
        
        st.write("---")
        st.subheader("Section A: Cluster Overview & Composition")
        
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.strip(df, x='Cluster', y='category', color='category', 
                            hover_data=['image name'], stripmode='overlay',
                            title="KPI 1: Cluster Membership Dot Plot")
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, width="stretch")
            
        with col2:
            composition = df.groupby(['Cluster', 'category']).size().reset_index(name='count')
            composition['Percentage'] = composition.groupby('Cluster')['count'].transform(lambda x: x / x.sum() * 100)
            fig2 = px.bar(composition, x='Cluster', y='Percentage', color='category', 
                          title="KPI 2: Category Breakdown (100% Stacked Bar)")
            fig2.update_layout(yaxis_title="Percentage (%)")
            st.plotly_chart(fig2, width="stretch")

        st.write("---")
        st.subheader("Section B: Metric Analysis Deep Dive")
        
        col3, col4 = st.columns(2)
        with col3:
            heatmap_data = df.groupby(['category', 'Cluster']).size().unstack(fill_value=0)
            fig3 = px.imshow(heatmap_data, text_auto=True, aspect="auto", color_continuous_scale='Blues',
                             title="KPI 3: Category to Cluster Heatmap")
            st.plotly_chart(fig3, width="stretch")
            
        with col4:
            fig4 = px.scatter(df, x='width', y='height', color='category', size='size',
                              hover_data=['image name'],
                              title="KPI 4: Image Dimensions vs. File Size")
            st.plotly_chart(fig4, width="stretch")

        st.write("---")
        st.subheader("Section C: Classification Similarity Map")
        
        st.markdown("""
        #### **KPI 5: 2D Semantic Similarity Map (Interactive)**
        We use an algorithm called PCA to project the 2,048 mathematical features the machine learned for each image down into an interactive 2D map. **Images visually clustered close together here share structural similarities according to the model.**
        """)
        
        df_pca = get_pca_data(df)
        
        if df_pca is not None:
            fig5 = px.scatter(df_pca, x='PCA1', y='PCA2', color='Cluster', symbol='category',
                              hover_data=['image name', 'category', 'width', 'height'],
                              title="Interactive Cluster Similarity Map",
                              height=600)
            fig5.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(fig5, width="stretch")
        else:
            st.warning("Machine learning feature columns (n0-n2047) not found in the dataset.")

# ==========================================
# 5. Mode 2: Food Image Browser (The Menu)
# ==========================================
elif app_mode == "Browser":
    st.header("Search & Visualize Food Menu")
    
    if df is None:
        st.error("⚠️ Please upload `Embedded-images.csv` to enable the menu features.")
    else:
        categories = sorted(df['category'].unique().tolist())
        search_category = st.selectbox("Select a Food Category to View:", categories)
        
        st.subheader(f"Menu Items: {search_category}")
        
        cat_df = df[df['category'] == search_category]
        
        if len(cat_df) == 0:
            st.info("No images found for this category in the dataset.")
        else:
            images_to_show = cat_df.head(12) 
            
            columns_per_row = 4
            num_rows = -(-len(images_to_show) // columns_per_row)
            
            for r in range(num_rows):
                cols = st.columns(columns_per_row)
                for c in range(columns_per_row):
                    idx = r * columns_per_row + c
                    if idx < len(images_to_show):
                        row_data = images_to_show.iloc[idx]
                        
                        img_path = row_data['image'] 
                        img_name = row_data['image name']
                        cluster = row_data['Cluster']
                        
                        with cols[c]:
                            if os.path.exists(img_path):
                                st.image(img_path, use_container_width=True)
                            else:
                                st.image("https://via.placeholder.com/300x300.png?text=Upload+Images", use_container_width=True)
                            
                            st.caption(f"**{img_name}** | AI Assigned: {cluster}")

# ==========================================
# 6. Mode 3: Live AI Image Classifier
# ==========================================
elif app_mode == "AI":
    st.header("Upload a Food Image for Live AI Classification")
    st.markdown("Upload a picture of food, and our built-in deep learning model (MobileNetV2) will try to identify it in real-time!")
    
    from PIL import Image
    import numpy as np
    import tensorflow as tf
    
    uploaded_file = st.file_uploader("Choose an image file (JPG/PNG/WEBP)...", type=["jpg", "jpeg", "png", "webp"])
    
    @st.cache_resource
    def load_model():
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        return model
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Your Uploaded Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True)
            
        with col2:
            st.subheader("🤖 AI Predictions")
            with st.spinner("The AI is analyzing the pixels..."):
                try:
                    model = load_model()
                    
                    img_resized = image.resize((224, 224))
                    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
                    
                    predictions = model.predict(img_array)
                    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
                    
                    st.success("Analysis Complete!")
                    for i, (imagenet_id, label, probability) in enumerate(decoded_preds):
                        st.write(f"**#{i+1}: {label.replace('_', ' ').title()}**")
                        st.progress(float(probability))
                        st.caption(f"Confidence: {probability * 100:.2f}%")
                        
                except Exception as e:
                    st.error(f"Oops! The AI encountered an error processing this image: {e}")
