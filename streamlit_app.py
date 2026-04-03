import streamlit as st
import pandas as pd
import os

from streamlit_option_menu import option_menu

# ==========================================
# 1. Page Configuration & Smooth CSS Animation
# ==========================================
st.set_page_config(layout="wide", page_title="Food Image Classifier Dashboard", initial_sidebar_state="collapsed")

# Hide Streamlit's default menus and add a fade-in animation
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .main .block-container {
        animation: fadeIn 0.5s ease-out;
        padding-top: 2rem; 
    }
</style>
""", unsafe_allow_html=True)

st.title("🍔 Interactive Food Image Classification")

# ==========================================
# 2. Sleek, Animated Top Navigation
# ==========================================
app_mode = option_menu(
    menu_title=None, 
    options=["Analysis Dashboard", "Food Browser", "Live AI Classifier"], 
    icons=["bar-chart-line-fill", "grid-fill", "cpu-fill"], 
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "8px!important", 
            "background-color": "#f8f9fa", 
            "border-radius": "10px", 
            "margin-bottom": "20px",
            "display": "flex", 
            "width": "100%",   
        },
        "icon": {
            "color": "#ff4b4b", 
            "font-size": "22px", 
        }, 
        "nav-link": {
            "font-size": "0px", 
            "color": "transparent",
            "text-align": "center", 
            "margin": "0px 5px", 
            "padding": "12px 10px", 
            "border-radius": "8px", 
            "flex": "1", 
            "transition": "all 0.4s cubic-bezier(0.4, 0, 0.2, 1)", 
            "display": "flex",
            "justify-content": "center",
            "align-items": "center",
            "gap": "10px", 
            "--hover-color": "#e9ecef"
        },
        "nav-link-selected": {
            "font-size": "15px", 
            "color": "white", 
            "background-color": "#ff4b4b",
            "flex": "2.5", 
        },
    }
)

# ==========================================
# 3. Data Loading & Heavy Math (Cached)
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
# 4. Mode 1: Interactive Model Analysis 
# ==========================================
if app_mode == "Analysis Dashboard":
    if df is None:
        st.error("⚠️ **Data not found!** Please ensure `Embedded-images.csv` is uploaded to your working directory.")
    else:
        import plotly.express as px
        
        st.markdown("### Model Performance & Metrics")
        st.write("---")
        
        st.subheader("Section A: Cluster Overview & Composition")
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.strip(df, x='Cluster', y='category', color='category', 
                            hover_data=['image name'], stripmode='overlay',
                            title="KPI 1: Cluster Membership")
            fig1.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig1, width="stretch")
            
        with col2:
            composition = df.groupby(['Cluster', 'category']).size().reset_index(name='count')
            composition['Percentage'] = composition.groupby('Cluster')['count'].transform(lambda x: x / x.sum() * 100)
            fig2 = px.bar(composition, x='Cluster', y='Percentage', color='category', 
                          title="KPI 2: Category Breakdown")
            fig2.update_layout(yaxis_title="Percentage (%)", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig2, width="stretch")

        st.write("---")
        st.subheader("Section B: Metric Analysis Deep Dive")
        col3, col4 = st.columns(2)
        with col3:
            heatmap_data = df.groupby(['category', 'Cluster']).size().unstack(fill_value=0)
            fig3 = px.imshow(heatmap_data, text_auto=True, aspect="auto", color_continuous_scale='Blues',
                             title="KPI 3: Heatmap")
            fig3.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig3, width="stretch")
            
        with col4:
            fig4 = px.scatter(df, x='width', y='height', color='category', size='size',
                              hover_data=['image name'],
                              title="KPI 4: Image Dimensions vs. Size")
            fig4.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig4, width="stretch")

        st.write("---")
        st.subheader("Section C: Semantic Similarity Map")
        df_pca = get_pca_data(df)
        
        if df_pca is not None:
            fig5 = px.scatter(df_pca, x='PCA1', y='PCA2', color='Cluster', symbol='category',
                              hover_data=['image name', 'category', 'width', 'height'],
                              title="KPI 5: 2D Projection Map")
            fig5.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
            fig5.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig5, width="stretch")

# ==========================================
# 5. Mode 2: Food Image Browser (The Menu)
# ==========================================
elif app_mode == "Food Browser":
    st.header("Search Food Menu")
    
    if df is None:
        st.error("⚠️ Please upload `Embedded-images.csv`.")
    else:
        categories = sorted(df['category'].unique().tolist())
        search_category = st.selectbox("Select a Food Category to View:", categories)
        
        st.subheader(f"Items: {search_category}")
        cat_df = df[df['category'] == search_category]
        
        if len(cat_df) == 0:
            st.info("No images found for this category.")
        else:
            images_to_show = cat_df.head(12) 
            
            # 4 Columns so they fit beautifully on a PC!
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
                                st.image(img_path, width="stretch")
                            else:
                                st.image("https://via.placeholder.com/300x300.png?text=Image", width="stretch")
                            
                            st.caption(f"**{img_name}** | AI: {cluster}")

# ==========================================
# 6. Mode 3: Live AI Image Classifier
# ==========================================
elif app_mode == "Live AI Classifier":
    st.header("Live AI Classification")
    st.markdown("Upload a picture of food, and our upgraded AI will analyze it.")
    
    import tensorflow as tf
    from PIL import Image
    import numpy as np
    
    # Upgraded AI Model for better feature extraction
    @st.cache_resource(show_spinner="Booting up the AI Brain (EfficientNetB0)...")
    def load_model():
        model = tf.keras.applications.EfficientNetB0(weights='imagenet')
        return model
        
    model = load_model()
    
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        st.write("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, width="stretch")
            
        with col2:
            st.subheader("🤖 Predictions")
            with st.spinner("Analyzing pixels..."):
                try:
                    img_resized = image.resize((224, 224))
                    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
                    
                    predictions = model.predict(img_array)
                    decoded_preds = tf.keras.applications.efficientnet.decode_predictions(predictions, top=3)[0]
                    
                    st.success("Analysis Complete!")
                    for i, (imagenet_id, label, probability) in enumerate(decoded_preds):
                        st.write(f"**#{i+1}: {label.replace('_', ' ').title()}**")
                        st.progress(float(probability))
                        st.caption(f"Confidence: {probability * 100:.2f}%")
                        
                except Exception as e:
                    st.error(f"Error processing image: {e}")
