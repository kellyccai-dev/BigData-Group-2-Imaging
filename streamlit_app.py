import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
import os

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(layout="wide", page_title="Food Image Classifier Dashboard")
st.title("🍔 Interactive Food Image Classification Dashboard")

# ==========================================
# 2. Data Loading Function
# ==========================================
@st.cache_data
def load_data():
    try:
        # Orange CSV files have metadata in rows 1 and 2, so we skip them for clean pandas importing
        df = pd.read_csv('Embedded-images.csv', header=0, skiprows=[1, 2])
        
        # Sort clusters logically (C1, C2, ..., C11) for better chart visuals
        if 'Cluster' in df.columns:
            df['Cluster_num'] = df['Cluster'].str.extract('(\d+)').astype(int)
            df = df.sort_values('Cluster_num').drop(columns=['Cluster_num'])
        return df
    except Exception as e:
        return None

df = load_data()

# ==========================================
# 3. Sidebar Setup & Instructions
# ==========================================
with st.sidebar:
    st.title("Main Navigation")
    app_mode = st.radio("Choose the view", ["📊 Interactive Model Analysis", "🍽️ Food Image Browser (The Menu)"])
    
    st.markdown("---")
    st.title("⚙️ Repository Setup Instructions")
    st.markdown("""
    To make this app work perfectly on Streamlit Cloud, ensure your GitHub repository looks like this:
    - `streamlit_app.py` (This code)
    - `requirements.txt` (List of libraries)
    - `Embedded-images.csv` (The data)
    - `Burger/` (Folder containing burger images)
    - `Pizza/` (Folder containing pizza images)
    - *(...and so on for the other categories)*
    """)

# ==========================================
# 4. Mode 1: Interactive Model Analysis (The 5 KPIs)
# ==========================================
if app_mode == "📊 Interactive Model Analysis":
    if df is None:
        st.error("⚠️ **Data not found!** Please ensure `Embedded-images.csv` is uploaded to your GitHub repository in the same folder as this script.")
    else:
        st.markdown("### Interactive Model Performance & Interpretation Metrics")
        st.markdown("These charts are generated dynamically from your Orange output data. **Hover your mouse over the plots to interact with them!**")
        
        st.write("---")
        st.subheader("Section A: Cluster Overview & Composition")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # KPI 1: Cluster Dot Plot (Stripplot)
            fig1 = px.strip(df, x='Cluster', y='category', color='category', 
                            hover_data=['image name'], stripmode='overlay',
                            title="KPI 1: Cluster Membership Dot Plot")
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            # KPI 2: 100% Stacked Bar Composition
            composition = df.groupby(['Cluster', 'category']).size().reset_index(name='count')
            composition['Percentage'] = composition.groupby('Cluster')['count'].transform(lambda x: x / x.sum() * 100)
            fig2 = px.bar(composition, x='Cluster', y='Percentage', color='category', 
                          title="KPI 2: Category Breakdown (100% Stacked Bar)")
            fig2.update_layout(yaxis_title="Percentage (%)")
            st.plotly_chart(fig2, use_container_width=True)

        st.write("---")
        st.subheader("Section B: Metric Analysis Deep Dive")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # KPI 3: Heatmap of Categories to Clusters
            heatmap_data = df.groupby(['category', 'Cluster']).size().unstack(fill_value=0)
            fig3 = px.imshow(heatmap_data, text_auto=True, aspect="auto", color_continuous_scale='Blues',
                             title="KPI 3: Category to Cluster Heatmap")
            st.plotly_chart(fig3, use_container_width=True)
            
        with col4:
            # KPI 4: Image Dimensions and File Sizes (Feature Metrics)
            fig4 = px.scatter(df, x='width', y='height', color='category', size='size',
                              hover_data=['image name'],
                              title="KPI 4: Image Dimensions vs. File Size")
            st.plotly_chart(fig4, use_container_width=True)

        st.write("---")
        st.subheader("Section C: Classification Similarity Map")
        
        # KPI 5: PCA Mapping (Replaces the static Dendrogram)
        st.markdown("""
        #### **KPI 5: 2D Semantic Similarity Map (Interactive)**
        *Note: This replaces the static Orange dendrogram with a far more advanced interactive visual.* We use an algorithm called PCA to project the 2,048 mathematical features the machine learned for each image down into an interactive 2D map. **Images visually clustered close together here share structural similarities according to the model.**
        """)
        
        # Extract the machine learning embedding features (n0 to n2047)
        feature_cols = [col for col in df.columns if col.startswith('n') and col[1:].isdigit()]
        if feature_cols:
            features = df[feature_cols]
            pca = PCA(n_components=2) # Compress to 2D
            pca_result = pca.fit_transform(features)
            
            df_pca = df.copy()
            df_pca['PCA1'] = pca_result[:, 0]
            df_pca['PCA2'] = pca_result[:, 1]
            
            fig5 = px.scatter(df_pca, x='PCA1', y='PCA2', color='Cluster', symbol='category',
                              hover_data=['image name', 'category', 'width', 'height'],
                              title="Interactive Cluster Similarity Map",
                              height=600)
            fig5.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.warning("Machine learning feature columns (n0-n2047) not found in the dataset.")

# ==========================================
# 5. Mode 2: Food Image Browser (The Menu)
# ==========================================
elif app_mode == "🍽️ Food Image Browser (The Menu)":
    st.header("Search & Visualize Food Menu")
    
    if df is None:
        st.error("⚠️ Please upload `Embedded-images.csv` to enable the menu features.")
    else:
        # Pull categories directly from your dataset
        categories = sorted(df['category'].unique().tolist())
        search_category = st.selectbox("Select a Food Category to View:", categories)
        
        st.subheader(f"Menu Items: {search_category}")
        
        # Filter dataframe for selected category
        cat_df = df[df['category'] == search_category]
        
        if len(cat_df) == 0:
            st.info("No images found for this category in the dataset.")
        else:
            images_to_show = cat_df.head(12) # Shows a grid of up to 12 images
            
            columns_per_row = 4
            num_rows = -(-len(images_to_show) // columns_per_row)
            
            for r in range(num_rows):
                cols = st.columns(columns_per_row)
                for c in range(columns_per_row):
                    idx = r * columns_per_row + c
                    if idx < len(images_to_show):
                        row_data = images_to_show.iloc[idx]
                        
                        # The CSV already contains the relative path (e.g., 'Burger/Burger 1.png')
                        img_path = row_data['image'] 
                        img_name = row_data['image name']
                        cluster = row_data['Cluster']
                        
                        with cols[c]:
                            # Automatically links the file paths from your CSV to your GitHub folders!
                            if os.path.exists(img_path):
                                st.image(img_path, use_column_width=True)
                            else:
                                st.image("https://via.placeholder.com/300x300.png?text=Upload+Images+To+GitHub", use_column_width=True)
                            
                            st.caption(f"**{img_name}** | AI Assigned: {cluster}")
