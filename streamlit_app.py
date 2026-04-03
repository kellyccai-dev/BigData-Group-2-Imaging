import streamlit as st
from PIL import Image
import os
import random

# ==========================================
# 1. Page Configuration & Title
# ==========================================
st.set_page_config(layout="wide", page_title="Food Image Classifier Dashboard")
st.title("🍔 Food Image Classification & Analysis Dashboard")

# Define all categories seen in the analytical charts to populate search options.
ALL_CATEGORIES = ["Baked Potato", "Burger", "Crispy Chicken", "Donut", "Fries", "Hot Dog", "Pizza", "Sandwich", "Taco", "Taquito"]

# ==========================================
# 2. Sidebar Navigation and Key Instructions
# ==========================================
with st.sidebar:
    st.image(Image.open('image_0.png'), use_column_width=True) # A small preview of analysis
    st.title("Main Navigation")
    app_mode = st.radio("Choose the view", ["📊 Model Analysis (The 5 KPIs)", "🍽️ Food Image Browser (The Menu)"])
    
    st.markdown("---")
    st.title("⚙️ Data Integration Instructions")
    st.markdown("""
    This section is critical for making the **Food Image Browser** work.
    Because the full dataset of images is not directly accessible to this app,
    you must organize your local project or GitHub repository following this structure:
    """)
    st.text("""
your_project/
├── streamlit_app.py  <-- This file
├── image_0.png      <-- The KPIs
├── ...
│
└── food_dataset/    <-- (REQUIRED) New folder
    ├── Burger/        <-- Subfolder per category
    │   ├── burger1.jpg
    │   ├── burger2.png
    ├── Pizza/
    │   └── pizza1.jpeg
    └── ... (folders for ALL food types)
    """)
    st.info("💡 Make sure to uncompress your zip file and place the category folders inside a new folder named `food_dataset` at the top level.")

# ==========================================
# 3. Mode 1: Model Analysis (The 5 KPIs)
# ==========================================
if app_mode == "📊 Model Analysis (The 5 KPIs)":
    st.header("Model Performance & Interpretation Metrics")
    st.markdown("""
    This section replicates the five key analytical outputs from the machine learning process.
    Use these metrics to understand the classifier's performance and cluster-based reasoning.
    """)

    st.write("---")
    st.subheader("Section A: Cluster Overview & Composition")
    
    # KPIs 1 & 2 side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### **KPI 1: Cluster Membership Dot Plot**")
        try:
            image_0 = Image.open('image_0.png')
            st.image(image_0, caption="Visual mapping of individual food categories to clusters C1-C11. Shows initial cluster distribution.", use_column_width=True)
        except FileNotFoundError:
            st.error("Missing file: Please ensure `image_0.png` is in your repository.")

    with col2:
        st.write("#### **KPI 2: Cluster Category Breakdown (100% Stacked Bar)**")
        try:
            image_1 = Image.open('image_1.png')
            st.image(image_1, caption="Detailed percentage composition breakdown of food categories within each identified cluster.", use_column_width=True)
        except FileNotFoundError:
            st.error("Missing file: Please ensure `image_1.png` is in your repository.")

    st.write("---")
    st.subheader("Section B: Metric Analysis Deep Dive")

    # KPIs 3 & 4 stacked, with space
    st.write("#### **KPIs 3 & 4: Detailed Feature & Distance Metrics per Category**")
    st.markdown("""
    These charts break down specific model metrics (like feature importance scores, average cluster distances, or classification probabilities) for each food category.
    Values are presented in parentheses for direct reading, with bars visualizing the scale and direction (negative to positive).
    """)
    
    col3, col4 = st.columns([1, 10]) # Use a thin left column for vertical alignment
    with col4:
        try:
            image_2 = Image.open('image_2.png')
            st.image(image_2, caption="KPI 3: Detailed metrics for first set of categories.", use_column_width=True)
        except FileNotFoundError:
            st.error("Missing file: Please ensure `image_2.png` is in your repository.")

        st.write("<br>", unsafe_allow_html=True) # spacer

        try:
            image_3 = Image.open('image_3.png')
            st.image(image_3, caption="KPI 4: Detailed metrics for remaining categories.", use_column_width=True)
        except FileNotFoundError:
            st.error("Missing file: Please ensure `image_3.png` is in your repository.")

    st.write("---")
    st.subheader("Section C: Classification Hierarchy")
    
    # KPI 5
    st.write("#### **KPI 5: Classification Dendrogram and Cluster Map**")
    try:
        image_4 = Image.open('image_4.png')
        st.image(image_4, caption="Hierarchical clustering tree (dendrogram) visualizing the similarity relationships between individual image instances and the identified clusters C1-C11. This visualizes the *why* of the classifications.", use_column_width=True)
    except FileNotFoundError:
        st.error("Missing file: Please ensure `image_4.png` is in your repository.")

# ==========================================
# 4. Mode 2: Food Image Browser (The Menu)
# ==========================================
elif app_mode == "🍽️ Food Image Browser (The Menu)":
    st.header("Search & Visualize Food Categories")
    st.markdown("""
    Enter a food category keyword or select from the list below to simulate an online menu.
    This feature browses the image dataset used to train the model, displaying representative images.
    """)

    # Dropdown and text input for searching
    search_category = st.selectbox("Select or Search Food Category:", ALL_CATEGORIES, index=1)
    
    st.subheader(f"Viewing Images from Category: {search_category}")

    # --- INTEGRATION MECHANISM: LOADING FROM FOLDER ---
    # Define the required base path
    dataset_root = "food_dataset"
    # Create the full path to the specific category subfolder
    category_dir = os.path.join(dataset_root, search_category.replace(" ", "_")) # Handling potential space-to-underscore naming

    # Logic to handle user image integration status
    if not os.path.exists(dataset_root):
        st.warning("⚠️ **Image Dataset Missing:** This section requires actual images to display.")
        st.info("💡 **Ready to Integrate Your Images?** Please follow the structure instructions in the sidebar. Create a 'food_dataset' folder and follow the category/image structure to populate this section.")
        st.image("https://via.placeholder.com/800x400.png?text=Waiting+for+Food+Image+Data", caption="Placeholder: Waiting for Data Integration", use_column_width=True)
        
    elif not os.path.exists(category_dir):
        st.warning(f"⚠️ **Category Folder Missing:** The folder for **{search_category}** was not found.")
        st.info(f"💡 Create a folder named exactly '{search_category.replace(' ', '_')}' inside 'food_dataset' and add your images to see them here.")
        
    else:
        # User has integrated data correctly
        # 1. Fetch all image files from the folder
        image_files = [f for f in os.listdir(category_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            st.info("This category folder exists but contains no image files. Please add images (.png, .jpg).")
        else:
            # 2. Limit the display (e.g., show 8 images to keep the screen clean)
            images_to_show = image_files[:8]
            
            # 3. Create a clean grid layout (4 images per row)
            columns_per_row = 4
            num_images = len(images_to_show)
            num_rows = -(-num_images // columns_per_row) # Ceiling division

            for r in range(num_rows):
                cols = st.columns(columns_per_row)
                for c in range(columns_per_row):
                    img_idx = r * columns_per_row + c
                    if img_idx < num_images:
                        img_path = os.path.join(category_dir, images_to_show[img_idx])
                        try:
                            with cols[c]:
                                st.image(img_path, use_column_width=True)
                                st.caption(images_to_show[img_idx]) # File name as caption
                        except Exception as e:
                            with cols[c]:
                                st.error(f"Error loading image: {images_to_show[img_idx]}. Check file type.")

# ==========================================
# 5. Footer & About Section
# ==========================================
st.markdown("---")
with st.expander("About This Dashboard"):
    st.write("""
    This website is a demonstration dashboard built with Streamlit.
    - **Machine Learning Output:** The analytical visuals presented are screenshots of the completed Orange machine learning project, demonstrating performance on an image classification task with 11 clusters.
    - **Image Browser (Simulated Menu):** This feature uses a folder-based integration mechanism to browse the training dataset. See the sidebar instructions on how to integrate your unzipped image data into this section.
    """)
    st.write("---")
    st.write("**⚠️ Data Integration Reminder for Repository Owner:**")
    st.info("Remember to add your images from the unzipped file to a new 'food_dataset' folder structured as `food_dataset/Category_Name/image.jpg` at the top level of your GitHub repo for the image browser to work.")
