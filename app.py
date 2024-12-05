import streamlit as st
from ReIDSearcher import ReIDSearcher 

# App Configuration
st.set_page_config(page_title="Person Search Using Re-ID Model", page_icon="🔍", layout="centered")

# Title and Description
st.title("🔍 Person Search Using Re-ID Model")
st.markdown("""This app enables you to find a person in a gallery dataset using a Re-ID model. 
Simply upload the query image and specify the gallery folder.""")

# Upload Section
st.subheader("Upload Query Image and Enter Gallery Path")
col1, col2 = st.columns([2, 3])

with col1:
    query_image = st.file_uploader("Upload a Query Image", type=["jpg", "jpeg", "png"])
    gallery_path = st.text_input("Gallery Folder Path:", placeholder="e.g., /path/to/gallery/")

with col2:
    if query_image:
        st.image(query_image, caption="Query Image", width=150)

# Process Button and Results
if st.button("🔍 Start Search"):
    if not query_image or not gallery_path:
        st.error("⚠️ Please provide both a query image and gallery folder path.")
    else:
        # Simulate search results (this should be replaced with backend API integration)
        config_file = "configs/msmt17/swin_base.yml"
        searcher = ReIDSearcher(config_file=config_file, weight_path='weights/swin_base_msmt17.pth')
        top_matches = searcher.search_top_k_matches(query_image, gallery_path, k=9)
        
        # Display query image and dummy processing
        st.success("✅ Processing complete!")
        st.markdown("### Results")
        
        # Display results in grid format
        cols = st.columns(4)
        # Remove one query image
        top_matches = top_matches[1:]
        for idx, (img, score) in enumerate(top_matches):
            img_path = gallery_path + '/' + img
            with cols[idx % 4]:  # Cycle through columns
                st.image(img_path, caption=f"Rank {idx + 1}\nScore: {score:.4f}", width=120)

# Style Enhancements
st.markdown("""
    <style>
        /* Reduce padding for compact layout */
        .css-18e3th9 {
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
        }
        .css-1d391kg {
            padding: 10px;
            background-color: #f9f9f9;
        }
        h1, h2, h3, p {
            color: #2c3e50;
        }
    </style>
""", unsafe_allow_html=True)
