import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
from modules.image_analysis import run_image_analysis

def compare_pose_images(analyzer, threshold, model_choice):
    st.header("Comparison Mode")
    st.markdown("Upload **two images** to compare their pose analysis side by side.", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file1 = st.file_uploader("Upload First Image", type=['jpg', 'jpeg', 'png'], key='comp_img1')
    with col2:
        uploaded_file2 = st.file_uploader("Upload Second Image", type=['jpg', 'jpeg', 'png'], key='comp_img2')
    
    if uploaded_file1 is not None and uploaded_file2 is not None:
        try:
            image1 = Image.open(uploaded_file1).convert('RGB')
            image2 = Image.open(uploaded_file2).convert('RGB')
        except Exception as e:
            st.error(f"Error processing images: {e}")
            return
        
        with st.spinner("Processing first image..."):
            pose_image1, metrics_df1, metrics1 = run_image_analysis(
                analyzer, image1, threshold, model_choice, "Detailed Metrics", unique_key="img1"
            )
        with st.spinner("Processing second image..."):
            pose_image2, metrics_df2, metrics2 = run_image_analysis(
                analyzer, image2, threshold, model_choice, "Detailed Metrics", unique_key="img2"
            )
        
        st.subheader("Pose Estimation Comparison")
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(pose_image1, caption="Image 1 Pose", use_container_width=True)
            st.markdown("**Metrics:**")
            st.write(metrics1)
        with col_img2:
            st.image(pose_image2, caption="Image 2 Pose", use_container_width=True)
            st.markdown("**Metrics:**")
            st.write(metrics2)
        
        st.subheader("Detailed Metrics Comparison Table")
        col_table1, col_table2 = st.columns(2)
        with col_table1:
            st.markdown("**Image 1 Metrics**")
            st.dataframe(metrics_df1)
        with col_table2:
            st.markdown("**Image 2 Metrics**")
            st.dataframe(metrics_df2)
    else:
        st.info("Please upload both images to compare.")

# --- For standalone testing ---
if __name__ == "__main__":
    st.set_page_config(page_title="Pose Comparison", layout="wide")
    st.title("Comparison Mode Demo")

    # For demonstration, we create a dummy analyzer.
    # Replace this with your actual analyzer instance.
    class DummyAnalyzer:
        def detect_pose(self, image, threshold):
            # Dummy implementation: return an array with 33 None values.
            return [None] * 33
        def draw_pose(self, image, points, threshold):
            # Dummy implementation: return the original image.
            return image
        def calculate_body_metrics(self, points):
            # Dummy metrics for demonstration.
            return {"dummy_metric": 0}
    
    analyzer = DummyAnalyzer()
    model_choice = "MediaPipe"
    threshold = 0.5

    compare_pose_images(analyzer, threshold, model_choice)
