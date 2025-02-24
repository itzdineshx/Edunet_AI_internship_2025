import streamlit as st
import os
import sys
import absl.logging

def set_config():
    st.set_page_config(
        page_title="PoseJi", 
        page_icon="/workspaces/Edunet_AI_internship_2025/assets/images/poseji-logo.png", 
        layout="wide"
    )
    # Set environment variables and logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    absl.logging.set_verbosity(absl.logging.ERROR)
    sys.stderr = open(os.devnull, 'w')

def inject_custom_css():
    st.markdown("""
    <style>
    /* Main background */
    .main { 
        background-color: #f8f9fa;
    }
    /* Sidebar customizations */
    .css-1d391kg {
        padding: 1rem;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #2c3e50, #3498db);
        color: white;
        padding: 2rem;
    }
    .sidebar .sidebar-content h1, .sidebar .sidebar-content h2, .sidebar .sidebar-content h3 {
        color: #f8f9fa;
    }
    .css-18e3th9 {
        padding: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
