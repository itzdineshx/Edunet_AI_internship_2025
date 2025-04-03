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
    # Suppressing unwanted logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    absl.logging.set_verbosity(absl.logging.ERROR)
    sys.stderr = open(os.devnull, 'w')

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
    # Suppressing unwanted logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    absl.logging.set_verbosity(absl.logging.ERROR)
    sys.stderr = open(os.devnull, 'w')

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
    # Suppressing unwanted logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    absl.logging.set_verbosity(absl.logging.ERROR)
    sys.stderr = open(os.devnull, 'w')

def inject_custom_css():
    st.markdown("""
    <style>
    /* Smooth Gradient Background */
    .main {
        background: linear-gradient(135deg, #F8F9FA, #E3E7EC);
        color: #333333;
        font-family: 'Inter', sans-serif;
    }

    /* Frosted Glass Sidebar */
    .css-1d391kg {
        padding: 1rem;
    }

    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.5);
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Elegant Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: #2C3E50;
    }

    /* Softly Elevated Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #6C7A89, #3B4B5A);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.15);
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #3B4B5A, #6C7A89);
        transform: translateY(-2px);
        box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.2);
    }

    /* Modern Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }

    ::-webkit-scrollbar-track {
        background: #E3E7EC;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: #B0BEC5;
        border-radius: 10px;
    }

    /* Input Fields */
    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.8);
        color: #333333;
        border-radius: 8px;
        padding: 10px;
        border: 1px solid rgba(0, 0, 0, 0.1);
    }

    .stTextInput>div>div>input:focus {
        border: 1px solid #6C7A89;
        box-shadow: 0px 0px 10px rgba(108, 122, 137, 0.5);
    }

    </style>
    """, unsafe_allow_html=True)
