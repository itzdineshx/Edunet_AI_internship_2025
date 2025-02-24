import streamlit as st
import pandas as pd
from modules.video_estimation import share_session

def display_session_history():
    st.header("Session History")
    if "session_history" in st.session_state and st.session_state.session_history:
        history_df = pd.DataFrame([
            {"Session Type": s["session_type"],
             "Timestamp": s["timestamp"],
             "Metrics": s["metrics"]} 
            for s in st.session_state.session_history
        ])
        st.dataframe(history_df)
        st.download_button("Download History CSV",
                           history_df.to_csv(index=False).encode("utf-8"),
                           "session_history.csv", "text/csv")
        if st.button("Share Session"):
            share_session()
    else:
        st.info("No sessions saved yet.")
