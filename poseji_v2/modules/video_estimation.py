import streamlit as st
import cv2
import time
import zipfile
import tempfile
import imageio
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image

def generate_report(original_image, pose_image, metrics_df):
    """
    Generate a ZIP report containing the original image, processed pose image, and body metrics CSV.
    """
    report_zip = BytesIO()
    with zipfile.ZipFile(report_zip, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        with BytesIO() as buffer:
            Image.fromarray(original_image).save(buffer, format='PNG')
            zip_file.writestr("original_image.png", buffer.getvalue())
        with BytesIO() as buffer:
            # Convert BGR to RGB for correct color representation
            rgb_image = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb_image).save(buffer, format='PNG')
            zip_file.writestr("pose_image.png", buffer.getvalue())
        csv_data = metrics_df.to_csv(index=False).encode('utf-8')
        zip_file.writestr("body_metrics.csv", csv_data)
    report_zip.seek(0)
    return report_zip

def save_session(metrics, session_type):
    """
    Save the current session metrics into Streamlit's session state.
    """
    if "session_history" not in st.session_state:
        st.session_state.session_history = []
    session_data = {
        "session_type": session_type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics
    }
    st.session_state.session_history.append(session_data)
    st.success("Session saved successfully!")

def share_session():
    """
    Stub function to indicate session sharing.
    """
    st.success("Session shared to cloud!")

def run_video_estimation(analyzer, video_file, threshold, record_video=False, extract_skeleton=False):
    """
    Process an uploaded video file using the given analyzer.
    
    - analyzer: Your pose estimation analyzer instance.
    - video_file: The uploaded video file.
    - threshold: Confidence threshold for keypoint detection.
    - record_video: If True, record and offer download of the processed video.
    - extract_skeleton: If True, extract a skeleton-only video.
    
    The function displays frames in real time and updates real-time metrics.
    Additionally, it records metrics (with frame number and elapsed time) and offers a CSV download.
    """
    # Create a temporary file to store the uploaded video.
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    frame_placeholder = st.empty()  # Placeholder to update video frames in real time
    recorded_frames = []
    skeleton_frames = []
    metrics_placeholder = st.empty()  # Placeholder for real-time metrics display
    
    # Try to obtain total frame count for progress estimation
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    progress_bar = st.progress(0) if total_frames > 0 else None
    frame_count = 0

    st.info("Processing video...")
    start_time = time.time()
    last_points = None
    
    # List to store metrics for each frame
    metrics_list = []

    # Process video frame-by-frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame for faster processing (adjust as needed)
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        try:
            # Detect pose keypoints
            points = analyzer.detect_pose(frame, threshold)
            last_points = points
            
            # Draw pose overlay on a copy of the frame
            overlay_frame = analyzer.draw_pose(frame.copy(), points, threshold)
            frame_rgb = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            st.error(f"Error during frame processing: {e}")
            continue
        
        # Update the frame display in real time
        frame_placeholder.image(frame_rgb, channels="RGB")
        recorded_frames.append(frame_rgb)
        
        # If skeleton extraction is enabled, process the skeleton image
        if extract_skeleton:
            if hasattr(analyzer, 'draw_skeleton'):
                skeleton_frame = analyzer.draw_skeleton(points, frame.shape)
            else:
                blank = np.zeros_like(frame)
                skeleton_frame = analyzer.draw_pose(blank.copy(), points, threshold)
            skeleton_frames.append(skeleton_frame)
        
        # Update and display real-time metrics
        metrics = analyzer.calculate_body_metrics(points)
        metrics_placeholder.markdown("**Realtime Metrics:** " + str(metrics))
        
        # Record metrics along with frame number and elapsed time
        current_time = time.time() - start_time
        row = {"frame": frame_count, "time": current_time}
        row.update(metrics)
        metrics_list.append(row)
        
        # Update progress bar if total frame count is available
        frame_count += 1
        if progress_bar:
            progress_bar.progress(min(int((frame_count / total_frames) * 100), 100))
        
        # Small delay to simulate real-time processing (adjust if needed)
        time.sleep(0.03)
    
    cap.release()
    elapsed_time = time.time() - start_time
    st.success(f"Video processing complete. Processed {frame_count} frames in {elapsed_time:.2f} seconds.")
    
    rec_video_bytes = None
    skel_video_bytes = None
    
    # Create a DataFrame for the per-frame metrics
    metrics_df = pd.DataFrame(metrics_list)
    csv_metrics = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Metrics CSV",
                       data=csv_metrics,
                       file_name="video_metrics.csv",
                       mime="text/csv")
    
    # If recording is enabled, write the processed video to file for download
    if record_video and recorded_frames:
        video_filename = "recorded_pose_video.mp4"
        imageio.mimwrite(video_filename, recorded_frames, fps=30, codec='libx264')
        with open(video_filename, "rb") as vid_file:
            rec_video_bytes = vid_file.read()
        st.download_button(label="Download Recorded Video", 
                           data=rec_video_bytes, 
                           file_name=video_filename, 
                           mime="video/mp4")
    
    # If skeleton extraction is enabled, write the skeleton video to file for download
    if extract_skeleton and skeleton_frames:
        skeleton_filename = "extracted_skeleton_video.mp4"
        imageio.mimwrite(skeleton_filename, skeleton_frames, fps=30, codec='libx264')
        with open(skeleton_filename, "rb") as vid_file:
            skel_video_bytes = vid_file.read()
        st.download_button(label="Download Skeleton Video", 
                           data=skel_video_bytes, 
                           file_name=skeleton_filename, 
                           mime="video/mp4")
    
    metrics_final = analyzer.calculate_body_metrics(last_points) if last_points is not None else {}
    return rec_video_bytes, skel_video_bytes, metrics_final
