# Advanced Human Pose Estimation 🤖

This project is a **Pose Estimation Application** designed to provide comprehensive pose analysis and insights into biomechanics. It utilizes cutting-edge machine learning models for accurate pose detection and analysis, offering users a powerful tool for applications like sports analysis 🏃‍♂️, fitness tracking 💪, and more.

---

Try Now : https://humanpose-estimation-apps.streamlit.app/

---
## Features ✨

- **Real-time Pose Detection:** Supports analysis for images and videos.
- **Customizable Settings:** Adjust confidence thresholds and choose between analysis modes.
- **User-Friendly Interface:** Simple drag-and-drop functionality for uploading media files.
- **Detailed Output:** Visualized key points and skeletal connections for accurate pose estimation.
- **Output Formats:** Provides both visual and JSON-based output for further integration.

---

## Project Structure 🗂️

```plaintext
EDUNET_AI_INTERNSHIP_2025
├── Demo
│   ├── int.txt
│   ├── images
│   │   ├── OutPut-image.png
│   │   ├── pose-gif.gif
│   │   ├── run.jpg
│   │   ├── stand.jpg
│   └── models
│       └── graph_opt.pb
├── src
│   ├── app.py  # Streamlit app entry point
│   ├── pose_estimation_Video.py  # Script for video-based pose detection
│   ├── pose_estimation.py  # Script for image-based pose detection
│
├── videos
│   ├── output.mov
│   ├── run.mov
│   └── run1.mp4
├── requirements.txt
├── README.md  # Project documentation
```

---

## Model Details 🤖

The application uses TensorFlow's pre-trained **PoseNet model** for pose detection. The key features of this model include:

- **Keypoint Detection:** Identifies major joints such as elbows, knees, and shoulders.
- **Skeletal Visualization:** Draws connections between keypoints to create a skeletal representation.
- **Optimized for Performance:** Runs efficiently on both CPU and GPU.
- **Scalability:** Can be extended to multiple people detection in a single frame.

---

## App Interface 🎨

The interface is designed to be intuitive and user-friendly:

### Pose Analysis Settings ⚙️
- **Confidence Threshold:** Adjustable slider to control the accuracy of detected poses.
- **Analysis Mode:** Dropdown menu to toggle between:
  - Basic Pose Detection
  - Advanced Analysis (e.g., angles, postural evaluation).

### Upload Section 📤
- Drag and drop your file (images: PNG, JPG, JPEG) into the designated area.
- A clear set of instructions is displayed to guide users:
  1. Click the "Browse Files" button.
  2. Select an appropriate file for analysis.
  3. Ensure the file is clear and high-quality for best results.

![App Interface](Demo/Advanced-Pose-Estimation_interference.png)

---

## Demo 🎥

- **Image Input:** Example results of pose estimation from input images.
## Demo - Video Input 🎥

[Watch Demo](Edunet_AI_internship_2025/Demo/streamlit-app-2025-01-26-20-01-13.webm)



---

## How to Run the App 🚀

### Prerequisites 🛠️
- Python 3.8 or higher
- Install dependencies from `requirements.txt` using:

```bash
pip install -r requirements.txt
```

### Running the Application 💻
1. Navigate to the `src` directory.
2. Start the Streamlit app:

```bash
streamlit run app.py
```

3. Open the provided local or network URL in your browser to interact with the app.

---

## Use Cases 🌟

- **Sports Training:** Analyze athletes' movements for performance optimization 🏋️‍♂️.
- **Rehabilitation:** Track patients' recovery progress through biomechanical insights 🩺.
- **Fitness Tracking:** Enhance workout sessions by providing feedback on postures 🏃‍♀️.
- **Gaming:** Integrate pose estimation into AR/VR games for an interactive experience 🎮.

---

## Future Enhancements 🔮

- **Real-time Webcam Support:** Enable live pose detection through webcams.
- **Advanced Insights:** Include postural corrections and biomechanical analysis.
- **Mobile Compatibility:** Optimize for use on mobile devices 📱.
- **Multi-Person Detection:** Extend support for detecting multiple subjects in a single frame.
- **Integration APIs:** Offer REST APIs for third-party integration 🔗.

---

## Contributors 🤝
This application is part of the Edunet AI Internship 2025 program. Contributions to improve and extend its functionality are welcome.

---

## License 📜
This project is licensed under the Apache License 2.0. See the LICENSE file for more details.

