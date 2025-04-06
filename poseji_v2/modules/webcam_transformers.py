import cv2
import numpy as np
from streamlit_webrtc import VideoTransformerBase
from modules.helpers import calculate_angle

class WebcamPostureFeedbackTransformer(VideoTransformerBase):
    def __init__(self, analyzer, threshold, enable_alerts, alert_sensitivity):
        self.analyzer = analyzer
        self.threshold = threshold
        self.enable_alerts = enable_alerts
        self.alert_sensitivity = alert_sensitivity
        self.frame_counter = 0
        self.cached_metrics = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        height, width = img.shape[:2]
        font_scale = height / 720 * 0.8
        thickness = max(1, int(height / 720))
        points = self.analyzer.detect_pose(img, self.threshold)
        img = self.analyzer.draw_pose(img, points, self.threshold)
        
        # Update cached metrics every 10 frames
        self.frame_counter += 1
        if self.frame_counter % 10 == 0:
            self.cached_metrics = self.analyzer.calculate_body_metrics(points)
        
        metrics = self.cached_metrics if self.cached_metrics is not None else {}
        if metrics and "torso_alignment" in metrics:
            deviation = abs(180 - metrics["torso_alignment"])
            # Choose border color based on deviation
            border_color = (0, 0, 255) if deviation > self.alert_sensitivity else (0, 255, 0)
            # Draw a border around the frame with adaptive thickness
            cv2.rectangle(img, (0, 0), (width - 1, height - 1), border_color, thickness * 4)
            # Overlay feedback if deviation is high
            if deviation > self.alert_sensitivity and self.enable_alerts:
                cv2.putText(img, "Adjust your posture!", (int(50 * font_scale), int(50 * font_scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
            cv2.putText(img, f"Torso Alignment: {metrics['torso_alignment']:.1f}°", (int(50 * font_scale), int(90 * font_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, border_color, thickness)
        return img

##############################
# Advanced Webcam Posture Feedback Transformer
##############################
class WebcamPostureFeedbackTransformer(VideoTransformerBase):
    def __init__(self, analyzer, threshold, enable_alerts, alert_sensitivity):
        self.analyzer = analyzer
        self.threshold = threshold
        self.enable_alerts = enable_alerts
        self.alert_sensitivity = alert_sensitivity

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        height, width = img.shape[:2]
        font_scale = height / 720 * 0.8
        thickness = max(1, int(height / 720))
        points = self.analyzer.detect_pose(img, self.threshold)
        img = self.analyzer.draw_pose(img, points, self.threshold)
        metrics = self.analyzer.calculate_body_metrics(points)
        if metrics and "torso_alignment" in metrics:
            deviation = abs(180 - metrics["torso_alignment"])
            # Choose border color based on deviation
            border_color = (0, 0, 255) if deviation > self.alert_sensitivity else (0, 255, 0)
            # Draw a border around the frame with adaptive thickness
            cv2.rectangle(img, (0, 0), (width - 1, height - 1), border_color, thickness * 4)
            # Overlay feedback and metric information
            if deviation > self.alert_sensitivity and self.enable_alerts:
                cv2.putText(img, "Adjust your posture!", (int(50 * font_scale), int(50 * font_scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
            cv2.putText(img, f"Torso Alignment: {metrics['torso_alignment']:.1f}°", (int(50 * font_scale), int(90 * font_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, border_color, thickness)
        return img

##############################
# Advanced Exercise Analysis Transformer (Squat Analysis)
##############################
class ExerciseAnalysisTransformer(VideoTransformerBase):
    def __init__(self, analyzer, threshold):
        self.analyzer = analyzer
        self.threshold = threshold
        self.rep_count = 0
        self.in_squat = False
        self.squat_progress = 0  # Percentage of squat depth (0 to 100)
        self.last_angle = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        height, width = img.shape[:2]
        font_scale = height / 720 * 0.8
        thickness = max(1, int(height / 720))
        points = self.analyzer.detect_pose(img, self.threshold)
        img = self.analyzer.draw_pose(img, points, self.threshold)
        
        # Compute squat angle using left knee if available
        angle = None
        if points is not None:
            # Check for MediaPipe keypoints: left hip=23, left knee=25, left ankle=27
            if len(points) > 27 and points[23] and points[25] and points[27]:
                angle = calculate_angle(points[23], points[25], points[27])
            # Otherwise, check for MoveNet indices: left hip=11, left knee=13, left ankle=15
            elif len(points) > 15 and points[11] and points[13] and points[15]:
                angle = calculate_angle(points[11], points[13], points[15])
        
        if angle is not None:
            self.last_angle = angle
            # Determine squat state: when the knee angle is very low (< 90°), consider it a squat.
            if angle < 90 and not self.in_squat:
                self.in_squat = True
            # When angle recovers near standing (> 160°) and previously in squat, count a rep.
            if angle > 160 and self.in_squat:
                self.rep_count += 1
                self.in_squat = False
            # Map the knee angle to a squat progress (90° -> 100%, 180° -> 0%)
            progress = np.clip((180 - angle) / (180 - 90) * 100, 0, 100)
            self.squat_progress = progress
            
            # Draw a progress bar on the frame with adaptive sizes
            bar_x, bar_y, bar_w, bar_h = int(50 * font_scale), int(100 * font_scale), int(200 * font_scale), int(20 * font_scale)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            filled_w = int((progress / 100) * bar_w)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 255, 0), -1)
            cv2.putText(img, f"{progress:.0f}%", (bar_x + bar_w + int(10 * font_scale), bar_y + bar_h - int(5 * font_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # Overlay rep count and squat angle on the frame
            cv2.putText(img, f"Squat Reps: {self.rep_count}", (int(50 * font_scale), int(70 * font_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
            cv2.putText(img, f"Knee Angle: {angle:.1f}°", (int(50 * font_scale), int(140 * font_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        else:
            # If no angle is detected, inform the user on the frame.
            cv2.putText(img, "Squat angle not detected", (int(50 * font_scale), int(70 * font_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
        return img

##############################
# Advanced Pushup Analysis Transformer
##############################
class PushupAnalysisTransformer(VideoTransformerBase):
    def __init__(self, analyzer, threshold):
        self.analyzer = analyzer
        self.threshold = threshold
        self.rep_count = 0
        self.in_pushup = False
        self.last_elbow_angle = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        height, width = img.shape[:2]
        font_scale = height / 720 * 0.8
        thickness = max(1, int(height / 720))
        points = self.analyzer.detect_pose(img, self.threshold)
        img = self.analyzer.draw_pose(img, points, self.threshold)
        
        # Compute pushup elbow angle using both elbows if available.
        # MediaPipe: left shoulder=11, left elbow=13, left wrist=15; right shoulder=12, right elbow=14, right wrist=16.
        angle_left = None
        angle_right = None
        if points is not None:
            if len(points) > 16 and points[11] and points[13] and points[15]:
                angle_left = calculate_angle(points[11], points[13], points[15])
            if len(points) > 16 and points[12] and points[14] and points[16]:
                angle_right = calculate_angle(points[12], points[14], points[16])
        
        # Use average of both angles if available.
        if angle_left is not None and angle_right is not None:
            angle = (angle_left + angle_right) / 2
        else:
            angle = angle_left if angle_left is not None else angle_right

        if angle is not None:
            self.last_elbow_angle = angle
            # For pushups, when elbows flex below 90° consider down phase.
            if angle < 90 and not self.in_pushup:
                self.in_pushup = True
            # When elbows extend above 160° and previously in pushup, count a rep.
            if angle > 160 and self.in_pushup:
                self.rep_count += 1
                self.in_pushup = False
            cv2.putText(img, f"Pushup Reps: {self.rep_count}", (int(50 * font_scale), int(70 * font_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
            cv2.putText(img, f"Elbow Angle: {angle:.1f}°", (int(50 * font_scale), int(140 * font_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        else:
            cv2.putText(img, "Pushup angle not detected", (int(50 * font_scale), int(70 * font_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
        return img

##############################
# Advanced Pullup Analysis Transformer
##############################
class PullupAnalysisTransformer(VideoTransformerBase):
    def __init__(self, analyzer, threshold):
        self.analyzer = analyzer
        self.threshold = threshold
        self.rep_count = 0
        self.in_pullup = False
        self.last_elbow_angle = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        height, width = img.shape[:2]
        font_scale = height / 720 * 0.8
        thickness = max(1, int(height / 720))
        points = self.analyzer.detect_pose(img, self.threshold)
        img = self.analyzer.draw_pose(img, points, self.threshold)
        
        # Compute pullup elbow angle using both elbows.
        # For pullups, the elbows remain flexed more. We use similar keypoints as pushups.
        angle_left = None
        angle_right = None
        if points is not None:
            if len(points) > 16 and points[11] and points[13] and points[15]:
                angle_left = calculate_angle(points[11], points[13], points[15])
            if len(points) > 16 and points[12] and points[14] and points[16]:
                angle_right = calculate_angle(points[12], points[14], points[16])
        if angle_left is not None and angle_right is not None:
            angle = (angle_left + angle_right) / 2
        else:
            angle = angle_left if angle_left is not None else angle_right

        if angle is not None:
            self.last_elbow_angle = angle
            # For pullups, assume a rep is counted when elbows reach a low threshold (<70°) and then extend (>150°).
            if angle < 70 and not self.in_pullup:
                self.in_pullup = True
            if angle > 150 and self.in_pullup:
                self.rep_count += 1
                self.in_pullup = False
            cv2.putText(img, f"Pullup Reps: {self.rep_count}", (int(50 * font_scale), int(70 * font_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
            cv2.putText(img, f"Elbow Angle: {angle:.1f}°", (int(50 * font_scale), int(140 * font_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        else:
            cv2.putText(img, "Pullup angle not detected", (int(50 * font_scale), int(70 * font_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
        return img
