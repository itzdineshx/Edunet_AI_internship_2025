import cv2
from streamlit_webrtc import VideoTransformerBase

class WebcamPoseTransformer(VideoTransformerBase):
    def __init__(self, analyzer, threshold):
        self.analyzer = analyzer
        self.threshold = threshold

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        points = self.analyzer.detect_pose(img, self.threshold)
        img = self.analyzer.draw_pose(img, points, self.threshold)
        return img

class WebcamPostureFeedbackTransformer(VideoTransformerBase):
    def __init__(self, analyzer, threshold, enable_alerts, alert_sensitivity):
        self.analyzer = analyzer
        self.threshold = threshold
        self.enable_alerts = enable_alerts
        self.alert_sensitivity = alert_sensitivity

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        points = self.analyzer.detect_pose(img, self.threshold)
        img = self.analyzer.draw_pose(img, points, self.threshold)
        metrics = self.analyzer.calculate_body_metrics(points)
        if metrics and "torso_alignment" in metrics and self.enable_alerts:
            deviation = abs(180 - metrics["torso_alignment"])
            if deviation > self.alert_sensitivity:
                cv2.putText(img, "Adjust your posture!", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return img

class ExerciseAnalysisTransformer(VideoTransformerBase):
    def __init__(self, analyzer, threshold):
        self.analyzer = analyzer
        self.threshold = threshold
        self.rep_count = 0
        self.in_squat = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        points = self.analyzer.detect_pose(img, self.threshold)
        img = self.analyzer.draw_pose(img, points, self.threshold)
        if hasattr(self.analyzer, '_calculate_angle') and points[8] and points[9] and points[10]:
            angle = self.analyzer._calculate_angle(points[8], points[9], points[10])
            if angle < 90 and not self.in_squat:
                self.in_squat = True
            if angle > 160 and self.in_squat:
                self.rep_count += 1
                self.in_squat = False
            cv2.putText(img, f"Squat Reps: {self.rep_count}", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        return img
