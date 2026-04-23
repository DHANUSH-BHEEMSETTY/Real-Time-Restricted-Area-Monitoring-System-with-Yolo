"""
Streamlit Live Dashboard — MobileNet-SSD + Centroid Tracking
=============================================================
Run with:
    streamlit run streamlit_mobilenet.py

This app mirrors the YOLOv8 dashboard (streamlit_run.py) but uses:
  • MobileNet-SSD (via cv2.dnn)  for object detection
  • Centroid Tracker             for persistent object-ID assignment

Detection logs are saved to  data/detection_log_mobilenet.csv
(the YOLOv8 log is NOT touched).
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import threading
import random
import time
import os
from datetime import datetime

import pygame
from mobilenet_detector import MobileNetDetector, MOBILENET_CLASSES


# ---------------------------------------------------------------------------
# Supported classes (PASCAL-VOC subset that makes sense for area monitoring)
# ---------------------------------------------------------------------------
SUPPORTED_CLASSES = [
    "person", "car", "bicycle", "bus", "motorbike",
    "aeroplane", "bottle", "chair", "dog", "cat",
]


# ---------------------------------------------------------------------------
# App class
# ---------------------------------------------------------------------------
class MobileNetMonitoringApp:
    def __init__(self):
        self.detector      = None          # loaded in run()
        self.cap           = None
        self.restricted_area = None
        self.class_colors  = {}
        self.csv_file      = "data/detection_log_mobilenet.csv"
        self.object_entry_times = {}

        # Alert state
        self.alert_active  = False
        self.alert_thread  = None

        # Ensure data dir & CSV header exist
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.csv_file):
            pd.DataFrame(
                columns=["Timestamp", "Class", "Confidence",
                         "Restricted Area Violation", "Object ID"]
            ).to_csv(self.csv_file, index=False)

        pygame.mixer.init()

    # --- colour helper ------------------------------------------------------

    def _make_colors(self, class_list: list) -> dict:
        return {cls: tuple(random.randint(60, 230) for _ in range(3))
                for cls in class_list}

    # --- webcam -------------------------------------------------------------

    def start_webcam(self) -> bool:
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            st.error("Error: Unable to access the webcam.")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return True

    def stop_webcam(self):
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
            self.cap = None
            self.stop_alert()

    # --- alert sound --------------------------------------------------------

    def _play_loop(self, path: str):
        pygame.mixer.music.load(path)
        pygame.mixer.music.play(-1)
        while self.alert_active:
            time.sleep(0.5)
        pygame.mixer.music.stop()

    def start_alert(self, path: str = "alert.mp3"):
        if not self.alert_active:
            self.alert_active = True
            self.alert_thread = threading.Thread(
                target=self._play_loop, args=(path,), daemon=True
            )
            self.alert_thread.start()

    def stop_alert(self):
        self.alert_active = False

    # --- ROI ----------------------------------------------------------------

    def draw_roi(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        axes   = (w // 4, h // 8)
        self.restricted_area = (center, axes)
        cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 0, 255), 2)
        return frame

    def is_near_restricted_area(self, box: tuple) -> bool:
        if self.restricted_area:
            center, axes = self.restricted_area
            x1, y1, x2, y2 = box
            obj_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            dist = np.linalg.norm(np.array(center) - np.array(obj_center))
            return dist < (min(axes) + 50)
        return False

    # --- CSV logging --------------------------------------------------------

    def save_detection(self, class_name: str, confidence: float,
                       near_roi: bool, object_id: int):
        if near_roi:
            row = {
                "Timestamp":               datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Class":                   class_name,
                "Confidence":              confidence,
                "Restricted Area Violation": "Yes",
                "Object ID":               object_id,
            }
            pd.DataFrame([row]).to_csv(self.csv_file, mode="a",
                                       header=False, index=False)

    # --- main frame processor -----------------------------------------------

    def update_frame(self, confidence_threshold: float,
                     selected_classes: list,
                     alert_classes: list):
        """Read one frame, run MobileNet-SSD + Centroid Tracker, annotate."""
        if not self.cap:
            return None, []

        ret, frame = self.cap.read()
        if not ret:
            return None, []

        # ---- detection ----
        self.detector.confidence_threshold = confidence_threshold
        detections = self.detector.detect(frame, selected_classes=selected_classes)

        # ---- tracking ----
        tracked_objects = self.detector.update_tracker(detections)
        # Build a quick reverse map: centroid_tuple -> objectID
        centroid_to_id = {
            tuple(centroid): oid for oid, centroid in tracked_objects.items()
        }

        annotated = frame.copy()
        object_in_roi = False
        detected_classes = []

        for det in detections:
            class_name = det["class_name"]
            conf       = det["confidence"]
            x1, y1, x2, y2 = det["box"]
            detected_classes.append(class_name)

            # Identify tracked ID for this detection (nearest centroid)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            obj_id = None
            min_d  = float("inf")
            for centroid, oid in centroid_to_id.items():
                d = np.linalg.norm(np.array([cx, cy]) - np.array(centroid))
                if d < min_d:
                    min_d, obj_id = d, oid

            color = self.class_colors.get(class_name, (0, 200, 100))
            label = f"ID:{obj_id} {class_name} {conf:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(y1 - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # ROI check
            near_roi = self.is_near_restricted_area((x1, y1, x2, y2))
            if near_roi:
                object_in_roi = True
                cv2.putText(annotated, "⚠ Object in Restricted Area!",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2, cv2.LINE_AA)

                key = f"{class_name}_{obj_id}"
                if key not in self.object_entry_times:
                    self.object_entry_times[key] = time.time()
                if time.time() - self.object_entry_times[key] > 2:
                    self.save_detection(class_name, conf, near_roi, obj_id)
                    self.object_entry_times[key] = time.time()

        # Alert logic
        if object_in_roi and any(c in alert_classes for c in detected_classes):
            self.start_alert()
        else:
            self.stop_alert()

        annotated = self.draw_roi(annotated)

        # Overlay model info badge
        cv2.rectangle(annotated, (0, 0), (280, 26), (30, 30, 30), -1)
        cv2.putText(annotated, "MobileNet-SSD + Centroid Tracker",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 180), 1)

        return annotated, detected_classes

    # --- Streamlit UI -------------------------------------------------------

    def run(self):
        st.set_page_config(
            page_title="MobileNet-SSD Restricted Area Monitor",
            layout="wide",
            page_icon="🤖",
        )

        # Header
        st.markdown(
            """
            <h2 style='text-align:center; color:#00e6b8;'>
              🤖 MobileNet-SSD + Centroid Tracking &nbsp;|&nbsp;
              Restricted Area Monitor
            </h2>
            <p style='text-align:center; color:#999; font-size:13px;'>
              Alternative model running alongside YOLOv8 — logs saved to
              <code>data/detection_log_mobilenet.csv</code>
            </p>
            """,
            unsafe_allow_html=True,
        )

        # ---- Sidebar ----
        st.sidebar.title("🔧 MobileNet Settings")
        st.sidebar.markdown(
            "**Model:** MobileNet-SSD (Caffe)\n\n"
            "**Algorithm:** Centroid Tracking\n\n"
            "**Classes:** PASCAL-VOC 20"
        )

        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold", 0.1, 1.0, 0.4, 0.05
        )

        selected_classes = st.sidebar.multiselect(
            "Objects to Detect",
            SUPPORTED_CLASSES,
            default=["person"],
        )

        alert_classes = st.sidebar.multiselect(
            "Alert Trigger Classes",
            SUPPORTED_CLASSES,
            default=["person"],
        )

        col_start, col_stop = st.sidebar.columns(2)
        start_clicked = col_start.button("▶️ Start")
        stop_clicked  = col_stop.button("⏹️ Stop")

        if start_clicked:
            if self.start_webcam():
                st.sidebar.success("Webcam started!")

        if stop_clicked:
            self.stop_webcam()
            st.sidebar.info("Webcam stopped.")

        # ---- Main area ----
        col_feed, col_info = st.columns([3, 1])

        with col_info:
            st.markdown("### 📊 Live Stats")
            metric_violations = st.empty()
            metric_tracked    = st.empty()
            metric_fps        = st.empty()
            st.markdown("---")
            st.markdown("### 🗒️ Recent Log")
            log_table = st.empty()

        with col_feed:
            frame_ph = st.empty()

        # ---- Detection loop ----
        if self.cap:
            fps_counter = 0
            fps_timer   = time.time()
            fps_display = 0.0

            while self.cap and self.cap.isOpened():
                result = self.update_frame(
                    confidence_threshold, selected_classes, alert_classes
                )
                if result:
                    frame, det_classes = result
                    if frame is not None:
                        frame_ph.image(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            channels="RGB", use_container_width=True
                        )

                # FPS
                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    fps_display = fps_counter
                    fps_counter = 0
                    fps_timer   = time.time()

                # Update sidebar metrics
                if os.path.exists(self.csv_file):
                    try:
                        df = pd.read_csv(self.csv_file)
                        violations = df[df["Restricted Area Violation"] == "Yes"].shape[0]
                        tracked_ids = (
                            df["Object ID"].nunique() if "Object ID" in df.columns else "-"
                        )
                        metric_violations.metric("🚨 Total Violations", violations)
                        metric_tracked.metric("🔖 Unique Objects Tracked", tracked_ids)
                        metric_fps.metric("⚡ FPS", fps_display)
                        log_table.dataframe(df.tail(8), use_container_width=True)
                    except Exception:
                        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = MobileNetMonitoringApp()

    # Load detector once (downloads model if needed)
    with st.spinner("Loading MobileNet-SSD model (downloading on first run)..."):
        app.detector     = MobileNetDetector(confidence_threshold=0.4)
        app.class_colors = app._make_colors(SUPPORTED_CLASSES)

    app.run()
