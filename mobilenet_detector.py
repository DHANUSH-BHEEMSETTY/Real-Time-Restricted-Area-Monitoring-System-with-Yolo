"""
MobileNet-SSD + Centroid Tracking Engine
=========================================
This module provides:
  - CentroidTracker  : Classical computer-vision algorithm that assigns persistent
                       object IDs across frames using Euclidean-distance matching.
  - MobileNetDetector: Wrapper around OpenCV's cv2.dnn module loaded with the
                       MobileNet-SSD Caffe model (PASCAL-VOC 20 classes).

The model weights are downloaded automatically on first run (~23 MB total).
No extra pip packages are required — cv2 is already in the project's requirements.
"""

import cv2
import numpy as np
import os
import urllib.request
from collections import OrderedDict
from scipy.spatial import distance as dist

# ---------------------------------------------------------------------------
# PASCAL-VOC 20 classes supported by MobileNet-SSD
# (index 0 = background)
# ---------------------------------------------------------------------------
MOBILENET_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor",
]

# ---------------------------------------------------------------------------
# Model file paths & download URLs
# ---------------------------------------------------------------------------
MODEL_DIR = "model"
PROTOTXT_PATH  = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.caffemodel")

PROTOTXT_URL    = (
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/"
    "master/MobileNetSSD_deploy.prototxt"
)
CAFFEMODEL_URL  = (
    "https://github.com/djmv/MobilNet_SSD_opencv/raw/master/"
    "MobileNetSSD_deploy.caffemodel"
)


# ---------------------------------------------------------------------------
# Centroid Tracker
# ---------------------------------------------------------------------------
class CentroidTracker:
    """
    Centroid-based multi-object tracker.

    Algorithm:
    1. For each detected bounding box compute centroid (cx, cy).
    2. On subsequent frames compute a pairwise Euclidean-distance matrix
       between existing centroids and new centroids.
    3. Greedily assign each existing object to its nearest new centroid
       (smallest distance wins; ties broken by row order).
    4. Objects with no match for `max_disappeared` consecutive frames are
       deregistered.
    5. Unmatched new centroids are registered as fresh objects.
    """

    def __init__(self, max_disappeared: int = 50):
        self.next_object_id = 0
        # objectID -> centroid np.array([cx, cy])
        self.objects: "OrderedDict[int, np.ndarray]" = OrderedDict()
        # objectID -> consecutive frames without a match
        self.disappeared: "OrderedDict[int, int]" = OrderedDict()
        self.max_disappeared = max_disappeared

    # --- internal helpers ---------------------------------------------------

    def _register(self, centroid: np.ndarray) -> None:
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def _deregister(self, object_id: int) -> None:
        del self.objects[object_id]
        del self.disappeared[object_id]

    # --- public API ---------------------------------------------------------

    def update(self, input_rects: list) -> "OrderedDict[int, np.ndarray]":
        """
        Update tracker with a list of bounding boxes (x1, y1, x2, y2).

        Returns:
            OrderedDict mapping objectID -> centroid (np.array [cx, cy])
        """
        # No detections this frame
        if len(input_rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return self.objects

        # Compute centroids for all input rects
        input_centroids = np.zeros((len(input_rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(input_rects):
            input_centroids[i] = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))

        # No existing tracked objects — register all
        if len(self.objects) == 0:
            for c in input_centroids:
                self._register(c)
            return self.objects

        # Pairwise Euclidean distances: shape (num_existing, num_new)
        object_ids       = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        D = dist.cdist(np.array(object_centroids), input_centroids)

        # Sort rows by their minimum value, then pick the column
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows: set = set()
        used_cols: set = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            oid = object_ids[row]
            self.objects[oid]     = input_centroids[col]
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Existing objects not matched → increment disappeared counter
        for row in set(range(D.shape[0])) - used_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self._deregister(oid)

        # New detections not matched to any existing object → register
        for col in set(range(D.shape[1])) - used_cols:
            self._register(input_centroids[col])

        return self.objects


# ---------------------------------------------------------------------------
# MobileNet-SSD Detector
# ---------------------------------------------------------------------------
class MobileNetDetector:
    """
    Object detector backed by MobileNet-SSD (Caffe) via cv2.dnn.

    Attributes
    ----------
    names : dict
        Map of class_id -> class_name (mirrors the YOLO model interface).
    """

    def __init__(self, confidence_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold
        self.net    = None
        self.tracker = CentroidTracker(max_disappeared=40)
        self.class_names = MOBILENET_CLASSES

        self._ensure_model_files()
        self._load_model()

    # --- model download / load ----------------------------------------------

    def _download_file(self, url: str, dest: str, label: str) -> None:
        print(f"[INFO] Downloading {label} ...")

        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                print(f"\r  {pct}% ({downloaded // 1024} KB / {total_size // 1024} KB)", end="")

        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print(f"\n[INFO] {label} saved to {dest}")

    def _ensure_model_files(self) -> None:
        os.makedirs(MODEL_DIR, exist_ok=True)

        if not os.path.exists(PROTOTXT_PATH):
            self._download_file(PROTOTXT_URL, PROTOTXT_PATH, "MobileNetSSD prototxt")
        else:
            print(f"[INFO] Prototxt found at {PROTOTXT_PATH}")

        if not os.path.exists(CAFFEMODEL_PATH):
            self._download_file(CAFFEMODEL_URL, CAFFEMODEL_PATH, "MobileNetSSD caffemodel (~23 MB)")
        else:
            print(f"[INFO] Caffemodel found at {CAFFEMODEL_PATH}")

    def _load_model(self) -> None:
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
        print("[INFO] MobileNet-SSD model loaded successfully.")

    # --- public properties --------------------------------------------------

    @property
    def names(self) -> dict:
        """Mirror the YOLO `model.names` interface."""
        return {i: name for i, name in enumerate(self.class_names)}

    # --- detection & tracking -----------------------------------------------

    def detect(self, frame: np.ndarray, selected_classes: list = None) -> list:
        """
        Run MobileNet-SSD inference on a single frame.

        Parameters
        ----------
        frame            : BGR frame from OpenCV
        selected_classes : if provided, only return detections whose class is in this list

        Returns
        -------
        list of dicts with keys: class_id, class_name, confidence, box (x1,y1,x2,y2)
        """
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=0.007843,
            size=(300, 300),
            mean=127.5,
        )
        self.net.setInput(blob)
        raw = self.net.forward()  # shape: (1, 1, N, 7)

        results = []
        for i in np.arange(0, raw.shape[2]):
            confidence = float(raw[0, 0, i, 2])
            if confidence < self.confidence_threshold:
                continue

            class_id   = int(raw[0, 0, i, 1])
            class_name = (
                self.class_names[class_id]
                if class_id < len(self.class_names)
                else "unknown"
            )

            if selected_classes and class_name not in selected_classes:
                continue

            box = raw[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            # Clip to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            results.append({
                "class_id":   class_id,
                "class_name": class_name,
                "confidence": confidence,
                "box":        (x1, y1, x2, y2),
            })

        return results

    def update_tracker(self, detections: list) -> "OrderedDict[int, np.ndarray]":
        """Pass detection bounding boxes to the Centroid Tracker and get object-ID map."""
        rects = [d["box"] for d in detections]
        return self.tracker.update(rects)

    def get_fps(self, frame: np.ndarray) -> float:
        """Measure inference speed on a single frame (ms → fps)."""
        import time
        start = time.perf_counter()
        self.detect(frame)
        elapsed = time.perf_counter() - start
        return round(1.0 / elapsed, 1) if elapsed > 0 else 0.0


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== MobileNet-SSD Self-Test ===")
    detector = MobileNetDetector(confidence_threshold=0.3)
    print(f"Supported classes ({len(detector.class_names) - 1}): "
          f"{detector.class_names[1:]}")

    # Create a blank test frame
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    dets  = detector.detect(dummy)
    objs  = detector.update_tracker(dets)
    print(f"Detections on blank frame: {len(dets)}")
    print(f"Tracked objects: {dict(objs)}")
    print("Self-test complete.")
