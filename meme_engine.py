import os
import time
import cv2
import mediapipe as mp
import numpy as np


class MemeEngine:
    def __init__(self, threshold=0.55, cooldown=1.2, duration=1.8):
        self.threshold = threshold
        self.cooldown = cooldown
        self.duration = duration
        self.overlays = {}
        self.last_expression = None
        self.last_change = 0.0
        self.last_trigger = 0.0
        self.active_until = 0.0
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def set_overlays(self, mapping):
        self.overlays = dict(mapping)

    @staticmethod
    def _dist(a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    def _features(self, lm):
        # MediaPipe normalized landmarks.
        p = lambda i: (lm[i].x, lm[i].y)
        mouth_open = self._dist(p(13), p(14))
        mouth_width = max(self._dist(p(61), p(291)), 1e-5)
        left_eye = self._dist(p(159), p(145))
        right_eye = self._dist(p(386), p(374))
        face_height = max(self._dist(p(10), p(152)), 1e-5)
        brow_left = self._dist(p(105), p(159)) / face_height
        brow_right = self._dist(p(334), p(386)) / face_height
        eye_open = (left_eye + right_eye) / (2 * face_height)
        return mouth_open / face_height, mouth_width / face_height, eye_open, (brow_left + brow_right) / 2

    def _classify(self, lm):
        mouth_open, mouth_width, eye_open, brow = self._features(lm)
        # Lightweight heuristics: these are intentionally adjustable, not a clinical emotion classifier.
        if mouth_open > 0.16 and brow > 0.18 and eye_open > 0.035:
            return "surprised", min(1.0, mouth_open / 0.24)
        if mouth_width > 0.36 and mouth_open > 0.035:
            return "happy", min(1.0, mouth_width / 0.48)
        if brow < 0.105 and mouth_open < 0.07:
            return "angry", min(1.0, 0.105 / max(brow, 0.001))
        if mouth_open < 0.025 and eye_open < 0.026:
            return "sad", min(1.0, 0.026 / max(eye_open, 0.001))
        return "neutral", 0.5

    def _load_overlay(self, expression):
        path = self.overlays.get(expression, "")
        if not path:
            return None
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        if not os.path.exists(path):
            return None
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return img

    def _apply_overlay(self, frame, overlay):
        if overlay is None:
            return frame
        h, w = frame.shape[:2]
        oh, ow = overlay.shape[:2]
        target_w = max(100, int(w * 0.30))
        scale = target_w / max(1, ow)
        target_h = max(1, int(oh * scale))
        overlay = cv2.resize(overlay, (target_w, target_h), interpolation=cv2.INTER_AREA)
        x = (w - target_w) // 2
        y = max(10, int(h * 0.06))
        if y + target_h > h:
            target_h = h - y
            overlay = overlay[:target_h]
        if overlay.shape[2] == 4:
            alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
            rgb = overlay[:, :, :3].astype(np.float32)
            roi = frame[y:y+target_h, x:x+target_w].astype(np.float32)
            frame[y:y+target_h, x:x+target_w] = (rgb * alpha + roi * (1-alpha)).astype(np.uint8)
        else:
            frame[y:y+target_h, x:x+target_w] = overlay[:, :, :3]
        return frame

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        now = time.monotonic()
        expression = "neutral"
        confidence = 0.0
        if result.multi_face_landmarks:
            expression, confidence = self._classify(result.multi_face_landmarks[0].landmark)
            if confidence >= self.threshold and expression != "neutral":
                if expression != self.last_expression and now - self.last_trigger >= self.cooldown:
                    self.last_expression = expression
                    self.last_change = now
                    self.last_trigger = now
                    self.active_until = now + self.duration
        if now < self.active_until and self.last_expression:
            overlay = self._load_overlay(self.last_expression)
            frame = self._apply_overlay(frame, overlay)
            cv2.putText(frame, self.last_expression.upper(), (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
