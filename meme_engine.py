import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


class MemeEngine:
    """
    Real-time facial-expression + hand-gesture engine.

    Inspired by the feature-based approach from make_me_a_meme:
    - richer facial geometry
    - hand detection
    - weighted expression scoring
    - temporal cooldown
    - face-relative overlay placement

    This is an entertainment-oriented heuristic classifier, not a clinical
    emotion-recognition system.
    """

    def __init__(self, threshold=0.55, cooldown=1.2, duration=1.8):
        self.threshold = float(threshold)
        self.cooldown = float(cooldown)
        self.duration = float(duration)
        self.overlays = {}

        self.last_expression = None
        self.last_trigger = 0.0
        self.active_until = 0.0
        self.last_confidence = 0.0
        self.last_gesture = "none"

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )

        self.feature_names = [
            "eye_openness",
            "eye_symmetry",
            "mouth_openness",
            "mouth_width",
            "smile",
            "brow_height",
            "brow_symmetry",
            "hand_raised",
            "num_hands",
        ]

    @staticmethod
    def _dist(a, b):
        return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))

    @staticmethod
    def _clamp(v, lo=0.0, hi=1.0):
        return max(lo, min(hi, float(v)))

    def set_overlays(self, mapping):
        self.overlays = dict(mapping)

    def _features(self, lm, hand_result):
        p = lambda i: np.array([lm[i].x, lm[i].y], dtype=np.float32)

        face_height = max(self._dist(p(10), p(152)), 1e-5)
        face_width = max(self._dist(p(234), p(454)), 1e-5)

        left_eye = self._dist(p(159), p(145)) / face_height
        right_eye = self._dist(p(386), p(374)) / face_height
        eye_open = (left_eye + right_eye) / 2.0
        eye_symmetry = abs(left_eye - right_eye)

        mouth_open = self._dist(p(13), p(14)) / face_height
        mouth_width = self._dist(p(61), p(291)) / face_width
        mouth_ratio = self._dist(p(78), p(308)) / max(self._dist(p(61), p(291)), 1e-5)

        left_brow = self._dist(p(105), p(159)) / face_height
        right_brow = self._dist(p(334), p(386)) / face_height
        brow_height = (left_brow + right_brow) / 2.0
        brow_symmetry = abs(left_brow - right_brow)

        # Continuous smile/surprise signals. Values are deliberately normalized
        # so they can be combined rather than relying on one hard threshold.
        smile = self._clamp((mouth_width - 0.28) / 0.18) * self._clamp((0.075 - mouth_open) / 0.05 + 0.35)
        smile = self._clamp(smile)

        num_hands = len(hand_result.multi_hand_landmarks or [])
        hand_raised = 0.0
        if num_hands:
            face_top = min(x.y for x in lm)
            face_center_y = sum(x.y for x in lm) / len(lm)
            for hand in hand_result.multi_hand_landmarks:
                wrist_y = hand.landmark[0].y
                middle_y = hand.landmark[12].y
                if middle_y < face_center_y + 0.18 or wrist_y < face_top + 0.35:
                    hand_raised = 1.0
                    break

        return {
            "eye_openness": eye_open,
            "eye_symmetry": eye_symmetry,
            "mouth_openness": mouth_open,
            "mouth_width": mouth_width,
            "mouth_ratio": mouth_ratio,
            "smile": smile,
            "brow_height": brow_height,
            "brow_symmetry": brow_symmetry,
            "hand_raised": hand_raised,
            "num_hands": float(num_hands),
        }

    def _scores(self, f):
        # Weighted feature scoring inspired by make_me_a_meme's similarity model.
        surprise = np.mean([
            self._clamp((f["eye_openness"] - 0.028) / 0.022),
            self._clamp((f["brow_height"] - 0.14) / 0.10),
            self._clamp((f["mouth_openness"] - 0.08) / 0.12),
        ])

        happy = np.mean([
            self._clamp((f["mouth_width"] - 0.30) / 0.16),
            self._clamp((f["mouth_ratio"] - 0.45) / 0.28),
            f["smile"],
        ])

        angry = np.mean([
            self._clamp((0.14 - f["brow_height"]) / 0.07),
            self._clamp((0.075 - f["mouth_openness"]) / 0.055),
            self._clamp((0.025 - f["eye_openness"]) / 0.018),
        ])

        sad = np.mean([
            self._clamp((0.028 - f["eye_openness"]) / 0.018),
            self._clamp((0.035 - f["mouth_openness"]) / 0.025),
            self._clamp((0.15 - f["mouth_width"]) / 0.12),
        ])

        # Raised-hand bonus makes the result more expressive without forcing a
        # hand gesture to be present for ordinary facial expressions.
        cheers = self._clamp(0.65 * happy + 0.35 * f["hand_raised"])

        scores = {
            "surprised": float(surprise),
            "happy": float(happy),
            "angry": float(angry),
            "sad": float(sad),
        }

        if f["hand_raised"] > 0:
            scores["happy"] = self._clamp(0.72 * scores["happy"] + 0.28 * cheers)

        return scores

    def _classify(self, lm, hand_result):
        f = self._features(lm, hand_result)
        scores = self._scores(f)
        expression, confidence = max(scores.items(), key=lambda item: item[1])

        # Avoid triggering on weak/noisy classifications.
        if confidence < self.threshold:
            return "neutral", confidence, f, scores
        return expression, confidence, f, scores

    def _load_overlay(self, expression):
        path = self.overlays.get(expression, "")
        if not path:
            return None

        if not os.path.isabs(path):
            # First try relative to the application/project directory.
            path = str(Path(__file__).resolve().parent / path)

        if not os.path.exists(path):
            return None

        return cv2.imread(path, cv2.IMREAD_UNCHANGED)

    def _face_box(self, lm, frame_shape):
        h, w = frame_shape[:2]
        xs = np.array([p.x for p in lm], dtype=np.float32)
        ys = np.array([p.y for p in lm], dtype=np.float32)

        x1 = int(np.clip(xs.min() * w, 0, w - 1))
        x2 = int(np.clip(xs.max() * w, 0, w - 1))
        y1 = int(np.clip(ys.min() * h, 0, h - 1))
        y2 = int(np.clip(ys.max() * h, 0, h - 1))

        return x1, y1, x2, y2

    def _apply_overlay(self, frame, overlay, lm):
        if overlay is None or lm is None:
            return frame

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self._face_box(lm, frame.shape)
        face_w = max(1, x2 - x1)
        face_h = max(1, y2 - y1)

        # Make the meme roughly face-sized and place it over the face.
        target_w = max(100, int(face_w * 1.25))
        oh, ow = overlay.shape[:2]
        scale = target_w / max(1, ow)
        target_h = max(1, int(oh * scale))

        # Center horizontally on the face; put the image slightly above the
        # face center so transparent PNGs can look like a face mask/sticker.
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        x = cx - target_w // 2
        y = cy - target_h // 2

        # Clip destination and corresponding source region to the frame.
        sx1, sy1 = 0, 0
        sx2, sy2 = target_w, target_h
        if x < 0:
            sx1 = -x
            x = 0
        if y < 0:
            sy1 = -y
            y = 0
        if x + (sx2 - sx1) > w:
            sx2 = sx1 + (w - x)
        if y + (sy2 - sy1) > h:
            sy2 = sy1 + (h - y)

        if sx2 <= sx1 or sy2 <= sy1:
            return frame

        resized = cv2.resize(
            overlay,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )[sy1:sy2, sx1:sx2]

        roi = frame[y:y + resized.shape[0], x:x + resized.shape[1]]
        if resized.ndim == 3 and resized.shape[2] == 4:
            alpha = resized[:, :, 3:4].astype(np.float32) / 255.0
            fg = resized[:, :, :3].astype(np.float32)
            bg = roi.astype(np.float32)
            frame[y:y + resized.shape[0], x:x + resized.shape[1]] = (
                fg * alpha + bg * (1.0 - alpha)
            ).astype(np.uint8)
        else:
            frame[y:y + resized.shape[0], x:x + resized.shape[1]] = resized[:, :, :3]

        return frame

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_result = self.face_mesh.process(rgb)
        hand_result = self.hands.process(rgb)
        now = time.monotonic()

        expression = "neutral"
        confidence = 0.0
        face_landmarks = None
        features = None
        scores = {}

        if face_result.multi_face_landmarks:
            face_landmarks = face_result.multi_face_landmarks[0].landmark
            expression, confidence, features, scores = self._classify(
                face_landmarks,
                hand_result,
            )
            self.last_confidence = confidence

            if expression != "neutral" and confidence >= self.threshold:
                if (
                    expression != self.last_expression
                    and now - self.last_trigger >= self.cooldown
                ):
                    self.last_expression = expression
                    self.last_trigger = now
                    self.active_until = now + self.duration

            self.last_gesture = (
                "hand raised"
                if features and features["hand_raised"] > 0.5
                else "none"
            )

        if now < self.active_until and self.last_expression:
            overlay = self._load_overlay(self.last_expression)
            frame = self._apply_overlay(frame, overlay, face_landmarks)

        label = expression.upper() if expression != "neutral" else "NEUTRAL"
        cv2.putText(
            frame,
            f"{label}  {confidence:.0%}",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if self.last_gesture != "none":
            cv2.putText(
                frame,
                self.last_gesture,
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return frame
