import numpy as np

try:
    import pyvirtualcam
except Exception:
    pyvirtualcam = None


class VirtualCameraOutput:
    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.cam = None
        self.is_open = False

    def open(self):
        if pyvirtualcam is None:
            return False
        try:
            self.cam = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
                backend="obs",
            )
            self.is_open = True
            return True
        except Exception:
            self.cam = None
            self.is_open = False
            return False

    def send(self, frame):
        if not self.is_open or self.cam is None:
            return
        frame = np.ascontiguousarray(frame)
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            import cv2
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.cam.send(frame)
        self.cam.sleep_until_next_frame()

    def close(self):
        if self.cam:
            self.cam.close()
        self.cam = None
        self.is_open = False
