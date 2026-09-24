import cv2


class VideoStream:
    def __init__(self, source):
        self.source = source
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.source)
        if isinstance(self.source, str) and self.source.lower().startswith(("rtsp://", "http://", "https://")):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return bool(self.cap and self.cap.isOpened())

    def read(self):
        if not self.cap:
            return False, None
        return self.cap.read()

    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None
