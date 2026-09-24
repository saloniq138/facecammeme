# MemeCam 🎭📹

MemeCam is a Windows Python application that reads a camera/video stream, detects facial expressions locally, overlays configurable memes, and can publish the processed video to a virtual camera.

## Features

- HTTP/MJPEG, RTSP and local video-file input streams
- Optional direct webcam input with camera index
- Local face/expression detection using MediaPipe + OpenCV
- Richer facial geometry analysis inspired by the feature-scoring approach in `make_me_a_meme`
- Hand detection and raised-hand gesture signal
- Happy, surprised, angry, sad and neutral expression scoring
- Face-relative meme overlays that follow the detected face
- Configurable image overlays from `memes/`
- Per-expression cooldown
- GUI built with Tkinter
- Preview window
- Optional virtual camera output using `pyvirtualcam` + OBS Virtual Camera
- Windows executable build with GitHub Actions

## Run from source

Python 3.11 is recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Enter your stream URL, for example `http://127.0.0.1:8080/video` or `rtsp://user:password@host:554/stream`.

For a local webcam, leave the URL empty and select a camera index.

## Expression + meme matching

MemeCam now uses more facial landmarks/features instead of a single hard threshold. Eye openness, eyebrow position, mouth geometry, smile signal and hand gestures are combined into a confidence score. When an expression crosses the configured threshold, its meme is shown over the detected face for the configured duration.

Transparent PNGs work especially well because the alpha channel is preserved when the meme is placed over the face.

## Virtual camera

Install OBS Studio and enable **OBS Virtual Camera**. In MemeCam enable **Virtual camera**. Google Meet/Discord/Zoom can then select the virtual camera as their input.

The processed frame is never uploaded by MemeCam itself; detection runs locally.

## Project layout

- `main.py` — application and GUI
- `meme_engine.py` — expression detection and overlays
- `stream.py` — stream input
- `virtual_camera.py` — virtual camera output
- `config.json` — default settings
- `memes/` — user-provided images
