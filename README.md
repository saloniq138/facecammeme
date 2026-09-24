# MemeCam 🎭📹

MemeCam is a Windows Python application that reads a camera/video stream, detects facial expressions locally, overlays configurable memes, and can publish the processed video to a virtual camera.

## Features

- HTTP/MJPEG, RTSP and local video-file input streams
- Optional direct webcam input with camera index
- Local face/expression detection using MediaPipe + OpenCV
- Happy, surprised, angry, sad and neutral expression heuristics
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
