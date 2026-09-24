import json
import os
import shutil
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

from meme_engine import MemeEngine
from stream import VideoStream
from virtual_camera import VirtualCameraOutput


def app_dir():
    # For PyInstaller --onefile use the executable directory; for source use this file's directory.
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = app_dir()
USER_DIR = os.path.join(os.getenv("APPDATA", BASE_DIR), "FaceCamMeme")
REACTIONS_DIR = os.path.join(USER_DIR, "reactions")
CONFIG_PATH = os.path.join(USER_DIR, "config.json")
BUNDLE_DIR = getattr(sys, "_MEIPASS", BASE_DIR)
os.makedirs(REACTIONS_DIR, exist_ok=True)


def load_config():
    if not os.path.exists(CONFIG_PATH):
        bundled = os.path.join(BUNDLE_DIR, "config.json")
        if os.path.exists(bundled):
            import shutil
            try:
                shutil.copy2(bundled, CONFIG_PATH)
            except OSError:
                pass
    if not os.path.exists(CONFIG_PATH):
        os.makedirs(USER_DIR, exist_ok=True)
        default = {
            "stream_url": "",
            "camera_index": 0,
            "width": 1280,
            "height": 720,
            "fps": 30,
            "expression_threshold": 0.55,
            "cooldown_seconds": 1.2,
            "overlay_duration_seconds": 1.8,
            "virtual_camera": False,
            "flip": False,
            "expressions": {
                "happy": "memes/happy.png",
                "surprised": "memes/surprised.png",
                "angry": "memes/angry.png",
                "sad": "memes/sad.png"
            }
        }
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2, ensure_ascii=False)
        return default
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class MemeCamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MemeCam")
        self.root.geometry("760x600")
        self.root.resizable(False, False)
        self.config = load_config()
        self.running = False
        self.stream = None
        self.engine = MemeEngine(
            threshold=float(self.config.get("expression_threshold", 0.55)),
            cooldown=float(self.config.get("cooldown_seconds", 1.2)),
            duration=float(self.config.get("overlay_duration_seconds", 1.8)),
        )
        self.output = VirtualCameraOutput(
            width=int(self.config.get("width", 1280)),
            height=int(self.config.get("height", 720)),
            fps=int(self.config.get("fps", 30)),
        )
        self.build_ui()

    def build_ui(self):
        pad = {"padx": 10, "pady": 6}
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="MemeCam", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(frm, text="Kamera/stream → wykrywanie min → MEME → wirtualna kamera").pack(anchor="w", pady=(0, 12))

        ttk.Label(frm, text="Adres streamu (HTTP/MJPEG/RTSP/plik):").pack(anchor="w", **pad)
        self.url = ttk.Entry(frm, width=78)
        self.url.pack(fill="x", **pad)
        self.url.insert(0, self.config.get("stream_url", ""))

        row = ttk.Frame(frm)
        row.pack(fill="x", **pad)
        ttk.Label(row, text="Kamera lokalna, jeśli URL pusty:").pack(side="left")
        self.camera_index = tk.StringVar(value=str(self.config.get("camera_index", 0)))
        ttk.Spinbox(row, from_=0, to=9, textvariable=self.camera_index, width=6).pack(side="left", padx=8)

        ttk.Label(frm, text="Memes (opcjonalnie):").pack(anchor="w", **pad)
        self.labels = {}
        for expression, filename in self.config["expressions"].items():
            r = ttk.Frame(frm)
            r.pack(fill="x", padx=10, pady=2)
            ttk.Label(r, text=f"{expression:10}", width=12).pack(side="left")
            var = tk.StringVar(value=filename)
            self.labels[expression] = var
            ttk.Entry(r, textvariable=var, width=48).pack(side="left")
            ttk.Button(r, text="Wybierz", command=lambda e=expression: self.choose_meme(e)).pack(side="left", padx=5)

        opts = ttk.Frame(frm)
        opts.pack(fill="x", pady=12)
        self.virtual_var = tk.BooleanVar(value=bool(self.config.get("virtual_camera", False)))
        self.flip_var = tk.BooleanVar(value=bool(self.config.get("flip", False)))
        ttk.Checkbutton(opts, text="Wirtualna kamera (OBS)", variable=self.virtual_var).pack(side="left")
        ttk.Checkbutton(opts, text="Odbij obraz", variable=self.flip_var).pack(side="left", padx=20)

        buttons = ttk.Frame(frm)
        buttons.pack(fill="x", pady=8)
        ttk.Button(buttons, text="▶ START", command=self.start).pack(side="left")
        ttk.Button(buttons, text="■ STOP", command=self.stop).pack(side="left", padx=8)
        ttk.Button(buttons, text="Zapisz ustawienia", command=self.save_config).pack(side="left")

        self.status = tk.StringVar(value="Gotowy")
        ttk.Label(frm, textvariable=self.status).pack(anchor="w", pady=10)
        ttk.Label(frm, text="Q w oknie podglądu = stop", foreground="gray").pack(anchor="w")

    def choose_meme(self, expression):
        path = filedialog.askopenfilename(
            title=f"Wybierz własną reakcję dla {expression}",
            filetypes=[("Obrazy", "*.png *.jpg *.jpeg *.webp"), ("Wszystkie", "*.*")]
        )
        if not path:
            return
        try:
            os.makedirs(REACTIONS_DIR, exist_ok=True)
            filename = os.path.basename(path)
            stem, ext = os.path.splitext(filename)
            destination = os.path.join(REACTIONS_DIR, filename)
            counter = 1
            while os.path.exists(destination) and os.path.abspath(destination) != os.path.abspath(path):
                destination = os.path.join(REACTIONS_DIR, f"{stem}_{counter}{ext}")
                counter += 1
            if os.path.abspath(path) != os.path.abspath(destination):
                shutil.copy2(path, destination)
            self.labels[expression].set(destination)
            self.save_config()
            self.status.set(f"Dodano własny meme dla: {expression}")
        except OSError as e:
            messagebox.showerror("MemeCam", f"Nie udało się dodać obrazka:\n{e}")

    def save_config(self):
        self.config["stream_url"] = self.url.get().strip()
        try:
            self.config["camera_index"] = int(self.camera_index.get())
        except ValueError:
            self.config["camera_index"] = 0
        self.config["virtual_camera"] = self.virtual_var.get()
        self.config["flip"] = self.flip_var.get()
        self.config["expressions"] = {k: v.get().strip() for k, v in self.labels.items()}
        os.makedirs(BASE_DIR, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        self.status.set("Ustawienia zapisane.")

    def start(self):
        if self.running:
            return
        self.save_config()
        source = self.url.get().strip()
        if not source:
            try:
                source = int(self.camera_index.get())
            except ValueError:
                source = 0
        self.stream = VideoStream(source)
        if not self.stream.open():
            messagebox.showerror("MemeCam", "Nie udało się otworzyć streamu.")
            self.stream = None
            return
        self.engine = MemeEngine(
            threshold=float(self.config.get("expression_threshold", 0.55)),
            cooldown=float(self.config.get("cooldown_seconds", 1.2)),
            duration=float(self.config.get("overlay_duration_seconds", 1.8)),
        )
        self.engine.set_overlays(self.config["expressions"])
        self.output = VirtualCameraOutput(
            width=int(self.config.get("width", 1280)),
            height=int(self.config.get("height", 720)),
            fps=int(self.config.get("fps", 30)),
        )
        if self.virtual_var.get() and not self.output.open():
            self.status.set("Wirtualna kamera niedostępna — działa tylko podgląd.")
        self.running = True
        self.status.set("🟢 Działa")
        self.loop()

    def loop(self):
        if not self.running or self.stream is None:
            return
        ok, frame = self.stream.read()
        if not ok:
            self.status.set("Stream przerwany.")
            self.stop()
            return
        if self.flip_var.get():
            frame = cv2.flip(frame, 1)
        frame = self.engine.process(frame)
        if self.virtual_var.get() and self.output.is_open:
            self.output.send(frame)
        cv2.imshow("MemeCam Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.stop()
            return
        self.root.after(1, self.loop)

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.close()
            self.stream = None
        self.output.close()
        cv2.destroyAllWindows()
        self.status.set("Zatrzymano.")

    def on_close(self):
        self.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MemeCamApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
