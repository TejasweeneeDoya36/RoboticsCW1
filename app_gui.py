"""
Tkinter GUI for running the trained classifier in two modes:
1) Live Camera – real‑time classification from your webcam with FPS overlay
2) Classify Image – select a single image file and show top‑k predictions

Features:
- Dark/Light theme toggle
- Optional CSV logging to results/predictions.csv (image + live)
- Simple camera index selector (0 = internal, 1 = USB)

Notes:
- Uses torchvision.mobilenet_v2 directly to keep GUI self‑contained
- Loads weights from models/mobilenet_v2_office.pth and labels from docs/classes.txt
- Runs on CPU by default for laptop demo stability
"""
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import os, csv, time
import cv2
import numpy as np
from PIL import Image, ImageTk

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v2

# CONFIG
WEIGHTS_PATH = Path("models/mobilenet_v2_office.pth") # model checkpoint saved by train.py
CLASSES_PATH = Path("docs/classes.txt")        # one label per line
RESULTS_DIR  = Path("results")
CSV_PATH     = RESULTS_DIR / "predictions.csv"

IMG_SIZE   = 224        # should match training size
CONF_THRESH = 0.70      # below this → show "unknown" to avoid over‑confident noise
DEVICE      = torch.device("cpu")  # keep CPU for laptop demo

# THEME PALETTES
LIGHT = dict(bg="#f5f6f7", fg="#111", panel="#ffffff", sunken="#101010", btn="#e8eaed", accent="#0aa370")
DARK  = dict(bg="#0f1115", fg="#e8eaed", panel="#151922", sunken="#0f1115", btn="#1f2430", accent="#3bd671")

def apply_theme(win: tk.Tk, palette):
    """Apply theme colors recursively to window and its children."""
    win.configure(bg=palette["bg"])
    for w in win.winfo_children():
        _apply_widget_theme(w, palette)

def _apply_widget_theme(w, p):
    # Minimal theming for common widgets
    if isinstance(w, (tk.Frame, tk.LabelFrame)):
        w.configure(bg=p["bg"])
    if isinstance(w, tk.Label):
        w.configure(bg=p["bg"], fg=p["fg"])
    if isinstance(w, tk.Button):
        w.configure(bg=p["btn"], fg=p["fg"], activebackground=p["accent"], activeforeground=p["fg"], bd=0, padx=8, pady=6)
    if isinstance(w, tk.Entry):
        w.configure(bg=p["panel"], fg=p["fg"], insertbackground=p["fg"], bd=1, relief=tk.FLAT)
    if isinstance(w, tk.Checkbutton):
        w.configure(bg=p["bg"], fg=p["fg"], activebackground=p["bg"], selectcolor=p["panel"])
    for c in w.winfo_children():
        _apply_widget_theme(c, p)

# HELPERS
def load_classes(path: Path):
    """Read class names from a text file."""
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def build_transform(img_size=224):
    """Preprocessing chain matching training normalization."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def load_model(weights_path: Path, num_classes: int):
    """Create a MobileNetV2, swap head for `num_classes`, and load saved weights."""
    model = mobilenet_v2(weights=None)
    in_feats = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_feats, num_classes)

    # map_location ensures weights load on CPU cleanly
    state = torch.load(weights_path, map_location=DEVICE)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        # If keys don't match perfectly, load what we can and print diagnostics
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Loaded with strict=False. Missing:", missing, "Unexpected:", unexpected)
    model.eval().to(DEVICE)
    return model

@torch.inference_mode()
def predict_image(model, pil_img, classes, tfm):
    """Run a forward pass on a PIL image and return label/confidence/top‑k.


    Returns:
    final_label: predicted label or "unknown" if below CONF_THRESH
    pred_conf: probability of top‑1
    top: list of (class_name, probability) for top‑k
    """
    x = tfm(pil_img).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    idx = int(probs.argmax())
    pred_conf = float(probs[idx])
    pred_label = classes[idx]

    final_label = pred_label if pred_conf >= CONF_THRESH else "unknown"
    return final_label, pred_conf

def cv2_put_text_multiline(img, lines, org=(10, 30)):
    """Draw multiple overlay lines on a BGR frame nicely spaced."""
    y = org[1]
    for line in lines:
        cv2.putText(img, line, (org[0], y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        y += 26

def ensure_csv(path: Path):
    """Create CSV and header if it doesn't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp","mode","source","label","confidence","fps","top3"])

def append_row(path: Path, row):
    """Append a single row to the CSV."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

# ====== GUI APP ======
class App(tk.Tk):
    """Main window: left panel has controls, right shows a preview.


    Lifecycle:
    - On init, load classes and model, build UI, apply theme, set status.
    - Button handlers call `run_live_camera` or `classify_image`.
    """
    def __init__(self):
        super().__init__()
        self.title("Office Perception — Demo Menu")
        self.geometry("1100x700")
        self.resizable(True, True)

        # UI State
        self.palette = DARK  # start dark
        self.dark_on = tk.BooleanVar(value=True)
        self.save_csv = tk.BooleanVar(value=False)
        self.cam_var  = tk.StringVar(value="0")

        # Load resources (labels + model). If anything fails, show a popup and exit.
        try:
            self.classes = load_classes(CLASSES_PATH)
        except Exception as e:
            messagebox.showerror("Classes Error", f"Failed to load classes:\n{e}")
            self.destroy(); return
        try:
            self.model = load_model(WEIGHTS_PATH, num_classes=len(self.classes))
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model weights:\n{e}")
            self.destroy(); return

        self.tfm = build_transform(IMG_SIZE)
        self._build_ui()
        apply_theme(self, self.palette)
        self._update_status("Ready.")

    # UI
    def _build_ui(self):
        # Left controls
        left = tk.Frame(self, padx=16, pady=16)
        left.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(left, text="Menu", font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(0,10))

        # Dark theme toggle
        tk.Checkbutton(left, text="Dark theme", variable=self.dark_on,
                       command=self.toggle_theme).pack(anchor="w", pady=(0,8))

        # CSV toggle
        tk.Checkbutton(left, text="Save predictions to CSV", variable=self.save_csv,
                       command=self._maybe_prepare_csv).pack(anchor="w", pady=(0,12))

        # Camera index
        tk.Label(left, text="Camera index (0=internal, 1=USB):").pack(anchor="w")
        tk.Entry(left, textvariable=self.cam_var, width=5).pack(anchor="w", pady=(0,12))

        # Actions
        tk.Button(left, text="▶ Live Camera", width=24, command=self.run_live_camera)\
            .pack(anchor="w", pady=6)
        tk.Button(left, text="📁 Classify Image (Upload)", width=24, command=self.classify_image)\
            .pack(anchor="w", pady=6)
        tk.Button(left, text="ℹ Instructions", width=24, command=self.show_instructions)\
            .pack(anchor="w", pady=6)
        tk.Button(left, text="📂 Open Results Folder", width=24, command=self.open_results)\
            .pack(anchor="w", pady=(14,6))
        tk.Button(left, text="✖ Exit", width=24, command=self.destroy)\
            .pack(anchor="w", pady=(10,0))

        # Right preview
        right = tk.Frame(self, padx=16, pady=16)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(right, text="Preview", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        self.preview = tk.Label(right, relief=tk.SUNKEN, width=88, height=24, bg=self.palette["sunken"])
        self.preview.pack(fill=tk.BOTH, expand=True, pady=(10,0))

        self.status = tk.Label(self, text="", anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def toggle_theme(self):
        self.palette = DARK if self.dark_on.get() else LIGHT
        apply_theme(self, self.palette)

    def _maybe_prepare_csv(self):
        if self.save_csv.get():
            ensure_csv(CSV_PATH)
            self._update_status(f"CSV logging enabled → {CSV_PATH}")
        else:
            self._update_status("CSV logging disabled")

    def _update_preview_pil(self, pil_img, caption=""):
        """Resize a PIL image for display and draw a caption bar if provided."""
        W, H = 720, 420
        img = pil_img.copy()
        img.thumbnail((W, H))
        if caption:
            cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv2.rectangle(cv, (0,0), (cv.shape[1], 36), (0,0,0), -1)
            cv2.putText(cv, caption, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
            img = Image.fromarray(cv2.cvtColor(cv, cv2.COLOR_BGR2RGB))
        tkimg = ImageTk.PhotoImage(img)
        self.preview.configure(image=tkimg)
        self.preview.image = tkimg

    def _update_status(self, text):
        self.status.configure(text=text)

    # Actions
    def show_instructions(self):
        text = (
            "How it works:\n"
            "1) Loads a trained MobileNetV2 model and class list.\n"
            "2) Live Camera → real-time classification.\n"
            "   Upload Image → classify a single image.\n"
            "3) Shows predicted class, confidence, and FPS.\n\n"
            "Tips:\n"
            "- Use camera index 0 for built-in, 1 for USB webcam.\n"
            "- Good lighting improves accuracy.\n"
            "- Press ESC in the camera window to stop."
        )
        messagebox.showinfo("Instructions", text)

    def classify_image(self):
        """Select an image, run inference, update preview + optional CSV logging."""
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not open image:\n{e}")
            return

        label, conf = predict_image(self.model, pil, self.classes, self.tfm)
        caption = f"Predicted: {label} ({conf:.2f})"
        self._update_preview_pil(pil, caption=caption)
        self._update_status(caption)

        if self.save_csv.get():
            ensure_csv(CSV_PATH)
            append_row(CSV_PATH, [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                "image",
                os.path.abspath(path),
                label,
                f"{conf:.4f}",
                ""
            ])

    def run_live_camera(self):
        """Open the webcam, stream frames, overlay predictions + FPS, ESC to exit."""
        try:
            idx = int(self.cam_var.get().strip())
        except Exception:
            idx = 0
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", f"Cannot open camera index {idx}")
            return

        last = time.time(); fps = 0.0
        last_log = 0.0
        last_label = None

        while True:
            ok, frame = cap.read()
            if not ok: break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            label, conf = predict_image(self.model, pil, self.classes, self.tfm)

            now = time.time(); dt = now - last
            fps = (1.0 / dt) if dt > 0 else fps; last = now

            lines = [
                f"Class: {label} | Conf: {conf:.2f} | FPS: {fps:.1f}",
                "ESC to exit"
            ]

            # Draw a top bar and write text
            cv2.rectangle(frame, (0,0), (frame.shape[1], 60), (0,0,0), -1)
            cv2_put_text_multiline(frame, lines, org=(10, 24))

            cv2.imshow("Live Camera — Office Perception", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            # CSV logging throttle (2 Hz) or on label change
            if self.save_csv.get():
                if (now - last_log) >= 0.5 or label != last_label:
                    ensure_csv(CSV_PATH)
                    append_row(CSV_PATH, [
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        "live",
                        f"camera:{idx}",
                        label,
                        f"{conf:.4f}",
                        f"{fps:.2f}"
                    ])
                    last_log = now
                    last_label = label

        cap.release()
        cv2.destroyAllWindows()
        self._update_status("Live camera stopped.")

    def open_results(self):
        """Open the results folder in the OS file explorer."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(RESULTS_DIR)  # Windows
            elif sys.platform == "darwin":
                os.system(f'open "{RESULTS_DIR}"')
            else:
                os.system(f'xdg-open "{RESULTS_DIR}"')
        except Exception:
            messagebox.showinfo("Results", f"Folder: {RESULTS_DIR.resolve()}")

if __name__ == "__main__":
    app = App()
    app.mainloop()