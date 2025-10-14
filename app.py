# app.py
"""
Real-time webcam or single-image classification demo.

Usage:
  # webcam (ESC to quit)
  python app.py --camera 0 --weights models/mobilenet_v2_office.pth --classes docs/classes.txt

  # single image
  python app.py --image data/test/mug/img_001.jpg --weights models/mobilenet_v2_office.pth --classes docs/classes.txt
"""
import argparse, time
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from model import load_model
from utils import read_classes

def preprocess_bgr(frame_bgr, img_size=224):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return tf(rgb).unsqueeze(0)

def draw_overlay(frame, text: str):
    cv2.rectangle(frame, (0,0), (frame.shape[1], 40), (0,0,0), -1)
    cv2.putText(frame, text, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
    return frame

def infer_tensor(model, x, device="cpu"):
    with torch.no_grad():
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(probs.argmax())
        conf = float(probs[pred])
        return pred, conf

def run_webcam(args, model, class_names, device):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")
    t0, frames = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        x = preprocess_bgr(frame, img_size=args.img_size)
        pred, conf = infer_tensor(model, x, device)
        label = class_names[pred]
        frames += 1
        fps = frames / (time.time() - t0 + 1e-6)
        text = f"Class: {label} | Conf: {conf:.2f} | FPS: {fps:.1f}"
        frame = draw_overlay(frame, text)
        cv2.imshow("Office Classifier", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
    cap.release()
    cv2.destroyAllWindows()

def run_image(args, model, class_names, device):
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Image not found: {args.image}")
    x = preprocess_bgr(img, img_size=args.img_size)
    pred, conf = infer_tensor(model, x, device)
    label = class_names[pred]
    out = draw_overlay(img.copy(), f"Class: {label} | Conf: {conf:.2f}")
    cv2.imshow("Prediction", out)
    cv2.waitKey(0)

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--camera", type=int, help="webcam index")
    g.add_argument("--image", type=str, help="path to an image")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    class_names = read_classes(args.classes)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = load_model(args.weights, num_classes=len(class_names), device=device)

    if args.camera is not None:
        run_webcam(args, model, class_names, device)
    else:
        run_image(args, model, class_names, device)

if __name__ == "__main__":
    main()
