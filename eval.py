# eval.py
"""
Evaluate on the held-out test set and save:
  - metrics.json (accuracy, macro-F1, per-class precision/recall)
  - confusion_matrix.png
Usage:
  python eval.py --data data/test --weights models/mobilenet_v2_office.pth --classes docs/classes.txt --out results/
"""
from pathlib import Path
import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from utils import read_classes, save_json, plot_confmat
from model import load_model

def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    class_names = read_classes(args.classes)
    num_classes = len(class_names)

    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    ds = datasets.ImageFolder(args.data, transform=tf)
    ld = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = load_model(args.weights, num_classes=num_classes, device=device)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in ld:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(y.numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    prec, rec, f1c, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(num_classes)), zero_division=0)

    # Save
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    plot_confmat(y_true, y_pred, class_names, str(outdir/"confusion_matrix.png"))
    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": [
            {"class": class_names[i], "precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1c[i])}
            for i in range(num_classes)
        ]
    }
    save_json(metrics, str(outdir/"metrics.json"))
    print(f"Accuracy: {acc:.3f} | Macro-F1: {macro_f1:.3f}")
    print(f"Saved metrics.json and confusion_matrix.png to {outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="test folder (ImageFolder layout)")
    ap.add_argument("--weights", required=True, help="model weights (.pth)")
    ap.add_argument("--classes", required=True, help="path to classes.txt (one label per line)")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="results")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)
