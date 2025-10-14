# train.py
"""
Transfer learning on MobileNetV2 for 8–10 office classes.

Dataset structure (ImageFolder):
data/
  train/
    mug/..., bottle/..., book/... (etc.)
  val/
    mug/..., ...
  test/
    mug/..., ...

Usage:
  python train.py --data data --classes docs/classes.txt --out models/mobilenet_v2_office.pth
"""
from pathlib import Path
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, accuracy_score
from utils import set_seed, read_classes, save_json
from model import get_mobilenet_v2

def make_loaders(data_root, img_size=224, batch_size=32):
    common_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(), common_norm
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(), common_norm
    ])
    train_ds = datasets.ImageFolder(Path(data_root)/"train", transform=train_tf)
    val_ds   = datasets.ImageFolder(Path(data_root)/"val",   transform=val_tf)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_ld   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_ld, val_ld, train_ds.classes

def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    all_y, all_p, total_loss = [], [], 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            all_y.extend(y.cpu().tolist())
            all_p.extend(preds.cpu().tolist())
    acc = accuracy_score(all_y, all_p)
    f1  = f1_score(all_y, all_p, average="macro")
    return total_loss/len(loader.dataset), acc, f1

def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    train_ld, val_ld, class_names = make_loaders(args.data, args.img_size, args.batch_size)
    num_classes = len(class_names)

    model = get_mobilenet_v2(num_classes=num_classes, pretrained=True).to(device)
    # Freeze backbone except classifier for faster convergence
    for p in model.features.parameters():
        p.requires_grad = False

    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    best_f1, best_path = -1.0, Path(args.out)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for x, y in tqdm(train_ld, desc=f"Epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        tr_loss = running/len(train_ld.dataset)
        va_loss, va_acc, va_f1 = evaluate(model, val_ld, device)
        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_acc={va_acc:.3f} | val_f1={va_f1:.3f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ Saved best model to {best_path} (macro-F1 {best_f1:.3f})")

    meta = {"classes": class_names, "img_size": args.img_size, "best_val_macro_f1": best_f1}
    save_json(meta, best_path.with_suffix(".meta.json"))
    print("Done. Best weights:", best_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root folder containing train/ val/ test/ subfolders")
    ap.add_argument("--out", default="models/mobilenet_v2_office.pth")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    train(args)
