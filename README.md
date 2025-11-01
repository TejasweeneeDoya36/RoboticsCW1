# Office Organizer Robot – Object Classification (PDE3802 – AI in Robotics)


A lightweight MobileNetV2 image classifier that recognizes common office items and a desktop GUI for live webcam demos.


## How it works
1. **Prepare data** – Put your dataset under `data/all/<class_name>/*.jpg`.
2. **Split** – `python split_from_all.py` creates `data/train`, `data/val`, `data/test`.
3. **Train** – `python train.py --data data --out models/mobilenet_v2_office.pth`.
4. **Evaluate** – `python eval.py --data data/test --weights models/mobilenet_v2_office.pth --classes docs/classes.txt --out results/`.
5. **Run GUI** – `python app_gui.py` → Live Camera / Classify Image.


## Metrics we report
- Accuracy (overall)
- Precision (macro), Recall (macro), F1 (macro)
- AUC‑ROC (macro, one‑vs‑rest)
- MAE between one‑hot ground truth and predicted probability vectors


## Project structure

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/office-organizer-robot.git
   cd office-organizer-robot
