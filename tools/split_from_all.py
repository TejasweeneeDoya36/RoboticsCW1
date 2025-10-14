from pathlib import Path
import shutil, random

SRC = Path("data/all")   # has <class>/images
DST = Path("data")       # will produce train/ val/ test structure
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

random.seed(42)

def main():
    classes = [p for p in SRC.iterdir() if p.is_dir()]
    if not classes:
        print("No class folders in data/all"); return
    for split in SPLITS:
        for c in classes:
            (DST/split/c.name).mkdir(parents=True, exist_ok=True)
    for c in classes:
        files = [p for p in c.iterdir() if p.suffix.lower() in EXTS]
        random.shuffle(files)
        n = len(files); n_tr = int(n*SPLITS["train"]); n_va = int(n*SPLITS["val"])
        parts = {"train": files[:n_tr], "val": files[n_tr:n_tr+n_va], "test": files[n_tr+n_va:]}
        for split, items in parts.items():
            for src in items:
                shutil.copy2(src, DST/split/c.name/src.name)
        print(f"{c.name}: {n} -> train {len(parts['train'])}, val {len(parts['val'])}, test {len(parts['test'])}")

if __name__ == "__main__":
    main()

