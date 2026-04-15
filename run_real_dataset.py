"""
Real Fakeddit Dataset — Full Training & Evaluation Pipeline
=============================================================
Uses the locally stored Fakeddit TSV files from:
  c:/Ghiri Laptop Backup Nov 3/Deep Learning Package/multimodal_only_samples/

Pipeline:
  1. Parse multimodal_train.tsv and download images from image_url
  2. Train the Dual-Stream CLIP model (3 epochs)
  3. Parse multimodal_validate.tsv → evaluate and save confusion matrix

Usage (from project root with venv activated):
  python run_real_dataset.py
  python run_real_dataset.py --samples 150
"""

import os, sys, time, argparse
import requests, pandas as pd
import torch, torch.nn as nn
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import CLIPProcessor

sys.path.append(os.path.dirname(__file__))
from src.dataset  import MultimodalFakeNewsDataset
from src.model    import MultimodalFakeNewsModel
from src.train    import train_model
from src.evaluate import evaluate_model

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(__file__)
TSV_DIR          = r"c:\Ghiri Laptop Backup Nov 3\Deep Learning Package\multimodal_only_samples"
TRAIN_TSV        = os.path.join(TSV_DIR, "multimodal_train.tsv")
VAL_TSV          = os.path.join(TSV_DIR, "multimodal_validate.tsv")
IMAGES_DIR       = os.path.join(BASE_DIR, "datasets", "real_images")
RESULTS_DIR      = os.path.join(BASE_DIR, "results")
CKPT_DIR         = os.path.join(BASE_DIR, "checkpoints")

# ── Image downloader ───────────────────────────────────────────────────────────
def download_single_image(row_data, split_name, out_dir):
    url, text, label, idx = row_data
    fname = os.path.join(out_dir, f"{split_name}_{idx:05d}.jpg")
    
    if os.path.exists(fname):
        return {"image_path": os.path.abspath(fname), "text": text, "label": label}
        
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img.thumbnail((384, 384)) # optimized resolution
            img.save(fname)
            return {"image_path": os.path.abspath(fname), "text": text, "label": label}
    except Exception:
        pass
    return None

def parse_and_download(tsv_path: str, split_name: str, max_samples: int = None) -> pd.DataFrame:
    print(f"\n[→] Loading {split_name} TSV: {tsv_path}")
    df = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip')
    
    df = df[df['image_url'].notna() & df['image_url'].str.startswith('http')]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    out_dir = os.path.join(IMAGES_DIR, split_name)
    os.makedirs(out_dir, exist_ok=True)

    tasks_data = []
    # Build task tuples: (url, text, label, index)
    for i, row in df.iterrows():
        if max_samples and i >= max_samples * 2: # buffer for dead links
            break
        text = str(row.get('clean_title', row.get('title', '')))
        label = float(row.get('2_way_label', 0))
        tasks_data.append((str(row['image_url']), text, label, i))

    valid = []
    target_count = max_samples if max_samples else len(tasks_data)
    print(f"[→] Multi-threaded downloading up to {target_count} images for {split_name} …")
    
    # Use 20 workers for rapid scraping
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(download_single_image, data, split_name, out_dir): data for data in tasks_data}
        for future in tqdm(as_completed(futures), total=len(futures)):
            if len(valid) >= target_count:
                # Cancel remaining
                for f in futures:
                    f.cancel()
                break
            res = future.result()
            if res is not None:
                valid.append(res)

    result = pd.DataFrame(valid)
    print(f"    ✓ {len(result)} usable samples "
          f"({int(result['label'].sum())} real, "
          f"{len(result)-int(result['label'].sum())} fake)")
    return result


# ── Main ───────────────────────────────────────────────────────────────────────
def main(num_samples: int = 120):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*58}")
    print(f"  Multimodal Fake News Detector — Real Fakeddit Trial")
    print(f"  Device : {device.upper()}    |    Samples (train) : {num_samples}")
    print(f"{'='*58}")

    # 1. Parse & download training images
    train_df = parse_and_download(TRAIN_TSV, "train", max_samples=num_samples)
    # Use validate split for val+test (download max 40 samples)
    val_all  = parse_and_download(VAL_TSV,   "val",   max_samples=40)

    if len(train_df) < 10 or len(val_all) < 4:
        print("[ERROR] Not enough samples downloaded. Check internet connection.")
        sys.exit(1)

    # Split val_all into val / test (50 / 50)
    stratify = val_all['label'] if val_all['label'].nunique() > 1 else None
    val_df, test_df = train_test_split(val_all, test_size=0.5,
                                       random_state=42, stratify=stratify)

    # 2. Build DataLoaders
    processor    = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    train_loader = DataLoader(MultimodalFakeNewsDataset(train_df, processor),
                              batch_size=4, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(MultimodalFakeNewsDataset(val_df,   processor),
                              batch_size=4, shuffle=False, num_workers=0)
    test_loader  = DataLoader(MultimodalFakeNewsDataset(test_df,  processor),
                              batch_size=4, shuffle=False, num_workers=0)

    # 3. Model, loss, optimiser
    model     = MultimodalFakeNewsModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2
    )

    # 4. Train
    os.makedirs(CKPT_DIR, exist_ok=True)
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=3, device=device, save_dir=CKPT_DIR)

    # 5. Load best checkpoint and evaluate
    best_ckpt = os.path.join(CKPT_DIR, "best_model.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print(f"\n[✓] Loaded best checkpoint → {best_ckpt}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics = evaluate_model(model, test_loader, device=device, save_dir=RESULTS_DIR)

    # 6. Summary
    print(f"\n{'='*58}")
    print("  FINAL TEST RESULTS  (Real Fakeddit Data)")
    print(f"{'='*58}")
    for k, v in metrics.items():
        bar = "█" * int(v * 20)
        print(f"  {k:<14}: {v:.4f}  {bar}")
    print(f"\n  Confusion matrix → {RESULTS_DIR}/confusion_matrix.png")
    print(f"  Best checkpoint  → {CKPT_DIR}/best_model.pth")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3000,
                        help="Number of training samples to download (default 3000)")
    args = parser.parse_args()
    main(num_samples=args.samples)
