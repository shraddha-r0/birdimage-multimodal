import os, random
from datasets import load_dataset

# Config
DS_NAME = "vishnun0027/BirdsSpecies"
SPLIT = "train"           # adjust if the dataset has other splits
MAX_SAMPLES = 200       
OUT_DIR = "data/birds"

os.makedirs(OUT_DIR, exist_ok=True)

# Load
ds = load_dataset(DS_NAME, split=SPLIT)
print(ds)               # see features
print(ds.features)

# Take a random subset
idx = list(range(len(ds)))
random.shuffle(idx)
idx = idx[:MAX_SAMPLES]
ds_small = ds.select(idx)

# Persist images to disk with basic metadata in filename
from PIL import Image

def safe(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c in ("_", "-")).strip()

counts = 0
for ex in ds_small:
    # many HF vision datasets expose PIL images directly under 'image'
    img = ex["image"]  # PIL.Image
    species = safe(str(ex.get("label", ex.get("species", "unknown"))))
    img = img.convert("RGB")
    fname = f"{species}_{counts:05d}.jpg"
    img.save(os.path.join(OUT_DIR, fname), quality=90)
    counts += 1

print(f"Saved {counts} images to {OUT_DIR}")