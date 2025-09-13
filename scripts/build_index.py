import os
import glob
import numpy as np
import pyarrow as pa
import lancedb
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Configuration
DB_DIR = "db"
IMG_DIR = "data/birds"
TABLE = "bird_images"
BATCH_SIZE = 32
EMBEDDING_DIM = 512  # CLIP base model produces 512-dimensional vectors

# Set up device and load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_images(paths):
    """Embed a batch of images using CLIP model."""
    try:
        batch = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                batch.append(img)
            except Exception as e:
                print(f"Error loading {p}: {e}")
                continue
                
        if not batch:
            return None
            
        inputs = clip_proc(images=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()
    except Exception as e:
        print(f"Error in embed_images: {e}")
        return None

def create_schema():
    """Create a schema for the LanceDB table."""
    return pa.schema([
        ("path", pa.string()),
        ("species", pa.string()),
        ("vector", pa.list_(pa.float32(), EMBEDDING_DIM))
    ])

def main():
    # Connect to LanceDB
    print(f"Connecting to LanceDB at {DB_DIR}")
    db = lancedb.connect(DB_DIR)
    
    # Get list of image files
    image_pattern = os.path.join(IMG_DIR, "*.jpg")
    paths = sorted(glob.glob(image_pattern))
    
    if not paths:
        print(f"No images found in {IMG_DIR}")
        return
        
    print(f"Found {len(paths)} images to process")
    
    # Create or recreate the table with proper schema
    schema = create_schema()
    if TABLE in db.table_names():
        print(f"Dropping existing table: {TABLE}")
        db.drop_table(TABLE)
    
    print(f"Creating new table: {TABLE}")
    tbl = db.create_table(TABLE, schema=schema, mode="overwrite")
    
    # Process images in batches
    total_processed = 0
    rows = []
    
    for i in range(0, len(paths), BATCH_SIZE):
        chunk = paths[i:i + BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(paths)-1)//BATCH_SIZE + 1}")
        
        # Get embeddings for the current batch
        emb = embed_images(chunk)
        if emb is None or len(emb) == 0:
            print("Skipping batch due to errors")
            continue
            
        # Prepare batch data
        batch_rows = []
        for j, p in enumerate(chunk):
            if j >= len(emb):  # In case some images failed to process
                continue
                
            base = os.path.basename(p)
            # Extract species from filename (assuming format: species_XXXXX.jpg)
            species = base.split("_")[0] if "_" in base else "unknown"
            
            batch_rows.append({
                "path": p,
                "species": species,
                "vector": emb[j].astype(np.float32).tolist()
            })
        
        # Add batch to table
        if batch_rows:
            tbl.add(batch_rows)
            total_processed += len(batch_rows)
            print(f"Added {len(batch_rows)} rows (total: {total_processed})")
    
    print(f"\nIndexing complete!")
    print(f"Total images processed: {total_processed}")
    print(f"Total rows in table: {tbl.count_rows()}")
    print(f"Table schema: {tbl.schema}")

if __name__ == "__main__":
    main()