import os
import gradio as gr
import lancedb
from PIL import Image
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

DB = lancedb.connect("db").open_table("bird_images")
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_text(q: str) -> np.ndarray:
    inputs = clip_proc(text=[q], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        feats = clip_model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0]

def embed_image(img: Image.Image) -> np.ndarray:
    inputs = clip_proc(images=[img.convert("RGB")], return_tensors="pt").to(device)
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0]

def search_text(q, k=8, species_filter=""):
    try:
        vec = embed_text(q)
        query = DB.search(vec).limit(int(k))
        if species_filter and species_filter.strip():
            query = query.where(f"species = '{species_filter.strip()}'")
        hits = query.to_pandas().to_dict('records')
        return [(h["path"], f"{h.get('species', 'unknown')}") for h in hits]
    except Exception as e:
        print(f"Error in text search: {e}")
        return []

def search_by_image(img, k=8):
    try:
        vec = embed_image(img)
        hits = DB.search(vec).limit(int(k)).to_pandas().to_dict('records')
        return [(h["path"], f"{h.get('species', 'unknown')}") for h in hits]
    except Exception as e:
        print(f"Error in image search: {e}")
        return []

# Build species dropdown from table
try:
    # Get all species from the table
    species_vals = sorted({row["species"] for row in DB.to_pandas().to_dict('records')})
    species_dropdown = gr.Dropdown(choices=species_vals, label="Filter by species (optional)", value=None)
except Exception as e:
    print(f"Error initializing species dropdown: {e}")
    species_vals = []
    species_dropdown = gr.Dropdown(choices=[], label="Filter by species (not available)", value=None)

with gr.Blocks(title="BirdSearch") as demo:
    gr.Markdown("# 🐦 BirdSearch\nCLIP-powered text ↔ image retrieval on the Ez-Clap/bird-species dataset.")
    with gr.Tab("Text → Image"):
        q = gr.Textbox(label="Describe the bird / photo style (e.g., 'small brown bird with streaked chest', 'blue head red throat', 'close-up portrait')")
        k = gr.Slider(3, 16, value=8, step=1, label="Top-K")
        filt = species_dropdown
        out = gr.Gallery(label="Results", columns=4, height="auto")
        gr.Button("Search").click(search_text, inputs=[q, k, filt], outputs=out)
        q.submit(search_text, inputs=[q, k, filt], outputs=out)
    with gr.Tab("Image → Similar"):
        img = gr.Image(type="pil", label="Upload a bird image")
        k2 = gr.Slider(3, 16, value=8, step=1, label="Top-K")
        out2 = gr.Gallery(label="Nearest neighbors", columns=4, height="auto")
        gr.Button("Find Similar").click(search_by_image, inputs=[img, k2], outputs=out2)

demo.launch()