import os
import cv2
import pandas as pd
import numpy as np
import faiss
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from PIL import Image

# 1. Load Models
print("Loading Models (ViT-L-14)...")
yolo_model = YOLO("yolov8n-seg.pt")
clip_model = SentenceTransformer("clip-ViT-L-14") 

IMAGE_FOLDER = "dataset_images"
URL_FILE = "urls.txt" 
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif")

# --- Load Real URLs ---
url_map = {}
if os.path.exists(URL_FILE):
    with open(URL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                url_map[parts[0].strip()] = parts[1].strip()

all_vectors = []
data_list = []
yolo_names = yolo_model.names

print("Processing dataset...")

for filename in os.listdir(IMAGE_FOLDER):
    if not filename.lower().endswith(VALID_EXTENSIONS): continue
    img_path = os.path.join(IMAGE_FOLDER, filename)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: continue

    pure_id = os.path.splitext(filename)[0]
    real_url = url_map.get(filename) or url_map.get(pure_id) or "No URL found"

    results = yolo_model(img_bgr)

    if results[0].masks is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        
        for i, box in enumerate(boxes):
            category_name = yolo_names[int(class_ids[i])]
            x1, y1, x2, y2 = map(int, box)
            crop_bgr = img_bgr[y1:y2, x1:x2]
            if crop_bgr.size == 0: continue
            
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            vector = clip_model.encode(Image.fromarray(crop_rgb), normalize_embeddings=True)
            
            all_vectors.append(vector)
            data_list.append({
                "image_name": filename,
                "local_path": img_path,
                "url": real_url,
                "category": category_name
            })
    else:
        # Fallback
        crop_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        vector = clip_model.encode(Image.fromarray(crop_rgb), normalize_embeddings=True)
        all_vectors.append(vector)
        data_list.append({
            "image_name": filename,
            "local_path": img_path,
            "url": real_url,
            "category": "unknown"
        })

if all_vectors:
    vectors_np = np.array(all_vectors).astype("float32")
    index = faiss.IndexFlatIP(vectors_np.shape[1]) 
    index.add(vectors_np)
    faiss.write_index(index, "dataset.index")
    pd.DataFrame(data_list).to_csv("database.csv", index=False)
    print("\nSuccess! Database re-indexed.")