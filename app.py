import gradio as gr
import cv2
import numpy as np
import pandas as pd
import faiss
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from PIL import Image

# 1. Load Models & Database
yolo_model = YOLO('yolov8n-seg.pt')
clip_model = SentenceTransformer("clip-ViT-L-14") 
index = faiss.read_index("dataset.index")
df = pd.read_csv("database.csv")

# Lowered threshold to 0.60 to allow Black -> White shirt matches
SIMILARITY_THRESHOLD = 0.60 
TOP_K_SEARCH = 50 

# Define high-level groups to prevent Phones matching Clothes
ELECTRONICS = ['cell phone', 'laptop', 'tv', 'mouse', 'remote', 'keyboard']

def segment_and_search(image, evt: gr.SelectData):
    img_rgb = image 
    click_x, click_y = evt.index[0], evt.index[1]
    results = yolo_model(img_rgb, conf=0.3) 
    
    query_category = "unknown"
    search_crop = None
    display_image = None
    found_segmented_object = False

    if results[0].masks is not None:
        polygons = results[0].masks.xy
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()
        names = yolo_model.names

        for i, poly in enumerate(polygons):
            contour = np.array(poly, dtype=np.int32)
            if cv2.pointPolygonTest(contour, (click_x, click_y), False) >= 0:
                query_category = names[int(clss[i])]
                x1, y1, x2, y2 = map(int, boxes[i])
                search_crop = img_rgb[y1:y2, x1:x2]
                
                mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                display_image = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
                found_segmented_object = True
                break

    if not found_segmented_object:
        size = 150 
        y1, y2 = max(0, click_y - size), min(img_rgb.shape[0], click_y + size)
        x1, x2 = max(0, click_x - size), min(img_rgb.shape[1], click_x + size)
        search_crop = img_rgb[y1:y2, x1:x2]
        display_image = search_crop

    if search_crop is None: return None, None, "No object found."

    # Search
    query_vec = clip_model.encode(Image.fromarray(search_crop), normalize_embeddings=True)
    query_vec = np.array(query_vec).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vec, k=TOP_K_SEARCH)

    matches = []
    url_text = ""
    seen_images = set()

    for i in range(len(indices[0])):
        idx = indices[0][i]
        score = distances[0][i]
        if idx == -1: continue
        
        row = df.iloc[idx]
        db_category = str(row.get('category', 'unknown'))
        
        # --- SMART CATEGORY GUARD ---
        # 1. If we clicked a phone, ONLY show other electronics
        if query_category in ELECTRONICS:
            if db_category not in ELECTRONICS and db_category != "unknown":
                continue
        # 2. If we clicked something else (Shirt/Person), do NOT show electronics
        else:
            if db_category in ELECTRONICS:
                continue

        # Show if it passes the 0.60 threshold
        if score >= SIMILARITY_THRESHOLD and row['local_path'] not in seen_images:
            matches.append((row['local_path'], f"{score:.2%}"))
            url_text += f"Score {score:.2%} [{db_category}]: {row['url']}\n\n"
            seen_images.add(row['local_path'])
        
        if len(matches) >= 6: break

    return display_image, matches, url_text or "No matches found."

with gr.Blocks() as demo:
    gr.Markdown("# 🔍 AI Object Search (Smart Grouping)")
    input_img = gr.Image(label="Upload & Click")
    with gr.Row():
        segmented_display = gr.Image(label="Your Selection")
        gallery_display = gr.Gallery(label="Matches", columns=3)
    url_display = gr.Textbox(label="Links", lines=8)

    input_img.select(segment_and_search, [input_img], [segmented_display, gallery_display, url_display])

demo.launch()