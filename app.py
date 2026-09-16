import os
import cv2
import faiss
import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer

# variables

MODEL_NAME = "clip-ViT-L-14"

YOLO_WEIGHTS = "yolov8n-seg.pt"

INDEX_FILE = "dataset.index"
CSV_FILE = "database.csv"
URL_FILE = "urls.txt"

TOP_K_SEARCH = 50

SIMILARITY_THRESHOLD = 0.50

YOLO_CONF = 0.25

ELECTRONICS = {
    "cell phone",
    "laptop",
    "tv",
    "mouse",
    "remote",
    "keyboard"
}

# loading models & stuffs

print("Loading YOLO...")

yolo_model = YOLO(YOLO_WEIGHTS)

print("Loading CLIP...")

clip_model = SentenceTransformer(MODEL_NAME)

print("Loading FAISS...")

index = faiss.read_index(INDEX_FILE)

print("Loading database...")

df = pd.read_csv(CSV_FILE)

# urls

url_dict = {}
if os.path.exists(URL_FILE):
    with open(URL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                key = parts[0].strip()
                link = parts[1].strip()
                url_dict[key] = link
                url_dict[f"{key}.jpg"] = link
                if key.endswith(".jpg"):
                    url_dict[key[:-4]] = link

print()
print("Index vectors :", index.ntotal)
print("Index dimension :", index.d)
print("Database rows :", len(df))
print("Loaded URLs from urls.txt :", len(url_dict))
print()

# multi_click_segment

def segment_multi_click(img_rgb, points):
    if not points:
        return None, None

    h, w = img_rgb.shape[:2]
    pts = np.array(points, dtype=np.int32)
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)

    # Adaptive bounding box margin enclosing all clicked points
    span_w = max_x - min_x
    span_h = max_y - min_y
    pad_x = max(70, int(span_w * 0.6) + 40)
    pad_y = max(80, int(span_h * 0.6) + 50)

    x1 = max(0, min_x - pad_x)
    x2 = min(w, max_x + pad_x)
    y1 = max(0, min_y - pad_y)
    y2 = min(h, max_y + pad_y)

    roi = img_rgb[y1:y2, x1:x2].copy()
    rh, rw = roi.shape[:2]
    if rh < 20 or rw < 20:
        return None, None

    # Mask: 0=BGD, 1=FGD, 2=PR_BGD, 3=PR_FGD
    mask = np.full((rh, rw), cv2.GC_PR_BGD, dtype=np.uint8)

    roi_pts = []
    for px, py in points:
        rx = max(0, min(px - x1, rw - 1))
        ry = max(0, min(py - y1, rh - 1))
        roi_pts.append((rx, ry))
        # Definite foreground seed at click
        cv2.circle(mask, (rx, ry), 12, cv2.GC_FGD, -1)
        # Probable foreground zone around click
        cv2.circle(mask, (rx, ry), 28, cv2.GC_PR_FGD, -1)

    # Connect multiple clicks with foreground bridges to expand smoothly
    if len(roi_pts) > 1:
        for i in range(len(roi_pts) - 1):
            cv2.line(mask, roi_pts[i], roi_pts[i + 1], cv2.GC_PR_FGD, thickness=26)
            cv2.line(mask, roi_pts[i], roi_pts[i + 1], cv2.GC_FGD, thickness=10)

    # Re-apply definite foreground on the click centers
    for rx, ry in roi_pts:
        cv2.circle(mask, (rx, ry), 10, cv2.GC_FGD, -1)

    # Outer border of ROI is definite background (wall/table/adjacent box)
    b_margin = max(4, int(min(rw, rh) * 0.04))
    mask[:b_margin, :] = cv2.GC_BGD
    mask[-b_margin:, :] = cv2.GC_BGD
    mask[:, :b_margin] = cv2.GC_BGD
    mask[:, -b_margin:] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(roi, mask, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)
        bin_mask = np.where((mask == cv2.GC_FGD) | 
                (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        return None, None

    # Keep connected components that contain any of the clicked points
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask)
    obj_roi_mask = np.zeros((rh, rw), dtype=np.uint8)

    for rx, ry in roi_pts:
        lbl = labels[ry, rx]
        if lbl > 0:
            obj_roi_mask[labels == lbl] = 255
        else:
            sub = labels[max(0, ry - 3):min(rh, ry + 4), max(0, rx - 3):min(rw, rx + 4)]
            val = sub[sub > 0]
            if len(val) > 0:
                best_lbl = np.bincount(val).argmax()
                obj_roi_mask[labels == best_lbl] = 255

    # Guarantee all clicked seeds are in the mask
    for rx, ry in roi_pts:
        cv2.circle(obj_roi_mask, (rx, ry), 8, 255, -1)

    # Fill internal holes (important for clear plastic/tubes/reflection)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    obj_roi_mask = cv2.morphologyEx(obj_roi_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(obj_roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    cv2.drawContours(obj_roi_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Map back to original image dimensions
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = obj_roi_mask

    bx, by, bw, bh = cv2.boundingRect(full_mask)
    if bw < 15 or bh < 15:
        return None, None

    pad = 10
    bx1 = max(0, bx - pad)
    by1 = max(0, by - pad)
    bx2 = min(w, bx + bw + pad)
    by2 = min(h, by + bh + pad)

    crop_rgb = img_rgb[by1:by2, bx1:bx2]
    crop_m = full_mask[by1:by2, bx1:bx2]

    # Clean white background for catalog packshot matching
    segmented_white = np.full_like(crop_rgb, 255)
    segmented_white[crop_m == 255] = crop_rgb[crop_m == 255]

    return segmented_white, full_mask

# fallback crop

def create_fallback_crop(image_rgb, click_x, click_y):
    size = 150
    h, w = image_rgb.shape[:2]
    y1 = max(0, click_y - size)
    y2 = min(h, click_y + size)
    x1 = max(0, click_x - size)
    x2 = min(w, click_x + size)
    return image_rgb[y1:y2, x1:x2]

# search

def segment_and_search(image, evt: gr.SelectData, points_history):
    if image is None:
        return None, [], "Please upload an image.", points_history

    if points_history is None:
        points_history = []

    if isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
    else:
        img_rgb = np.asarray(image)
        if img_rgb.ndim == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

    try:
        click_x = int(evt.index[0])
        click_y = int(evt.index[1])
    except Exception:
        return None, [], "Could not read click position.", points_history

    print()
    print("=" * 60)
    print("SEARCH")
    print(f"Click: ({click_x}, {click_y}) | Previous points: {len(points_history)}")
    print("=" * 60)

    # 1. First check if YOLO detects an object (phones, clothes, etc.)
    results = yolo_model(img_rgb, conf=YOLO_CONF, verbose=False)
    query_category = "unknown"
    search_crop = None
    display_image = None
    found_segmented_object = False

    if results and results[0].masks is not None:
        polygons = results[0].masks.xy
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        candidates = []
        for i, polygon in enumerate(polygons):
            contour = np.asarray(polygon, dtype=np.int32)
            if cv2.pointPolygonTest(contour, (click_x, click_y), False) >= 0:
                candidates.append((i, polygon, boxes[i], classes[i], confidences[i]))

        if candidates:
            candidates.sort(key=lambda x: ((x[2][2] - x[2][0]) * (x[2][3] - x[2][1])))
            i, polygon, box, cls_id, confidence = candidates[0]
            query_category = yolo_model.names[int(cls_id)]

            x1, y1, x2, y2 = map(int, box)
            h, w = img_rgb.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [np.asarray(polygon, dtype=np.int32)], 
                             -1, 255, thickness=cv2.FILLED)

            crop_rgb = img_rgb[y1:y2, x1:x2]
            crop_m = mask[y1:y2, x1:x2]
            search_crop = np.full_like(crop_rgb, 255)
            search_crop[crop_m == 255] = crop_rgb[crop_m == 255]

            display_image = search_crop
            found_segmented_object = True
            points_history = []  # Clear points since YOLO segmented full item
            print("Detected via YOLO:", query_category, f"({confidence:.3f})")

    # 2. If YOLO did not detect, activate multi-click expansion mode
    if not found_segmented_object:
        points_history = list(points_history) + [(click_x, click_y)]
        print(f"YOLO not found. Expanding with {len(points_history)} point(s)...")

        seg_crop, seg_mask = segment_multi_click(img_rgb, points_history)

        if seg_crop is not None:
            search_crop = seg_crop
            display_image = seg_crop
            found_segmented_object = True
            query_category = "product"
            print(f"Successfully expanded mask with {len(points_history)} click(s)!")
        else:
            print("Fallback to rectangular crop.")
            search_crop = create_fallback_crop(img_rgb, click_x, click_y)
            display_image = search_crop
            query_category = "unknown"

    if search_crop is None or search_crop.size == 0:
        return None, [], "No object found.", points_history

    # rotation search
    rotations = [
        search_crop,
        cv2.rotate(search_crop, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(search_crop, cv2.ROTATE_180),
        cv2.rotate(search_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]

    pil_images = [Image.fromarray(r) for r in rotations]
    vectors = clip_model.encode(pil_images, normalize_embeddings=True)
    query_vecs = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(query_vecs)

    k = min(TOP_K_SEARCH, index.ntotal)
    distances, indices = index.search(query_vecs, k)

    best_scores = {}
    for r in range(len(indices)):
        for pos in range(len(indices[r])):
            idx = int(indices[r][pos])
            score = float(distances[r][pos])
            if idx == -1:
                continue
            if idx not in best_scores or score > best_scores[idx]:
                best_scores[idx] = score

    sorted_results = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)

    # results

    matches = []
    url_lines = []
    seen_images = set()

    for idx, score in sorted_results:
        if idx >= len(df):
            continue

        row = df.iloc[idx]
        local_path = str(row.get("local_path", ""))

        if local_path in seen_images:
            continue

        db_type = str(row.get("search_type", "product")) if "search_type" in df.columns else "product"
        query_is_electronic = (query_category in ELECTRONICS)

        if query_is_electronic:
            if db_type != "electronics":
                continue
        else:
            if db_type == "electronics":
                continue

        if score < SIMILARITY_THRESHOLD:
            continue

        # get URL from url.txt
        img_name = str(row.get("image_name", "")).strip()
        base_name = os.path.basename(local_path).strip()
        stem_name = os.path.splitext(img_name)[0]
        base_stem = os.path.splitext(base_name)[0]

        url = ""
        for key in [img_name, base_name, stem_name, base_stem]:
            if key in url_dict:
                url = url_dict[key]
                break

        matches.append((local_path, f"{score:.2% readiness}" if False else f"{score:.2%}"))
        category = str(row.get("category", "unknown"))
        url_lines.append(f"Score {score:.2%} [{category}]: {url}")
        seen_images.add(local_path)

        if len(matches) >= 6:
            break

    if not matches:
        top_raw_score = sorted_results[0][1] if sorted_results else 0.0
        message = f"No matches found.\n\nDetected object: {query_category}\n\nTop raw score: {top_raw_score:.2%}"
        print(message)
        return display_image, [], message, points_history

    url_text = "\n\n".join(url_lines)

    print("Detected object:", query_category)
    print("Results:", len(matches))
    for path, score in matches:
        print(score, path)

    return display_image, matches, url_text, points_history


def reset_selection():
    return None, [], "", []

# gradio UI

with gr.Blocks() as demo:
    gr.Markdown("# Object Search")
    gr.Markdown(
        "Upload an image and click on an object.\n"
        "**For makeup/cosmetics:** Click multiple times (e.g. body, cap, top) to expand and perfect the selection just like Photoshop!"
    )

    points_state = gr.State(value=[])

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload & Click to Select / Expand", type="numpy")
            reset_btn = gr.Button("Reset Clicks", variant="secondary")

        with gr.Column(scale=1):
            segmented_display = gr.Image(label="Your Selection (Expands with clicks)")

    gallery_display = gr.Gallery(label="Matches", columns=3, rows=2, height="auto")
    url_display = gr.Textbox(label="Product Links", lines=8)

    input_img.select(
        segment_and_search,
        inputs=[input_img, points_state],
        outputs=[segmented_display, gallery_display, url_display, points_state]
    )

    reset_btn.click(
        reset_selection,
        inputs=[],
        outputs=[segmented_display, gallery_display, url_display, points_state]
    )

    input_img.change(
        reset_selection,
        inputs=[],
        outputs=[segmented_display, gallery_display, url_display, points_state]
    )

demo.launch()