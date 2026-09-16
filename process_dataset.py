import os
import cv2
import faiss
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer

#variables

DATASET_DIR = "dataset_images"
INDEX_FILE = "dataset.index"
CSV_FILE = "database.csv"
URLS_FILE = "urls.txt"
MODEL_NAME = "clip-ViT-L-14"
YOLO_WEIGHTS = "yolov8n-seg.pt"

IMAGE_EXTENSIONS = (".jpg",".jpeg",".png",".webp",".bmp",".avif")

ELECTRONICS = {"cell phone","laptop","tv","mouse","remote","keyboard"}

YOLO_CONF = 0.30

# loading modesl & stuffs

print("Loading CLIP model...")
clip_model = SentenceTransformer(
    MODEL_NAME,
    device="cuda"
)

print("Loading YOLO model...")
yolo_model = YOLO(YOLO_WEIGHTS)
yolo_model.to("cuda")

print("Models loaded.")

# urls

def load_urls():
    data = {}

    if not os.path.exists(URLS_FILE):
        print("WARNING: urls.txt not found.")
        return data

    with open(URLS_FILE, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if "\t" in line:
                parts = line.split("\t")
            else:
                parts = line.split(maxsplit=1)

            if len(parts) < 2:
                continue

            key = parts[0].strip()
            url = parts[1].strip()

            data[key.replace("\\", "/")] = url

    return data

URL_DATA = load_urls()

def get_url(category, filename, full_path):

    candidates = [
        f"{category}/{filename}",
        f"dataset_images/{category}/{filename}",
        filename,
        full_path.replace("\\", "/"),
    ]

    for key in candidates:

        key = key.replace("\\", "/")

        if key in URL_DATA:
            return URL_DATA[key]
        
    for key, url in URL_DATA.items():

        if key.endswith("/" + filename) or key == filename:
            return url

    return ""

def get_title_from_url(url):

    return ""

# finding images

def get_all_images():

    image_paths = []

    for root, dirs, files in os.walk(DATASET_DIR):

        for filename in files:

            if filename.lower().endswith(IMAGE_EXTENSIONS):

                full_path = os.path.join(root, filename)
                image_paths.append(full_path)

    image_paths.sort()

    return image_paths

# YOLO electronic crop

def get_electronics_crop(img_rgb):

    results = yolo_model(
    img_rgb,
    conf=YOLO_CONF,
    verbose=False,
    device=0
    )

    if not results:
        return None, None, False

    result = results[0]

    if result.boxes is None:
        return None, None, False

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    best_crop = None
    best_label = None
    best_conf = -1

    image_h, image_w = img_rgb.shape[:2]

    for box, cls_id, conf in zip(
        boxes,
        classes,
        confidences
    ):

        label = yolo_model.names[int(cls_id)]

        if label not in ELECTRONICS:
            continue

        if conf < best_conf:
            continue

        x1, y1, x2, y2 = map(int, box)

        x1 = max(0, min(x1, image_w - 1))
        y1 = max(0, min(y1, image_h - 1))
        x2 = max(0, min(x2, image_w))
        y2 = max(0, min(y2, image_h))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = img_rgb[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        best_crop = crop
        best_label = label
        best_conf = float(conf)

    if best_crop is None:
        return None, None, False

    return best_crop, best_label, True

# embedding

def get_embedding(image_rgb):

    pil_image = Image.fromarray(image_rgb)

    vector = clip_model.encode(
        pil_image,
        normalize_embeddings=True
    )

    vector = np.asarray(
        vector,
        dtype="float32"
    )

    return vector

# process image

def process_image(image_path):

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print("Could not read:", image_path)
        return None

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )
    filename = os.path.basename(image_path)
    relative_path = os.path.relpath(
        image_path,
        DATASET_DIR
    ).replace("\\", "/")
    parts = relative_path.split("/")
    if len(parts) >= 2:
        category = parts[0]
    else:
        category = "misc"

    # YOLO or CLIP

    electronics_crop, yolo_label, is_electronic = \
        get_electronics_crop(image_rgb)

    if is_electronic:
        embedding_image = electronics_crop
        search_type = "electronics"

        print(
            f"[ELECTRONICS] "
            f"{relative_path} -> "
            f"{yolo_label}"
        )

    else:
        #clothes and other things
        embedding_image = image_rgb
        search_type = "product"

        yolo_label = "product"

        print(
            f"[PRODUCT] "
            f"{relative_path}"
        )

    # CLIP

    vector = get_embedding(embedding_image)

    # making database

    url = get_url(
        category,
        filename,
        relative_path
    )

    title = get_title_from_url(url)

    row = {
        "image_name": filename,
        "local_path": os.path.join(
            DATASET_DIR,
            relative_path
        ).replace("\\", "/"),
        "url": url,
        "title": title,
        "category": category,
        "yolo_label": yolo_label,
        "search_type": search_type
    }

    return vector, row

# main

def main():

    image_paths = get_all_images()

    print()
    print("=" * 60)
    print("IMAGE SEARCH DATASET PROCESSING")
    print("=" * 60)
    print("Images:", len(image_paths))
    print("CLIP:", MODEL_NAME)
    print("YOLO:", YOLO_WEIGHTS)
    print("=" * 60)
    print()

    if len(image_paths) == 0:

        print("No images found.")

        return

    vectors = []
    rows = []

    # process images

    for number, image_path in enumerate(
        image_paths,
        start=1
    ):

        print(
            f"\n[{number}/{len(image_paths)}]"
        )

        try:

            result = process_image(image_path)
            if result is None:
                continue
            vector, row = result
            vectors.append(vector)
            rows.append(row)

        except Exception as e:
            print("ERROR:",image_path,e)

    #check

    if len(vectors) == 0:

        print("No vectors created.")
        return

    # convert to numpy 
     
    vectors = np.asarray(vectors,dtype="float32")

    # Normalize again for safety

    faiss.normalize_L2(vectors)

    # Create NEW FAISS index

    dimension = vectors.shape[1]

    print()
    print("Creating FAISS index...")
    print("Dimension:", dimension)

    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    # save index
    
    faiss.write_index(index,INDEX_FILE)

    # Save CSV(database)

    df = pd.DataFrame(rows)

    df.to_csv(CSV_FILE,index=False,encoding="utf-8-sig")

    # Final information

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print("Images processed :", len(rows))
    print("Vectors          :", index.ntotal)
    print("Vector dimension :", index.d)
    print("Index             :", INDEX_FILE)
    print("Database          :", CSV_FILE)
    print()
    print("Search type counts:")
    print(df["search_type"].value_counts())

if __name__ == "__main__":
    main()