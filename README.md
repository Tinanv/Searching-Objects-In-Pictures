# Searching Objects In Pictures

Click-to-search visual product search: upload a photo, click an object, get matching products from Torob with links. Pipeline: Brave scraper → background removal → YOLO + CLIP embeddings in FAISS → Gradio app.

## 1. Setup

Requires Python 3.10+. No browser install needed: the scraper auto-uses
Brave/Edge/Chrome if present, else Playwright's bundled Chromium (one-time:
`.\venv\Scripts\playwright install chromium`).

```powershell
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
```

Heavy downloads happen once, automatically: YOLO weights (`*.pt`, already in repo), CLIP model (~350MB on first embed), rembg u2net (~176MB to `~/.u2net` on first index run).

## 2. Scrape categories

```powershell
Copy-Item categories.example.txt categories.txt
# edit categories.txt: one `slug<TAB>https://torob.com/browse/...` per line
.\venv\Scripts\python scrape_torob.py --max-items 200
```

- First run opens a browser window: click «من ربات نیستم» if challenged, then press Enter. Cookies persist in `.browser-profile/`; later runs can add `--headless`.
- Saves `dataset_images/<slug>/<sha1>.jpg` (cover image per card) and appends to `urls.txt`. Re-runs skip existing files (resume-safe).
- Useful flags: `--categories FILE`, `--delay 0.5` (seconds between downloads).

## 3. Build the index

```powershell
.\venv\Scripts\python process_dataset.py
```

- YOLO-seg detects objects per image → background auto-removed (rembg, YOLO-mask fallback, original as last resort) → CLIP embeds → FAISS `dataset.index` + `database.csv`.
- Incremental: only new images are processed on re-runs.
- Full rebuild: delete `dataset.index` + `database.csv` first (required after changing `EMBED_MODEL`).
- Env knobs: `EMBED_MODEL=clip-ViT-L-14` (slower, slightly better), `BG_REMOVE=off|yolo|rembg|auto`, `REMBG_MODEL=isnet-general-use` (tighter edges, ~3x slower), `YOLO_WEIGHTS=yolov8m-seg.pt`.

## 4. Search

```powershell
.\venv\Scripts\python app.py
```

Opens a Gradio UI: upload a photo → click the object → top-6 matches with scores and product links. `EMBED_MODEL` must match the one used for indexing. `SIMILARITY_THRESHOLD=0.60` env-adjustable.

## Files

| File | Role |
|---|---|
| `scrape_torob.py` | Infinite-scroll Brave scraper |
| `process_dataset.py` | Detect → de-background → embed → index |
| `bg_remove.py` | Background removal helpers |
| `app.py` | Gradio click-to-search UI |
| `categories.txt` | Your category links (gitignored; see `categories.example.txt`) |
| `dataset.index` / `database.csv` | Built artifacts — re-index to regenerate |
