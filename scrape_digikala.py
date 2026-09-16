import argparse
import hashlib
import os
import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from PIL import Image
from playwright.sync_api import sync_playwright

BROWSER_PATHS = [
    r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


def resolve_browser(choice: str):
    """Installed browser if found, else bundled Chromium."""
    if choice in ("auto", "browser"):
        for p in BROWSER_PATHS:
            if os.path.exists(p):
                print(f"Using installed browser: {p}")
                return {"executable_path": p}
        if choice == "browser":
            raise SystemExit("No browser found. Use --browser chromium.")
    print("Using bundled Chromium.")
    return {}


PROFILE = os.path.join(os.getcwd(), ".browser-profile")
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "Chrome/126.0 Safari/537.36"
)

# Digikala OSS image resize parameters
SIZE_RE = re.compile(r"h_\d+,w_\d+")
QUALITY_RE = re.compile(r"q_\d+")


def best_img(src: str, srcset: str = "") -> str:
    """Extracts and enhances the image URL to request 800x800 high quality from Digikala CDN."""
    if srcset:
        parts = [p.strip().split(" ")[0] for p in srcset.split(",") if p.strip()]
        if parts:
            src = parts[-1]

    if not src:
        return ""

    # Digikala CDN uses Alibaba Cloud OSS image processing
    if "x-oss-process" in src:
        src = SIZE_RE.sub("h_800,w_800", src)
        src = QUALITY_RE.sub("q_90", src)
    elif "dkstatics-public.digikala.com" in src and "?" not in src:
        src += "?x-oss-process=image/resize,m_lfit,h_800,w_800/quality,q_90"

    return src


def parse_cards(html: str, base: str = "https://www.digikala.com"):
    soup = BeautifulSoup(html, "html.parser")
    out = []
    seen = set()

    # Find all product link elements containing Digikala product ID pattern (/product/dkp-...)
    product_links = soup.find_all("a", href=re.compile(r"/product/dkp-\d+"))
    for a in product_links:
        href = a.get("href", "")
        if not href:
            continue

        clean_url = urljoin(base, href.split("?")[0].split("#")[0])
        if clean_url in seen:
            continue

        # Look for the card container (article or wrapper parent)
        card = a.find_parent("article")
        if not card:
            curr = a
            for _ in range(4):
                if curr.parent:
                    curr = curr.parent
                    if curr.name in ("article", "div") and (curr.find("img") or curr.find("h3")):
                        card = curr
                        break
            if not card:
                card = a

        # Locate product image (handling lazy loading attributes: data-src, src, srcset)
        img = None
        for i in card.find_all("img"):
            c_src = i.get("data-src") or i.get("src") or ""
            if "dkstatics-public.digikala.com" in c_src or "digikala-products" in c_src:
                img = i
                break
        if not img:
            for i in card.find_all("img"):
                c_src = i.get("data-src") or i.get("src") or ""
                if c_src and not c_src.startswith("data:"):
                    img = i
                    break

        if not img:
            continue

        raw_src = img.get("data-src") or img.get("src") or ""
        srcset = img.get("srcset", "") or img.get("data-srcset", "")
        src = best_img(raw_src, srcset)

        if not src or src.startswith("data:image"):
            continue

        # Extract title (from h3, h2, img alt, or link title)
        h = card.find(["h3", "h2"])
        title = (h.get_text(strip=True) if h else (img.get("alt") or a.get("title") or "")).strip()

        seen.add(clean_url)
        out.append((urljoin(base, src), clean_url, title))

    return out


def valid_jpg(path: str) -> bool:
    try:
        if os.path.getsize(path) < 10_000:
            return False
        with Image.open(path) as im:
            im.verify()
        return True
    except OSError:
        return False


def download(session: requests.Session, url: str, dest: str) -> bool:
    for _ in range(3):
        try:
            r = session.get(url, timeout=20)
            if r.status_code != 200 or "image" not in r.headers.get("Content-Type", ""):
                return False
            with open(dest, "wb") as f:
                f.write(r.content)
            if valid_jpg(dest):
                return True
        except requests.RequestException:
            time.sleep(1)
    if os.path.exists(dest):
        os.remove(dest)
    return False


def ensure_human(pg):
    title = pg.title()
    content = pg.content()[:4000]
    bot_triggers = ["ربات", "کپچا", "Just a moment...", "Cloudflare", "Security Check", "من ربات نیستم"]
    if any(k in title for k in bot_triggers) or any(k in content for k in bot_triggers):
        print("\n[!] Bot/Security check detected.")
        print("    Please complete the check in the open browser window, then press Enter here...")
        try:
            input()
        except EOFError:
            pg.wait_for_timeout(30000)
        pg.wait_for_timeout(3000)


def scroll_collect(pg, base: str, max_items: int):
    found, still, prev = {}, 0, -1

    # Scroll down to load infinite scroll / lazy-loaded items
    while len(found) < max_items and still < 5:
        pg.evaluate("window.scrollBy(0, window.innerHeight * 2)")
        pg.wait_for_timeout(1400)
        pg.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        pg.wait_for_timeout(1800)

        for item in parse_cards(pg.content(), base):
            found.setdefault(item[1], item)
            if len(found) >= max_items:
                break

        still = still + 1 if len(found) == prev else 0
        prev = len(found)

    # If category page uses pagination parameters (?page=2) to load beyond infinite scroll
    page_num = 2
    while len(found) < max_items and page_num <= 10 and still >= 5:
        sep = "&" if "?" in base else "?"
        next_url = f"{base}{sep}page={page_num}"
        try:
            pg.goto(next_url, timeout=45000)
            pg.wait_for_timeout(3000)
            ensure_human(pg)

            page_still = 0
            while len(found) < max_items and page_still < 3:
                pg.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                pg.wait_for_timeout(1800)
                before_len = len(found)
                for item in parse_cards(pg.content(), base):
                    found.setdefault(item[1], item)
                    if len(found) >= max_items:
                        break
                page_still = page_still + 1 if len(found) == before_len else 0

            page_num += 1
        except Exception:
            break

    return list(found.values())[:max_items]


def scrape_category(pg, session, slug, url, max_items, delay):
    os.makedirs(os.path.join("dataset_images", slug), exist_ok=True)
    pg.goto(url, timeout=60000)
    pg.wait_for_timeout(4000)
    ensure_human(pg)

    cards = scroll_collect(pg, url, max_items)
    print(f"[{slug}] {len(cards)} cards listed")

    got = 0
    with open("urls.txt", "a", encoding="utf-8") as log:
        for img_url, prod_url, title in cards:
            name = hashlib.sha1(prod_url.encode()).hexdigest()[:16] + ".jpg"
            dest = os.path.join("dataset_images", slug, name)

            if os.path.exists(dest) and valid_jpg(dest):
                continue

            if download(session, img_url, dest):
                log.write(f"{slug}/{name}\t{prod_url}\t{title}\n")
                got += 1

            time.sleep(delay)

    print(f"[{slug}] +{got} images")
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="categories2.txt",
                    help="Path to categories file (default: categories2.txt)")
    ap.add_argument("--max-items", type=int, default=100)
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--headless", action="store_true",
                    help="hide browser window (use after initial trusted run)")
    ap.add_argument("--browser", default="auto", choices=["auto", "browser", "chromium"],
                    help="auto: installed Brave/Edge/Chrome else bundled Chromium")
    args = ap.parse_args()

    if not os.path.exists(args.categories):
        print(f"Missing {args.categories} — please create it with lines formatted as:")
        print("slug<TAB>https://www.digikala.com/search/category-.../")
        return

    jobs = []
    with open(args.categories, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line, maxsplit=1)
            if len(parts) == 2:
                jobs.append((parts[0].strip(), parts[1].strip()))

    if not jobs:
        print("No categories found.")
        return

    session = requests.Session()
    session.headers.update({
        "User-Agent": UA,
        "Referer": "https://www.digikala.com/",
    })

    total = 0
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            PROFILE,
            headless=args.headless,
            user_agent=UA,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
            **resolve_browser(args.browser)
        )
        pg = ctx.new_page()

        for slug, url in jobs:
            try:
                total += scrape_category(pg, session, slug, url, args.max_items, args.delay)
            except Exception as e:
                print(f"[{slug}] failed: {e}")

        pg.close()
        ctx.close()

    print(f"Done: +{total} images.")


if __name__ == "__main__":
    main()