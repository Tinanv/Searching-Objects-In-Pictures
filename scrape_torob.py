"""Torob browse scraper: cover image per card -> dataset_images/<slug>/.

Torob uses infinite scroll + a bot check, so listing runs in a real
browser (Brave/Edge/Chrome if installed, else Playwright's bundled
Chromium). First run: click "I'm not a robot" once if asked; cookies are
kept in .browser-profile/ for later runs. Image downloads stay on light
requests.

Usage:
    python scrape_torob.py [--categories categories.txt] [--max-items 200]

Input line format:  slug<TAB>https://torob.com/browse/...
Output: dataset_images/<slug>/<sha1(product_url)>.jpg + append to urls.txt
"""
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

BROWSER_PATHS = [  # auto-detected; nothing to install by hand
    r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",]


def resolve_browser(choice: str):
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
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "Chrome/126.0 Safari/537.36")
SIZE_RE = re.compile(r"_/0x\d+")


def best_img(srcset: str, src: str) -> str:
    if srcset:  # "u1 1x, u2 2x" take last (highest res)
        parts = [p.strip().split(" ")[0] for p in srcset.split(",") if p.strip()]
        if parts:
            src = parts[-1]
    return SIZE_RE.sub("_/0x800", src)  # ask torob CDN for big cover


def parse_cards(html: str, base: str):
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select('div[class*="ProductCard_desktop_card"]')
    if not cards:
        cards = soup.select('div[class*="ProductCards_cards"]')
    out = []
    for c in cards:
        a = c.select_one('a[href*="/p/"]') or c.find("a", href=re.compile(r"/p/"))
        img = c.select_one('img[src*="image.torob.com"]')
        if not a or not img:
            continue
        url = urljoin(base, a["href"].split("?")[0])
        src = best_img(img.get("srcset", ""), img.get("src", ""))
        if "image.torob.com" not in src:
            continue
        out.append((urljoin(base, src), url, (img.get("alt") or "").strip()))
    seen, clean = set(), []
    for item in out:
        if item[1] not in seen:
            seen.add(item[1])
            clean.append(item)
    return clean


def valid_jpg(path: str) -> bool:
    try:
        if os.path.getsize(path) < 20_000:
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
    if "ربات" in pg.title():
        print("Bot check: click «من ربات نیستم» in the browser window, then press Enter.")
        try:
            input()
        except EOFError:
            pg.wait_for_timeout(30000)
        pg.wait_for_timeout(3000)


def scroll_collect(pg, base: str, max_items: int):
    found, still, prev = {}, 0, -1
    while len(found) < max_items and still < 4:
        pg.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        pg.wait_for_timeout(1800)
        for item in parse_cards(pg.content(), base):
            found.setdefault(item[1], item)
        still = still + 1 if len(found) == prev else 0
        prev = len(found)
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
    ap.add_argument("--categories", default="categories.txt")
    ap.add_argument("--max-items", type=int, default=100)
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--headless", action="store_true",
                    help="hide browser window (use after first trusted run)")
    ap.add_argument("--browser", default="auto", choices=["auto", "browser", "chromium"],
                    help="auto: installed Brave/Edge/Chrome else bundled Chromium")
    args = ap.parse_args()
    if not os.path.exists(args.categories):
        print(f"Missing {args.categories} — copy categories.example.txt first.")
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
    session.headers.update({"User-Agent": UA})
    total = 0
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            PROFILE, headless=args.headless,
            user_agent=UA, args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
            **resolve_browser(args.browser))
        pg = ctx.new_page()  # one page for all categories; never closed mid-run
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
