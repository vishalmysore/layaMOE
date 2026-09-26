"""Capture documentation screenshots from the live GitHub Pages demo (headless Chromium).

    python scripts/screenshots.py                       # https://vishalmysore.github.io/layaMOE/ -> docs/images/
    python scripts/screenshots.py --url http://localhost:8765/

Loads the model (~490 MB from Hugging Face, cached by the browser profile only for this run),
runs a few presets with "compare with the general head" on, and saves element screenshots.
"""
import argparse, os
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SHOTS = [  # (file, preset)
    ("guardrail-drop-table.png", "Agent guardrail: drop a production table"),
    ("guardrail-dry-run.png", "Agent guardrail: dry run"),
    ("support-ticket.png", "Support ticket"),
    ("delivery-exception.png", "Delivery exception"),
    ("out-of-domain-sales.png", "Out of domain: sales lead (general head)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://vishalmysore.github.io/layaMOE/")
    ap.add_argument("--out", default=str(ROOT / "docs" / "images"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge")  # installed Edge; avoids a Playwright browser download
        pg = b.new_page(viewport={"width": 1100, "height": 900}, device_scale_factor=2)
        pg.goto(args.url)
        pg.wait_for_function("document.getElementById('status').textContent.includes('Not loaded')", timeout=60000)
        pg.wait_for_timeout(1500)  # the service worker may reload once for cross-origin isolation
        pg.wait_for_function("document.getElementById('status').textContent.includes('Not loaded')", timeout=60000)
        pg.screenshot(path=str(out / "page-start.png"))
        pg.click("#loadBtn")
        pg.wait_for_function("window.__moe && window.__moe.ready", timeout=15 * 60 * 1000)
        print("loaded:", pg.inner_text("#status"))
        pg.locator("section.card").first.screenshot(path=str(out / "model-loaded.png"))
        for fname, preset in SHOTS:
            pg.select_option("#preset", preset)
            pg.evaluate("window.__moe.run()")
            pg.wait_for_function("document.getElementById('status').textContent === 'Done.'", timeout=120000)
            pg.locator("#resultCard").screenshot(path=str(out / fname))
            print(fname, "->", pg.inner_text("#sExpert"))
            pg.evaluate("document.getElementById('status').textContent = ''")
        b.close()


if __name__ == "__main__":
    main()
