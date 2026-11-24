"""Render framework.html to ../framework.png with headless Chromium at 2x scale.

Needs Playwright (`pip install playwright && playwright install chromium`).
Usage: python docs/figures/src/render.py
"""

from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1640, "height": 1200}, device_scale_factor=2)
    page.goto((HERE / "framework.html").as_uri())
    page.wait_for_function("window.__ready === true", timeout=20000)
    page.locator("#canvas").screenshot(path=HERE.parent / "framework.png")
    browser.close()
