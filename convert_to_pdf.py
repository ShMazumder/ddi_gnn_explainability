import asyncio
from playwright.async_api import async_playwright
import os

ROOT = "/Applications/XAMPP/xamppfiles/htdocs/Research"
FILES = [
    ("raw/ddi_thesis_abstract.html", "docs/ddi_thesis_abstract.pdf"),
    ("raw/masters_thesis_ddi_gnn_explainability.html", "docs/masters_thesis_ddi_gnn_explainability.pdf"),
    ("raw/analysis.html", "docs/analysis.pdf"),
]

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for src, dst in FILES:
            src_path = os.path.join(ROOT, src)
            dst_path = os.path.join(ROOT, dst)
            page = await browser.new_page()
            await page.goto(f"file://{src_path}", wait_until="networkidle")
            await page.pdf(path=dst_path, format="A4", print_background=True)
            await page.close()
            print(f"Created {dst_path}")
        await browser.close()

asyncio.run(main())
