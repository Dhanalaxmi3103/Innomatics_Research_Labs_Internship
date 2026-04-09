import zipfile
import os
from services.webdev_service import generate_web_code

OUTPUT_DIR = "output"

def generate_webpage(article):
    response = generate_web_code(article)

    html = response.split('--html--')[1].strip()
    css = response.split('--css--')[1].strip()
    js = response.split('--js--')[1].strip()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(f"{OUTPUT_DIR}/index.html", "w") as f:
        f.write(html)

    with open(f"{OUTPUT_DIR}/style.css", "w") as f:
        f.write(css)

    with open(f"{OUTPUT_DIR}/script.js", "w") as f:
        f.write(js)

    zip_path = f"{OUTPUT_DIR}/website.zip"

    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(f"{OUTPUT_DIR}/index.html", "index.html")
        zipf.write(f"{OUTPUT_DIR}/style.css", "style.css")
        zipf.write(f"{OUTPUT_DIR}/script.js", "script.js")

    return html, zip_path