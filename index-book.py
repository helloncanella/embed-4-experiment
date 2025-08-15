import base64, json
from pdf2image import convert_from_path
from io import BytesIO

pdf_path = "strom.pdf"
pages = convert_from_path(pdf_path, dpi=200)

base64_pages = []
for page in pages:
    buf = BytesIO()
    page.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    base64_pages.append(b64)

with open("pages_base64.json", "w") as f:
    json.dump(base64_pages, f)
