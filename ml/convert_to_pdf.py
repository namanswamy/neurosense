"""
Convert Markdown documentation files to styled PDFs
Uses: markdown + xhtml2pdf (pisa)
"""
import subprocess
import sys
import os

# Install dependencies
def install(pkg):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Installing PDF dependencies...")
for pkg in ['markdown', 'xhtml2pdf']:
    try:
        __import__(pkg.replace('-','_'))
    except ImportError:
        print(f"  Installing {pkg}...")
        install(pkg)

import markdown
from xhtml2pdf import pisa

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSS = """
    @page {
        size: A4;
        margin: 2cm 2.5cm 2.5cm 2.5cm;
    }

    body {
        font-family: Helvetica, Arial, sans-serif;
        font-size: 11px;
        line-height: 1.6;
        color: #1a1a2e;
    }

    h1 {
        font-size: 26px;
        color: #0f3460;
        border-bottom: 3px solid #6c63ff;
        padding-bottom: 10px;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    h2 {
        font-size: 20px;
        color: #16213e;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 6px;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    h3 {
        font-size: 15px;
        color: #533483;
        margin-top: 18px;
        margin-bottom: 8px;
    }

    p {
        margin-bottom: 8px;
        text-align: justify;
    }

    strong {
        color: #0f3460;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        margin: 12px 0;
        font-size: 10px;
    }

    th {
        background-color: #6c63ff;
        color: white;
        padding: 8px 10px;
        text-align: left;
        font-weight: bold;
    }

    td {
        padding: 6px 10px;
        border-bottom: 1px solid #e0e0e0;
    }

    code {
        font-family: Courier, monospace;
        font-size: 10px;
        background-color: #f4f4f8;
        padding: 1px 4px;
    }

    pre {
        background-color: #1a1a2e;
        color: #e0e0e0;
        padding: 12px 15px;
        font-size: 9.5px;
        line-height: 1.5;
        overflow: hidden;
        white-space: pre-wrap;
        word-wrap: break-word;
        margin: 10px 0;
    }

    pre code {
        background-color: transparent;
        color: inherit;
        padding: 0;
    }

    hr {
        border: none;
        border-top: 2px solid #e0e0e0;
        margin: 20px 0;
    }

    ul, ol {
        margin-bottom: 10px;
        padding-left: 25px;
    }

    li {
        margin-bottom: 4px;
    }

    blockquote {
        border-left: 4px solid #6c63ff;
        padding-left: 12px;
        margin: 10px 0;
        color: #555;
    }

    a {
        color: #6c63ff;
        text-decoration: none;
    }

    .footer {
        text-align: center;
        font-size: 8px;
        color: #999;
        margin-top: 40px;
        border-top: 1px solid #e0e0e0;
        padding-top: 8px;
    }
"""

def convert_md_to_pdf(md_file, pdf_file, footer_text):
    """Convert a markdown file to a styled PDF."""
    print(f"\nReading: {md_file}")
    with open(md_file, 'r', encoding='utf-8') as f:
        md_text = f.read()

    print("  Converting Markdown to HTML...")
    html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'codehilite', 'toc'])

    html = '<!DOCTYPE html><html><head><meta charset="utf-8"><style>'
    html += CSS
    html += '</style></head><body>'
    html += html_body
    html += f'<div class="footer">{footer_text}</div>'
    html += '</body></html>'

    print("  Generating PDF...")
    with open(pdf_file, 'wb') as pf:
        status = pisa.CreatePDF(html, dest=pf)

    if status.err:
        print(f"  ERROR: PDF generation failed with {status.err} errors")
        return False

    file_size = os.path.getsize(pdf_file) / 1024
    print(f"  Saved: {pdf_file}")
    print(f"  Size:  {file_size:.1f} KB")
    return True


# ── Convert both documents ──────────────────────────────────

docs = [
    {
        'md': os.path.join(BASE_DIR, 'ML_MODEL_DOCUMENTATION.pdf.md'),
        'pdf': os.path.join(BASE_DIR, 'ML_MODEL_DOCUMENTATION.pdf'),
        'footer': 'NeuroSense - ML Model Documentation | February 2026'
    },
    {
        'md': os.path.join(BASE_DIR, 'DATASET_DOCUMENTATION.md'),
        'pdf': os.path.join(BASE_DIR, 'DATASET_DOCUMENTATION.pdf'),
        'footer': 'NeuroSense - Dataset Documentation | February 2026'
    }
]

success = 0
for doc in docs:
    if os.path.exists(doc['md']):
        if convert_md_to_pdf(doc['md'], doc['pdf'], doc['footer']):
            success += 1
    else:
        print(f"\nSkipped (not found): {doc['md']}")

print(f"\n{'='*50}")
print(f"  Done! {success} PDF(s) generated.")
print(f"{'='*50}")
