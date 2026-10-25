import subprocess
from pathlib import Path
import mistune

CSS_STYLE = """
@page {
    size: A4;
    margin: 20mm;
}
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #2d3748;
    line-height: 1.6;
    font-size: 11pt;
    margin: 0;
    padding: 0;
}
h1 {
    font-size: 20pt;
    color: #1a202c;
    border-bottom: 2px solid #3182ce;
    padding-bottom: 8px;
    margin-top: 0;
    margin-bottom: 12px;
}
h2 {
    font-size: 15pt;
    color: #2b6cb0;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 4px;
    margin-top: 24px;
    margin-bottom: 12px;
}
h3 {
    font-size: 12pt;
    color: #2d3748;
    margin-top: 18px;
    margin-bottom: 8px;
}
h4 {
    font-size: 11pt;
    color: #4a5568;
    margin-top: 14px;
    margin-bottom: 6px;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 10pt;
}
th, td {
    border: 1px solid #cbd5e0;
    padding: 6px 10px;
    text-align: left;
}
th {
    background-color: #edf2f7;
    color: #1a202c;
    font-weight: 600;
}
tr:nth-child(even) {
    background-color: #f7fafc;
}
pre {
    background-color: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 10px 14px;
    font-family: "Courier New", Courier, monospace;
    font-size: 9.5pt;
    line-height: 1.4;
    overflow-x: auto;
    margin: 12px 0;
}
code {
    font-family: "Courier New", Courier, monospace;
    background-color: #edf2f7;
    padding: 2px 4px;
    border-radius: 3px;
    font-size: 9.5pt;
}
pre code {
    background-color: transparent;
    padding: 0;
}
blockquote {
    border-left: 4px solid #3182ce;
    margin: 12px 0;
    padding: 6px 16px;
    color: #4a5568;
    background-color: #ebf8ff;
}
hr {
    border: 0;
    border-top: 1px solid #e2e8f0;
    margin: 20px 0;
}
"""


def convert_md_to_pdf(md_file: str, pdf_file: str):
    md_path = Path(md_file)
    pdf_path = Path(pdf_file)

    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file {md_file} not found.")

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    markdown_renderer = mistune.create_markdown(plugins=["table"])
    html_content = markdown_renderer(md_text)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
{CSS_STYLE}
</style>
</head>
<body>
{html_content}
</body>
</html>
"""

    temp_html = md_path.with_suffix(".temp.html")
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(full_html)

    try:
        cmd = [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            str(temp_html),
            "--outdir",
            str(pdf_path.parent),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        generated_pdf = temp_html.with_suffix(".pdf")
        if generated_pdf != pdf_path and generated_pdf.exists():
            generated_pdf.rename(pdf_path)

        print(f"Successfully generated PDF: {pdf_file}")
    finally:
        if temp_html.exists():
            temp_html.unlink()


if __name__ == "__main__":
    import sys

    target_md = sys.argv[1] if len(sys.argv) > 1 else "report.md"
    target_pdf = sys.argv[2] if len(sys.argv) > 2 else "report.pdf"
    convert_md_to_pdf(target_md, target_pdf)
