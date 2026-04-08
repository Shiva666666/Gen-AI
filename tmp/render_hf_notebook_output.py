from pathlib import Path
import html
import json


BASE_DIR = Path(r"C:\Users\licha\OneDrive\Desktop\Gen-AI")
NOTEBOOK = BASE_DIR / "Huggingface.ipynb"
OUT_HTML = BASE_DIR / "tmp" / "huggingface_notebook_output.html"
OUT_HTML_OUTPUTS = BASE_DIR / "tmp" / "huggingface_notebook_outputs_only.html"


data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
cell = data["cells"][0]
source = "".join(cell.get("source", []))

streams = []
html_block = ""
for out in cell.get("outputs", []):
    if "text" in out:
        streams.append("".join(out["text"]))
    if "data" in out and "text/html" in out["data"]:
        value = out["data"]["text/html"]
        html_block = "".join(value) if isinstance(value, list) else str(value)

stream_text = "\n\n".join(s.strip() for s in streams if s.strip())

rendered = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Huggingface.ipynb Output</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      background: #111827;
      color: #f3f4f6;
      margin: 0;
      padding: 32px;
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
    }}
    .card {{
      background: #1f2937;
      border: 1px solid #374151;
      border-radius: 14px;
      padding: 20px;
      margin-bottom: 20px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }}
    h1, h2 {{
      margin-top: 0;
      color: #f9fafb;
    }}
    .muted {{
      color: #9ca3af;
      margin-bottom: 12px;
    }}
    pre {{
      white-space: pre-wrap;
      word-wrap: break-word;
      background: #0f172a;
      color: #e5e7eb;
      border-radius: 10px;
      padding: 16px;
      border: 1px solid #334155;
      overflow-x: auto;
      font-family: Consolas, monospace;
      font-size: 14px;
      line-height: 1.45;
    }}
    .pill {{
      display: inline-block;
      padding: 6px 10px;
      margin-right: 8px;
      background: #2563eb;
      border-radius: 999px;
      font-size: 12px;
    }}
    .frame {{
      background: #fff;
      border-radius: 10px;
      overflow: hidden;
      padding: 8px;
    }}
    iframe {{
      width: 100%;
      height: 500px;
      border: 0;
      background: #fff;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Lab 9 Notebook Evidence</h1>
      <div class="muted">Rendered from saved outputs inside Huggingface.ipynb</div>
      <span class="pill">Transformers</span>
      <span class="pill">Gradio</span>
      <span class="pill">Notebook Output</span>
    </div>

    <div class="card">
      <h2>Notebook Code Cell</h2>
      <pre>{html.escape(source)}</pre>
    </div>

    <div class="card">
      <h2>Saved Console / Runtime Output</h2>
      <pre>{html.escape(stream_text)}</pre>
    </div>

    <div class="card">
      <h2>Saved Interface Output</h2>
      <div class="muted">The notebook stored a Gradio iframe as HTML output.</div>
      <div class="frame">{html_block}</div>
    </div>
  </div>
</body>
</html>
"""

OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
OUT_HTML.write_text(rendered, encoding="utf-8")
rendered_outputs_only = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Huggingface.ipynb Outputs Only</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      background: #111827;
      color: #f3f4f6;
      margin: 0;
      padding: 32px;
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
    }}
    .card {{
      background: #1f2937;
      border: 1px solid #374151;
      border-radius: 14px;
      padding: 20px;
      margin-bottom: 20px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }}
    h1, h2 {{
      margin-top: 0;
      color: #f9fafb;
    }}
    .muted {{
      color: #9ca3af;
      margin-bottom: 12px;
    }}
    pre {{
      white-space: pre-wrap;
      word-wrap: break-word;
      background: #0f172a;
      color: #e5e7eb;
      border-radius: 10px;
      padding: 16px;
      border: 1px solid #334155;
      overflow-x: auto;
      font-family: Consolas, monospace;
      font-size: 14px;
      line-height: 1.45;
    }}
    .frame {{
      background: #fff;
      border-radius: 10px;
      overflow: hidden;
      padding: 8px;
    }}
    iframe {{
      width: 100%;
      height: 500px;
      border: 0;
      background: #fff;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Lab 9 Notebook Output Evidence</h1>
      <div class="muted">Saved runtime output and embedded interface output from Huggingface.ipynb</div>
    </div>
    <div class="card">
      <h2>Saved Console / Runtime Output</h2>
      <pre>{html.escape(stream_text)}</pre>
    </div>
    <div class="card">
      <h2>Saved Interface Output</h2>
      <div class="muted">The notebook stored a Gradio iframe as HTML output.</div>
      <div class="frame">{html_block}</div>
    </div>
  </div>
</body>
</html>
"""
OUT_HTML_OUTPUTS.write_text(rendered_outputs_only, encoding="utf-8")
print(OUT_HTML)
print(OUT_HTML_OUTPUTS)
