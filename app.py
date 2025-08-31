from flask import Flask, request, send_file, render_template_string, url_for
import subprocess
import tempfile
import pathlib
import uuid
import time
import shutil

app = Flask(__name__)

BASE_DIR = pathlib.Path(__file__).parent.resolve()
OUTPUT_DIR = pathlib.Path(tempfile.gettempdir()) / "myrcm_reports"
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_AGE_SECONDS = 24 * 3600  # keep reports for 24 hours

HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
  <title>RC Race Report Generator</title>
</head>
<body>
  <h2>RC Race Report Generator</h2>
  <p style="color:darkred; font-weight:bold;">
    This is a free service that comes with no warranties!
  </p>
  <p>
    By providing the link (URL) to a <code>myrcm.ch</code> race page and specifying which final you
    are interested in, you will receive a PDF report with a lap chart and other statistics
    for the race/final.
  </p>
  <p>
    <b>Instructions:</b><br>
    1. Find the <code>myrcm.ch</code> page for the race and class you are interested in,
       copy the address (URL) of the page and paste it into the first field below.<br>
    2. In the second field, write the name of the race you want the report for,
       for example <i>"Final A"</i>.
  </p>

  <form method="post">
    Race URL: <input type="text" name="url" size="60"><br><br>
    Report (e.g. "Final A run 2"): <input type="text" name="query" size="40"><br><br>
    <button type="submit">Generate PDF</button>
  </form>

  {% if error %}
    <p style="color:red; white-space:pre-wrap;">{{ error }}</p>
  {% endif %}
</body>
</html>
"""

HTML_SUCCESS = """
<!DOCTYPE html>
<html>
<head>
  <title>Report Ready</title>
</head>
<body>
  <h2>Your Report is Ready ✅</h2>
  <p>You can download it here:</p>
  <p><a href="{{ download_url }}">{{ filename }}</a></p>
  <p><a href="{{ home_url }}">⬅️ Generate another report</a></p>

  <h3>Process Log</h3>
  <pre style="background:#f0f0f0; padding:10px; border:1px solid #ccc; white-space:pre-wrap;">
{{ logs }}
  </pre>
</body>
</html>
"""

def cleanup_old_reports():
    """Delete report folders older than MAX_AGE_SECONDS."""
    now = time.time()
    for subdir in OUTPUT_DIR.iterdir():
        if not subdir.is_dir():
            continue
        try:
            mtime = subdir.stat().st_mtime
            if now - mtime > MAX_AGE_SECONDS:
                shutil.rmtree(subdir)
        except Exception:
            pass  # ignore cleanup errors

@app.route("/", methods=["GET", "POST"])
def index():
    cleanup_old_reports()

    if request.method == "POST":
        url = request.form.get("url", "").strip()
        query = request.form.get("query", "").strip()

        if not url or not query:
            return render_template_string(HTML_FORM, error="Both URL and report name are required.")

        # Each run gets its own unique subdir in OUTPUT_DIR
        run_id = str(uuid.uuid4())
        run_dir = OUTPUT_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        runner_path = BASE_DIR / "myrcm_runner.py"
        cmd = ["python", str(runner_path), url] + query.split()
        result = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True)

        logs = (result.stdout or "") + "\n" + (result.stderr or "")

        if result.returncode != 0:
            return render_template_string(
                HTML_FORM,
                error="Scraper/Stats failed:\n" + logs,
            )

        # Parse PDF_FILE marker from runner output
        pdf_path = None
        for line in result.stdout.splitlines():
            if line.startswith("PDF_FILE:"):
                pdf_path = pathlib.Path(line.split("PDF_FILE:")[1].strip())
                break

        if not pdf_path or not pdf_path.exists():
            return render_template_string(
                HTML_FORM,
                error="No PDF was generated.\n\n" + logs,
            )

        download_url = url_for("download_file", run_id=run_id, filename=pdf_path.name)

        return render_template_string(
            HTML_SUCCESS,
            download_url=download_url,
            filename=pdf_path.name,
            home_url=url_for("index"),
            logs=logs.strip(),
        )

    return render_template_string(HTML_FORM, error=None)

@app.route("/download/<run_id>/<filename>")
def download_file(run_id, filename):
    file_path = OUTPUT_DIR / run_id / filename
    if not file_path.exists():
        return "File not found", 404
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
