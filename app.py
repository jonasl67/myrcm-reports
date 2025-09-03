from flask import Flask, request, render_template_string, url_for, Response
import subprocess
import tempfile
import pathlib
import uuid
import time
import shutil
import os
import logging

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
  <style>
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background-color: white;
      color: #333;
      text-align: center;
    }
    .header {
      width: 100%;
      background-color: #4CAF50; /* brighter green */
      text-align: center;
      padding: 20px 0;
    }
    .header img {
      max-height: 120px;
    }
    .title {
      font-size: 1.8em;
      margin: 20px 0;
      font-weight: bold;
      color: #4CAF50;
    }
    .form-box {
      display: inline-block;
      border: 2px solid #4CAF50;
      border-radius: 8px;
      padding: 20px;
      margin-top: 20px;
      max-width: 600px;
      background-color: #fafafa;
      text-align: left;
    }
    label {
      font-weight: bold;
      display: block;
      margin-top: 12px;
      margin-bottom: 4px;
    }
    input[type="text"] {
      width: 95%;
      padding: 8px;
      margin-bottom: 12px;
      border: 1px solid #ccc;
      border-radius: 4px;
      font-size: 1em;
    }
    button {
      background-color: #4CAF50;
      color: white;
      border: none;
      padding: 10px 16px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 1em;
    }
    button:hover {
      background-color: #388E3C;
    }
    h2 {
      text-align: left;
      color: #4CAF50;
      margin-top: 0;
    }
    .disclaimer {
      color: red;
      font-weight: bold;
      margin-bottom: 16px;
    }
    #loading {
      color: blue;
      font-weight: bold;
      margin-top: 10px;
    }
  </style>
</head>
<body>

  <div class="header">
    <img src="{{ url_for('static', filename='logo_small.png') }}" alt="Balthazar RC Logo">
  </div>

  <div class="title">Welcome to Balthazar RC's race reports page</div>

  <div class="form-box">
    <h2>Race Report – lapchart and statistics</h2>
    <p class="disclaimer">
      This is a free service that comes with no warranties!
    </p>
    <p>
      By providing the link (URL) to a <code>myrcm.ch</code> race page for the class you are interested in 
      and specifying which final you want the report for, you will receive a PDF report with a lap chart and statistics
      on time lost to fuel stops, events and incidents.
    </p>

    <form method="post" onsubmit="
      document.getElementById('loading').style.display='block';
      var err = document.getElementById('error');
      if (err) { err.innerHTML = ''; }
    ">

      <label>1. Paste the URL from <code>myrcm.ch</code> for the race and class:</label>
      <input type="text" name="url" size="60">

      <label>2. Enter the name of the final (for example <i>"Final A"</i>):</label>
      <input type="text" name="query" size="40">

      <button type="submit">Generate PDF</button>
    </form>

    <p id="loading" style="display:none;">⏳ Processing, please wait...</p>

    {% if error %}
      <p id="error" style="color:red; white-space:pre-wrap;">{{ error }}</p>
    {% else %}
      <p id="error"></p>
    {% endif %}
  </div>

</body>
</html>
"""

HTML_SUCCESS = """
<!DOCTYPE html>
<html>
<head>
  <title>Report Ready</title>
</head>
<body style="margin-left: 40px;">

  <!-- Green banner with logo -->
  <div style="background-color:#00c853; padding:20px; text-align:center;">
    <img src="{{ url_for('static', filename='logo_small.jpg') }}" alt="Logo" style="max-height:100px;">
  </div>

  <!-- Page title -->
  <h1 style="text-align:center; margin-top:10px;">
    Welcome to Balthazar RC's race reports page
  </h1>

  <div style="max-width:700px; margin:20px auto; border:2px solid #00c853; padding:20px; border-radius:8px; background:#fff;">
    <h2 style="color:#00c853; text-align:left;">Race Report - lapchart and statistics</h2>

    <h2>Your Report is Ready ✅</h2>
    <p>
      <a href="{{ download_url }}" download>
        <button type="button">⬇️ Download PDF</button>
      </a>
    </p>
    <p><a href="{{ home_url }}">⬅️ Generate another report</a></p>

    <!-- Collapsible process log -->
    <details style="margin-top:20px;">
      <summary style="cursor:pointer; font-weight:bold; color:#333;">Show process log</summary>
      <pre style="background:#f0f0f0; padding:10px; border:1px solid #ccc; white-space:pre-wrap; margin-top:10px;">
{{ logs }}
      </pre>
    </details>
  </div>
</body>
</html>
"""

# Create a dedicated logger for usage
usage_logger = logging.getLogger("usage")
usage_logger.setLevel(logging.INFO)

# Create a file handler for usage.log
fh = logging.FileHandler("usage.log")
fh.setLevel(logging.INFO)

# Define format
formatter = logging.Formatter("%(asctime)s - %(message)s")
fh.setFormatter(formatter)

# Add handler to logger (not to root)
usage_logger.addHandler(fh)

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

@app.before_request
def log_usage():
    if request.endpoint == "index" and request.method == "POST":
        user_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
        user_agent = request.headers.get("User-Agent", "unknown")
        submitted_url = request.form.get("url", "").strip()

        log_line = f"UserIP={user_ip} Agent={user_agent}"
        if submitted_url:
            log_line += f" URL={submitted_url}"

        usage_logger.info(log_line)


@app.route("/", methods=["GET", "POST"])
def index():
    cleanup_old_reports()

    if request.method == "POST":
        url = request.form.get("url", "").strip()
        query = request.form.get("query", "").strip()

        if not url or not query:
            return render_template_string(HTML_FORM, error="Both URL and report name are required.")

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

        # Look for PDF_FILE marker in stdout
        pdf_filename = None
        for line in result.stdout.splitlines():
            if line.startswith("PDF_FILE:"):
                pdf_filename = line.split("PDF_FILE:", 1)[1].strip()
                break

        if not pdf_filename:
            return render_template_string(
                HTML_FORM,
                error="No PDF was generated.\n\n" + logs,
            )

        pdf_path = run_dir / pdf_filename
        if not pdf_path.exists():
            return render_template_string(
                HTML_FORM,
                error="Generated PDF not found in run directory.\n\n" + logs,
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
        return f"File not found: {file_path}", 404

    try:
        with open(file_path, "rb") as f:
            data = f.read()
    except Exception as e:
        return f"Failed to read PDF file: {e}", 500

    response = Response(data, mimetype="application/pdf")

    # Force download with safe filename
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'

    # Explicit length is critical for Chrome
    response.headers["Content-Length"] = str(len(data))

    # Conservative caching headers
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response


    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))  # 5050 for local dev
    app.run(host="0.0.0.0", port=port)
