from flask import Flask, request, send_file, render_template_string
import subprocess
import tempfile
import os
import glob

app = Flask(__name__)

HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
  <title>RC Race Report Generator</title>
</head>
<body>
  <h2>RC Race Report Generator</h2>
  <form method="post">
    Race URL: <input type="text" name="url" size="60"><br><br>
    Report (e.g. "Final A run 2"): <input type="text" name="query" size="40"><br><br>
    <button type="submit">Generate PDF</button>
  </form>
  {% if error %}
    <p style="color:red;">{{ error }}</p>
  {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        query = request.form.get("query", "").strip()

        if not url or not query:
            return render_template_string(HTML_FORM, error="Both URL and report name are required.")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run your existing runner script inside a temp directory
            cmd = ["python", "myrcm_runner.py", url] + query.split()
            result = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True)

            if result.returncode != 0:
                return render_template_string(HTML_FORM, error="Scraper/Stats failed:\n" + result.stdout + result.stderr)

            # Find the generated PDF in tmpdir
            pdfs = glob.glob(os.path.join(tmpdir, "*.pdf"))
            if not pdfs:
                return render_template_string(HTML_FORM, error="No PDF was generated.")

            return send_file(pdfs[0], as_attachment=True)

    return render_template_string(HTML_FORM, error=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
