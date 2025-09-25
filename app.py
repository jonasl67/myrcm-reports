from flask import Flask, request, render_template_string, url_for, Response, jsonify
import requests
from requests.exceptions import Timeout, RequestException
from bs4 import BeautifulSoup
import subprocess
import tempfile
import pathlib
import uuid
import time
import shutil
import os
import logging
import re
import urllib.parse
import csv
import io
from datetime import datetime, timezone

TIMEOUT_SECONDS = 5  # timeout for myrcm.ch responses

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
      background-color: #4CAF50;
      text-align: center;
      padding: 10px 0;
    }
    .header img {
      max-height: 80px;
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
    input[type="text"], select {
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
    button:disabled {
      background-color: #9E9E9E;
      cursor: not-allowed;
    }
    button:hover:enabled {
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
        margin-bottom: 8px;
    }
    .note {
        color: #666; /* grey text */
        margin-top: 0;
        margin-bottom: 16px;
    }
    .blinking {
      color: blue;
      font-weight: bold;
      margin-top: 6px;
      animation: blink 1s infinite;
    }
    @keyframes blink {
      50% { opacity: 0; }
    }
    .search-row {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .search-row input {
      flex: 1;
    }
  </style>
</head>
<body>

  <div class="header">
    <img src="{{ url_for('static', filename='logo_small.png') }}" alt="Balthazar RC Logo">
  </div>

  <div class="title">Welcome to Balthazar RC's nitro onroad race reports page</div>

  <div class="form-box">
    <h2>Race Report – lapchart and statistics</h2>

    <p class="disclaimer">
        This is a free service that comes with no warranties!
    </p>
    <p class="note">
        Only onroad nitro 1/8 and 1/10 races are supported at this time, using the service for any other class will render incorrect data.
    </p>

    <label>1. Apply your events search filter:</label>
    <div class="search-row">
      <input type="text" id="eventSearch" placeholder="Type at least 3 characters and press Enter or Search">
      <button type="button" id="searchBtn" disabled>Search</button>
    </div>
    <p id="searching" class="blinking" style="display:none;">🔎 Searching...</p>

    <label>2. Select your event:</label>
    <select id="eventSelect"></select>

    <label>3. Select a class/section:</label>
    <select id="classSelect"></select>

    <label>4. Select a final:</label>
    <select id="finalSelect"></select>

    <button type="button" id="generateBtn" disabled>Generate PDF</button>
    <p id="working" class="blinking" style="display:none;">⚡ Working...</p>

    {% if error %}
        <p id="error" style="color:red; white-space:pre-wrap;">{{ error }}</p>
    {% else %}
        <p id="error" style="color:red; white-space:pre-wrap;"></p>
    {% endif %}

  </div>

<script>
(function () {
  document.addEventListener("DOMContentLoaded", () => {
    const eventSearch = document.getElementById("eventSearch");
    const searchBtn   = document.getElementById("searchBtn");
    const eventSelect = document.getElementById("eventSelect");
    const classSelect = document.getElementById("classSelect");
    const finalSelect = document.getElementById("finalSelect");
    const generateBtn = document.getElementById("generateBtn");
    const searchingMsg= document.getElementById("searching");
    const workingMsg  = document.getElementById("working");
    const errorBox    = document.getElementById("error") || { innerText: "" };

    function showError(msg) {
      console.error(msg);
      if (errorBox) errorBox.innerText = msg;
    }

    function resetDropdowns() {
      eventSelect.innerHTML = "";
      classSelect.innerHTML = "";
      finalSelect.innerHTML = "";
      generateBtn.disabled = true;
    }

    // --- EVENTS SEARCH ---
    async function fetchEvents(q) {
      try {
        searchingMsg.style.display = "block";
        const resp = await fetch("/search_events?q=" + encodeURIComponent(q));
        searchingMsg.style.display = "none";

        let data = null;
        try { data = await resp.json(); } catch (_) {}

        if (!resp.ok) {
          if (data && data.error) {
            showError(data.error);
          } else {
            showError("Event search failed: " + resp.status);
          }
          return;
        }

        if (data.error) {
          showError(data.error);
          eventSelect.innerHTML = "";
          const opt = document.createElement("option");
          opt.value = "";
          opt.textContent = "(no matches)";
          eventSelect.appendChild(opt);
          return;
        }

        eventSelect.innerHTML = "";
        classSelect.innerHTML = "";
        finalSelect.innerHTML = "";
        generateBtn.disabled = true;

        if (!Array.isArray(data) || data.length === 0) {
          const opt = document.createElement("option");
          opt.value = "";
          opt.textContent = "(no matches)";
          eventSelect.appendChild(opt);
          return;
        }

        data.forEach(ev => {
          const opt = document.createElement("option");
          opt.value = ev.url;
          opt.textContent = ev.name;
          eventSelect.appendChild(opt);
        });
        eventSelect.selectedIndex = 0;
        eventSelect.dispatchEvent(new Event("change"));
      } catch (e) {
        searchingMsg.style.display = "none";
        showError("Event search exception: " + e);
      }
    }

    // Enable Search button only if ≥3 chars
    eventSearch.addEventListener("input", () => {
      searchBtn.disabled = eventSearch.value.trim().length < 3;
      saveState();
    });

    // Trigger search on Enter
    eventSearch.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !searchBtn.disabled) {
        e.preventDefault();
        fetchEvents(eventSearch.value.trim());
      }
    });

    // Trigger search on button click
    searchBtn.addEventListener("click", () => {
      if (!searchBtn.disabled) {
        fetchEvents(eventSearch.value.trim());
      }
    });

    // --- LOAD CLASSES ---
    eventSelect.addEventListener("change", async function () {
      if (!this.value) {
        classSelect.innerHTML = "";
        finalSelect.innerHTML = "";
        generateBtn.disabled = true;
        return;
      }
      try {
        const resp = await fetch("/get_classes?url=" + encodeURIComponent(this.value));
        let data = null;
        try { data = await resp.json(); } catch (_) {}

        if (!resp.ok) {
          if (data && data.error) {
            showError(data.error);
          } else {
            showError("Get classes failed: " + resp.status);
          }
          return;
        }

        if (data.error) {
          showError(data.error);
          classSelect.innerHTML = "";
          finalSelect.innerHTML = "";
          generateBtn.disabled = true;
          return;
        }

        classSelect.innerHTML = "";
        finalSelect.innerHTML = "";
        generateBtn.disabled = true;

        data.forEach(cl => {
          const opt = document.createElement("option");
          opt.value = cl.url;
          opt.textContent = cl.name;
          classSelect.appendChild(opt);
        });
        if (classSelect.options.length > 0) {
          classSelect.selectedIndex = 0;
          classSelect.dispatchEvent(new Event("change"));
        }
      } catch (e) {
        showError("Get classes exception: " + e);
      }
    });

    // --- LOAD FINALS ---
    classSelect.addEventListener("change", async function () {
      if (!this.value) {
        finalSelect.innerHTML = "";
        generateBtn.disabled = true;
        return;
      }
      try {
        const resp = await fetch("/get_finals?url=" + encodeURIComponent(this.value));
        let data = null;
        try { data = await resp.json(); } catch (_) {}

        if (!resp.ok) {
          if (data && data.error) {
            showError(data.error);
          } else {
            showError("Get finals failed: " + resp.status);
          }
          return;
        }

        if (data.error) {
          showError(data.error);
          finalSelect.innerHTML = "";
          generateBtn.disabled = true;
          return;
        }

        finalSelect.innerHTML = "";
        generateBtn.disabled = true;

        data.forEach(fn => {
          const opt = document.createElement("option");
          opt.value = fn.name;
          opt.textContent = fn.name;
          finalSelect.appendChild(opt);
        });
        if (finalSelect.options.length > 0) {
          finalSelect.selectedIndex = 0;
          generateBtn.disabled = false;
        }
      } catch (e) {
        showError("Get finals exception: " + e);
      }
    });

    // --- GENERATE BUTTON ---
    generateBtn.addEventListener("click", function () {
    const classUrl = classSelect.value;
    const finalName = finalSelect.value;
    const eventName = eventSelect.options[eventSelect.selectedIndex]?.text || "";
    const className = classSelect.options[classSelect.selectedIndex]?.text || "";

    if (!classUrl || !finalName) {
        showError("Please choose an event, class, and final.");
        return;
    }

    workingMsg.style.display = "block";
    const form = document.createElement("form");
    form.method = "post";
    form.action = "/";

    const u = document.createElement("input");
    u.type = "hidden"; u.name = "url"; u.value = classUrl;

    const q = document.createElement("input");
    q.type = "hidden"; q.name = "query"; q.value = finalName;

    const e = document.createElement("input");
    e.type = "hidden"; e.name = "event_name"; e.value = eventName;

    const c = document.createElement("input");
    c.type = "hidden"; c.name = "class_name"; c.value = className;

    const f = document.createElement("input");
    f.type = "hidden"; f.name = "final_name"; f.value = finalName;

    form.appendChild(u);
    form.appendChild(q);
    form.appendChild(e);
    form.appendChild(c);
    form.appendChild(f);

    document.body.appendChild(form);
    form.submit();
    });


    // --- STATE PERSISTENCE (only search string) ---
    function saveState() {
      localStorage.setItem("reportFormSearch", eventSearch.value);
    }
    function loadState() {
      const saved = localStorage.getItem("reportFormSearch");
      if (saved) {
        eventSearch.value = saved;
        searchBtn.disabled = saved.trim().length < 3;
      }
    }
    loadState();
  });
})();
</script>
  

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
    <img src="{{ url_for('static', filename='logo_small.png') }}" alt="Logo" style="max-height:100px;">
  </div>

  <div style="max-width:700px; margin:20px auto; border:2px solid #00c853; padding:20px; border-radius:8px; background:#fff;">
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

log_path = pathlib.Path("usage.csv")
is_new_file = not log_path.exists()

# Create a dedicated logger for usage
usage_logger = logging.getLogger("usage")
usage_logger.setLevel(logging.INFO)

# Create a file handler for usage.csv in append mode
fh = logging.FileHandler(log_path, mode="a")
fh.setLevel(logging.INFO)

# Only log the message itself (we'll build CSV lines manually)
formatter = logging.Formatter("%(message)s")
fh.setFormatter(formatter)

usage_logger.addHandler(fh)

# If file is new, write header
if is_new_file:
    with open(log_path, "a") as f:
        f.write("timestamp,user_ip,country,url,event,class,final\n")

def csv_escape(*fields):
    """Return a CSV-safe line with all fields quoted."""
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
    writer.writerow(fields)
    return buf.getvalue().strip()  # strip newline

def shorten_url(url: str) -> str:
    """Extract eventId/sectionId from a myrcm report URL."""
    m = re.search(r"/report/en/(\d+)/(\d+)", url)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return url

def get_country_from_ip(ip: str) -> str:
    """Resolve IP → country using ip-api.com (free service)."""
    try:
        resp = requests.get(f"http://ip-api.com/json/{ip}", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                return data.get("country", "Unknown")
    except Exception as e:
        print(f"⚠️ Geo lookup failed for {ip}: {e}")
    return "Unknown"

def get_client_info():
    """Return a (user_ip, user_agent) tuple for logging."""
    user_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    user_agent = request.headers.get("User-Agent", "unknown")
    return user_ip, user_agent

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

@app.route("/search_events")
def search_events():
    query = request.args.get("q", "").strip()
    if not query or len(query) < 3:
        return jsonify([])

    user_ip, ua = get_client_info()
    events = []

    # --- Ongoing events ---
    try:
        ongoing_url = f"https://myrcm.ch/myrcm/main?hId[1]=evt&pLa=en&dFi={urllib.parse.quote(query)}&hId[1]=search"
        #print(f"🔎 Fetching ongoing events: {ongoing_url}")
        resp = requests.get(ongoing_url, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        rows = soup.select("table tr")
        #print(f"🔍 Ongoing events → found {len(rows)} table rows")

        for row in rows:
            cols = [c.get_text(strip=True) for c in row.find_all("td")]
            links = row.find_all("a", href=True)
            if not cols or not links:
                continue

            event_name = cols[2] if len(cols) > 2 else "Unknown Event"
            host = cols[1] if len(cols) > 1 else "Unknown Host"

            event_id = None
            for a in links:
                m = re.search(r"dId\[E\]=(\d+)", a["href"])
                if m:
                    event_id = m.group(1)
                    break
            if not event_id:
                continue

            event_url = f"https://myrcm.ch/myrcm/main?dId[E]={event_id}"
            #print(f"  ⚡ Ongoing Event: {event_name} ({host}) → {event_url}")
            events.append({"name": f"⚡ {event_name} ({host})", "url": event_url})
    
    except Timeout:
        msg = f"Timeout contacting myrcm.ch (query='{query}')"
        print(f"⏰ {msg}")
        #usage_logger.info(f"UserIP={user_ip} Agent={ua} {msg}")
        usage_logger.info(f"UserIP={user_ip} {msg}")
        return jsonify({"error": "Getting no response from myrcm.ch, please try again in a while"}), 504
    
    except RequestException as e:
        msg = f"Request error contacting myrcm.ch (query='{query}'): {e}"
        print(f"❌ {msg}")
        #usage_logger.info(f"UserIP={user_ip} Agent={ua} {msg}")
        usage_logger.info(f"UserIP={user_ip} {msg}")
        return jsonify({"error": f"Error contacting myrcm.ch: {e}"}), 502

    # --- Archived events ---
    try:
        archived_url = f"https://myrcm.ch/myrcm/main?pLa=en&dFi={urllib.parse.quote(query)}&hId[1]=search"
        #print(f"🗄️ Fetching archived events: {archived_url}")
        resp = requests.get(archived_url, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        rows = soup.select("table tr")
        #print(f"🔍 Archived events → found {len(rows)} table rows")

        for row in rows:
            cols = [c.get_text(strip=True) for c in row.find_all("td")]
            links = row.find_all("a", href=True)
            if not cols or not links:
                continue

            event_name = cols[2] if len(cols) > 2 else "Unknown Event"
            host = cols[1] if len(cols) > 1 else "Unknown Host"

            event_id = None
            for a in links:
                m = re.search(r"dId\[E\]=(\d+)", a["href"])
                if m:
                    event_id = m.group(1)
                    break
            if not event_id:
                continue

            event_url = f"https://myrcm.ch/myrcm/main?dId[E]={event_id}"
            #print(f"  🗄️ Archived Event: {event_name} ({host}) → {event_url}")
            events.append({"name": f"🗄️ {event_name} ({host})", "url": event_url})

    except Timeout:
        msg = f"Timeout contacting myrcm.ch (query='{query}')"
        print(f"⏰ {msg}")
        usage_logger.info(f"UserIP={user_ip} {msg}")
        return jsonify({"error": "Getting no response from myrcm.ch, please try again in a while"}), 504
    
    except RequestException as e:
        msg = f"Request error contacting myrcm.ch (query='{query}'): {e}"
        print(f"❌ {msg}")
        usage_logger.info(f"UserIP={user_ip} {msg}")
        return jsonify({"error": f"Error contacting myrcm.ch: {e}"}), 502

    print(f"📊 Extracted {len(events)} total events")
    return jsonify(events)

@app.route("/get_classes")
def get_classes():
    event_url = request.args.get("url", "").strip()
    if not event_url:
        return jsonify([])

    user_ip, ua = get_client_info()
    
    try:
        #print(f"🔎 Fetching classes from: {event_url}")
        resp = requests.get(event_url, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        classes = []
        for a in soup.select("a[onclick*='openNewWindows']"):
            onclick_attr = a.get("onclick", "")
            match = re.search(r"openNewWindows\((\d+),\s*(\d+)\)", onclick_attr)
            if not match:
                continue
            event_id, section_id = match.groups()
            class_name = a.get_text(strip=True)
            class_url = f"https://myrcm.ch/myrcm/report/en/{event_id}/{section_id}"

            #print(f"  📌 Class: {class_name} → {class_url}")
            classes.append({
                "name": class_name,
                "url": class_url
            })

        print(f"📊 Extracted {len(classes)} classes")
        return jsonify(classes)
    
    except Timeout:
        msg = f"Timeout fetching classes (url={event_url})"
        print(f"⏰ {msg}")
        usage_logger.info(f"UserIP={user_ip} {msg}")
        return jsonify({"error": "Getting no response from myrcm.ch, please try again in a while"}), 504
    
    except RequestException as e:
        msg = f"Request error fetching classes (url={event_url}): {e}"
        print(f"❌ {msg}")
        usage_logger.info(f"UserIP={user_ip} {msg}")
        return jsonify({"error": f"Error contacting myrcm.ch: {e}"}), 502

@app.route("/get_finals")
def get_finals():
    class_url = request.args.get("url", "").strip()
    if not class_url:
        return jsonify([])

    user_ip, ua = get_client_info()

    try:
        #print(f"🔎 Fetching finals from: {class_url}")
        resp = requests.get(class_url, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        finals = []
        in_final_section = False

        content_div = soup.select_one("div.l-content") or soup

        for elem in content_div.find_all(["h2", "h3", "h4", "div", "a"]):
            text = elem.get_text(strip=True).lower()

            if elem.name in ("h2", "h3", "h4", "div") and text:
                if text.startswith("final"):
                    in_final_section = True
                    #print(f"✅ Entering FINAL section: '{text}'")
                    continue
                elif in_final_section:
                    #print(f"⛔ Leaving FINAL section at: '{text}'")
                    break

            if elem.name == "a" and in_final_section and elem.has_attr("onclick"):
                m = re.search(r"doAjaxCall\('([^']*)',\s*'([^']*)'\)", elem["onclick"])
                if m:
                    raw_name = m.group(2).strip()

                    # Strip "Final :: " if present
                    pretty_name = re.sub(r"^Final\s*::\s*", "", raw_name, flags=re.IGNORECASE)

                    if not any(x in raw_name.lower() for x in ["practice", "qualif", "ranking", "timeschedule", "participants"]):
                        finals.append({
                            "name": pretty_name,
                            "onclick": m.group(1),
                        })
                        #print(f"   ✅ Accepted final: {pretty_name}")
                    else:
                        #print(f"   ❌ Rejected (non-final): {raw_name}")
                        pass

        print(f"📊 Extracted {len(finals)} finals")
        return jsonify(finals)

    except Timeout:
        msg = f"Timeout fetching finals (url={class_url})"
        print(f"⏰ {msg}")
        usage_logger.info(f"UserIP={user_ip}{msg}")
        return jsonify({"error": "Getting no response from myrcm.ch, please try again in a while"}), 504
    
    except RequestException as e:
        msg = f"Request error fetching finals (url={class_url}): {e}"
        print(f"❌ {msg}")
        usage_logger.info(f"UserIP={user_ip} {msg}")
        return jsonify({"error": f"Error contacting myrcm.ch: {e}"}), 502

@app.before_request
def log_usage():
    if request.endpoint == "index" and request.method == "POST":
        user_ip, _ = get_client_info()
        submitted_url = request.form.get("url", "").strip()
        event_name = request.form.get("event_name", "").strip()
        class_name = request.form.get("class_name", "").strip()
        final_name = request.form.get("final_name", "").strip()

        # ISO timestamp with timezone-aware UTC
        timestamp = datetime.now(timezone.utc).isoformat()

        # Resolve country immediately
        country = get_country_from_ip(user_ip)

        log_line = csv_escape(
            timestamp, user_ip, country,
            submitted_url, event_name, class_name, final_name
        )

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
        cmd = ["python", str(runner_path), url, query]

        # 🔎 Debug command string
        cmd_str = " ".join(cmd)
        print("⚡ Running runner with command:", cmd_str)

        result = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True)

        logs = (result.stdout or "") + "\n" + (result.stderr or "")

        if result.returncode != 0:
            return render_template_string(
                HTML_FORM,
                error=f"Scraper/Stats failed:\n⚡ Command: {cmd_str}\n\n{logs}",
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
                error=f"No PDF was generated.\n⚡ Command: {cmd_str}\n\n{logs}",
            )

        pdf_path = run_dir / pdf_filename
        if not pdf_path.exists():
            return render_template_string(
                HTML_FORM,
                error=f"Generated PDF not found.\n⚡ Command: {cmd_str}\n\n{logs}",
            )

        download_url = url_for("download_file", run_id=run_id, filename=pdf_path.name)

        return render_template_string(
            HTML_SUCCESS,
            download_url=download_url,
            filename=pdf_path.name,
            home_url=url_for("index"),
            logs=f"⚡ Command: {cmd_str}\n\n{logs}".strip(),
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
