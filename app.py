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
import re
import urllib.parse
import csv
import io
from datetime import datetime, timezone
import socket
from requests.exceptions import Timeout, RequestException, SSLError, ConnectionError
import json
import atexit

# Google sheet for logging imports
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from concurrent.futures import ThreadPoolExecutor
from google.oauth2 import service_account

TIMEOUT_SECONDS = 7  # timeout for myrcm.ch responses

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
        margin-bottom: 24px;
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

    /* alignment + equal height fix for search input + button */
    .search-row input[type="text"] {
    margin: 0;
    height: 40px;       /* fixed height for input */
    box-sizing: border-box;
    }
    .search-row button {
    margin: 0;
    height: 40px;       /* same height as input */
    box-sizing: border-box;
}
</style>

</head>
  <meta name="Select a RC car race timd and published on myrcm.ch and get a lapchart / graph and more race statistics for the race">

<body>

  <div class="header">
    <img src="{{ url_for('static', filename='logo_small.png') }}" alt="Balthazar RC Logo">
  </div>

  <div class="title">Welcome to Balthazar RC's race reports page</div>

  <div class="form-box">
    <h2>myrcm.ch lapchart and statistics generator</h2>

    <p class="disclaimer">
        This is a free service that comes with no warranties!
    </p>
    
    <label>1. Apply your search filter for races published on myrcm.ch:</label>
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

    function clearError() {
        if (errorBox) errorBox.innerText = "";
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
        showError("Event search exception (you may be behind a corporate web firewall/filtering/blocking service): " + e);
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
        clearError();   // <-- clear error before new search
        fetchEvents(eventSearch.value.trim());
      }
    });

    // Trigger search on button click
    searchBtn.addEventListener("click", () => {
      if (!searchBtn.disabled) {
        clearError();   // <-- clear error before starting a new search
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
  
  <div style="margin-top: 40px; text-align: center; font-size: 0.9em; color: #555;">
    If you have any pressing questions, the occasional rant,<br>or perhaps just a word of reluctant gratitude... <br>please send them over to <a href="mailto:info@balthazarrc.com">info@balthazarrc.com</a>
  </div>
</body>
</html>
"""

HTML_SUCCESS = """
<!DOCTYPE html>
<html>
<head>
  <title>Report Ready</title>
  <meta name="Download the lapchart for the selected final of the RC car race published and timed by myrcm.ch">
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
    
    <!--
    <div style="margin-top: 40px; text-align: center;">
      <p style="margin-bottom: 6px; font-size: 0.9em; color: #444;">
        💚 Enjoy this service? Help keeping the server turned on for the price of a coffee! In the remote event of surplus, we promise to give it all back to the RC community :-)
      </p>
      <a href="https://www.paypal.com/donate/?hosted_button_id=XXXXXXX" target="_blank">
        <button type="button" style="
          background-color:#4CAF50;
          color:white;
          border:none;
          padding:6px 14px;
          border-radius:4px;
          cursor:pointer;
          font-size:0.9em;
        ">
          Donate
        </button>
      </a>
    </div>
    -->

</body>
</html>
"""

# ---------------- Google sheet log ----------------

SPREADSHEET_ID = "1CTEi_f3mkCvNDXi7NxHHkpvbZ65mWzQ4Y54dZyuabjI"  # paste from Sheets URL
creds_info = None
executor = None
sheet = None

# --- Init Google Sheets logging only if credentials are provided ---
if os.environ.get("GOOGLE_CREDENTIALS"):
    try:
        creds_info = json.loads(os.environ["GOOGLE_CREDENTIALS"])
        creds = service_account.Credentials.from_service_account_info(
            creds_info,
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        service = build("sheets", "v4", credentials=creds)
        sheet = service.spreadsheets()
        executor = ThreadPoolExecutor(max_workers=2)
        print("✅ Google Sheets logging enabled")
    except Exception as e:
        print(f"⚠️ Failed to init Google Sheets logging: {e}")
else:
    print("⚠️ GOOGLE_CREDENTIALS not set, Sheets logging disabled")

# ---------------- Utilities ----------------

def log_to_sheets(row):
    try:
        body = {"values": [row]}
        sheet.values().append(
            spreadsheetId=SPREADSHEET_ID,
            range="A1",
            valueInputOption="USER_ENTERED",
            body=body
        ).execute()
    except Exception as e:
        print(f"⚠️ Sheets logging failed: {e}", flush=True)

def async_log_usage(user_ip, country, event_name, class_name, final_name):
    """Queue a log entry to Google Sheets if logging is enabled."""
    if not executor or not sheet:
        return  # logging disabled, no-op

    #ts = datetime.datetime.utcnow().isoformat()
    ts = datetime.now(timezone.utc).isoformat()

    row = [[ts, user_ip, country, event_name, class_name, final_name]]

    def task():
        try:
            sheet.values().append(
                spreadsheetId=SPREADSHEET_ID,
                range="A1",
                valueInputOption="USER_ENTERED",
                body={"values": row}
            ).execute()
        except Exception as e:
            print(f"⚠️ Sheets logging failed: {e}")

    executor.submit(task)

def parse_duration_to_seconds(duration_str: str) -> int | None:
    """Convert 'MM:SS' or 'HH:MM:SS' into total seconds."""
    try:
        parts = duration_str.strip().split(":")
        if len(parts) == 2:  # MM:SS
            m, s = map(int, parts)
            return m * 60 + s
        elif len(parts) == 3:  # HH:MM:SS
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        else:
            return None
    except Exception:
        return None

def safe_fetch(url, timeout=TIMEOUT_SECONDS):
    """
    Wrapper for requests.get() with better diagnostics.
    Returns (response, error_msg).
    """
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp, None

    except Timeout:
        return None, f"⏰ Timeout: no response from {url}"

    except SSLError as e:
        return None, f"🔒 SSL error contacting {url}: {e}"

    except ConnectionError as e:
        # Distinguish DNS errors
        if isinstance(e.__cause__, socket.gaierror):
            return None, f"🌐 DNS lookup failed for {url} (hostname not resolved)"
        return None, f"🚫 Connection blocked or reset when contacting {url}: {e}"

    except RequestException as e:
        return None, f"❌ Request failed for {url}: {e}"

    except Exception as e:
        return None, f"⚠️ Unexpected error contacting {url}: {e}"

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
    ongoing_url = f"https://myrcm.ch/myrcm/main?hId[1]=evt&pLa=en&dFi={urllib.parse.quote(query)}&hId[1]=search"
    resp, err = safe_fetch(ongoing_url)
    if err:
        print(err)
        return jsonify({"error": f"Network problem: {err}"}), 502

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.select("table tr")
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
        events.append({"name": f"⚡ {event_name} ({host})", "url": event_url})

    # --- Archived events ---
    archived_url = f"https://myrcm.ch/myrcm/main?pLa=en&dFi={urllib.parse.quote(query)}&hId[1]=search"
    resp, err = safe_fetch(archived_url)
    if err:
        print(err)
        return jsonify({"error": f"Network problem: {err}"}), 502

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.select("table tr")
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
        events.append({"name": f"🗄️ {event_name} ({host})", "url": event_url})

    print(f"📊 Extracted {len(events)} total events")
    return jsonify(events)

@app.route("/get_classes")
def get_classes():
    event_url = request.args.get("url", "").strip()
    if not event_url:
        return jsonify([])

    resp, err = safe_fetch(event_url)
    if err:
        print(err)
        return jsonify({"error": f"Network problem: {err}"}), 502

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
        classes.append({"name": class_name, "url": class_url})

    print(f"📊 Extracted {len(classes)} classes")
    return jsonify(classes)

@app.route("/get_finals")
def get_finals():
    class_url = request.args.get("url", "").strip()
    if not class_url:
        return jsonify([])

    user_ip, ua = get_client_info()

    resp, err = safe_fetch(class_url)
    if err:
        print(err)
        return jsonify({"error": f"Network problem: {err}"}), 502

    soup = BeautifulSoup(resp.text, "html.parser")
    finals = []
    in_final_section = False

    content_div = soup.select_one("div.l-content") or soup
    for elem in content_div.find_all(["h2", "h3", "h4", "div", "a"]):
        text = elem.get_text(strip=True).lower()

        if elem.name in ("h2", "h3", "h4", "div") and text:
            if text.startswith("final"):
                in_final_section = True
                continue
            elif in_final_section:
                break

        if elem.name == "a" and in_final_section and elem.has_attr("onclick"):
            m = re.search(r"doAjaxCall\('([^']*)',\s*'([^']*)'\)", elem["onclick"])
            if m:
                raw_name = m.group(2).strip()
                pretty_name = re.sub(r"^Final\s*::\s*", "", raw_name, flags=re.IGNORECASE)

                if not any(x in raw_name.lower() for x in ["practice", "qualif", "ranking", "timeschedule", "participants"]):
                    finals.append({"name": pretty_name, "onclick": m.group(1)})

    print(f"📊 Extracted {len(finals)} finals")
    return jsonify(finals)

@app.before_request
def log_usage():
    if request.endpoint == "index" and request.method == "POST":
        user_ip, _ = get_client_info()
        event_name = request.form.get("event_name", "").strip()
        class_name = request.form.get("class_name", "").strip()
        final_name = request.form.get("final_name", "").strip()

        # ISO timestamp with timezone-aware UTC
        timestamp = datetime.now(timezone.utc).isoformat()

        # Resolve country immediately
        country = get_country_from_ip(user_ip)

        #usage_logger.info(log_line)
        #async_log_usage(user_ip, event_name, class_name, final_name)
        async_log_usage(user_ip, country, event_name, class_name, final_name)

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

@app.route("/sitemap.xml")
def sitemap():
    # For a small site, enumerate routes or generate from stored events.
    # Example static sitemap with site root + some useful pages:
    urls = [
        url_for('index', _external=True),
        url_for('search_events', _external=True),
    ]
    xml = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for u in urls:
        xml.append("<url>")
        xml.append(f"<loc>{u}</loc>")
        xml.append("</url>")
    xml.append("</urlset>")
    return Response("\n".join(xml), mimetype='application/xml')

@app.route("/robots.txt")
def robots():
    lines = [
        "User-agent: *",
        "Sitemap: " + url_for('sitemap', _external=True)
    ]
    return Response("\n".join(lines), mimetype="text/plain")

@atexit.register
def cleanup_executor():
    global executor
    if executor:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
            print("✅ ThreadPoolExecutor shut down")
        except Exception as e:
            print(f"⚠️ Failed shutting down executor: {e}")

# -----------main entry -------------------------------------------    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))  # 5050 for local dev
    app.run(host="0.0.0.0", port=port)
