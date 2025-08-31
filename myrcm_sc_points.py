import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import csv
import sys
import re
import warnings
from urllib.parse import urljoin
import unicodedata

# Ignore the XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def normalize_name(name: str) -> str:
    """Normalize driver names to avoid mismatches due to spaces/encoding."""
    return unicodedata.normalize("NFKC", name).replace("\xa0", " ").strip()

if len(sys.argv) < 2:
    print("Usage: python myrcm_sc_points.py <MAIN_URL>")
    sys.exit(1)

main_url = sys.argv[1]

# Fetch main event page
resp = requests.get(main_url)
soup = BeautifulSoup(resp.text, "html.parser")

# --- Find "Rankinglists :: Final" link ---
final_url = None
for a in soup.find_all("a", onclick=True):
    onclick = a["onclick"]
    if "Rankinglists" in onclick and "Final" in onclick:
        match = re.search(r"doAjaxCall\('([^']+)'", onclick)
        if match:
            final_url = urljoin(main_url, match.group(1))
            break

if not final_url:
    print("❌ Could not find Rankinglists Final link")
    sys.exit(1)

# --- Find "Rankinglists :: Qualy" link ---
qualy_url = None
for a in soup.find_all("a", onclick=True):
    onclick = a["onclick"]
    if "Rankinglists" in onclick and "Qualy" in onclick:
        match = re.search(r"doAjaxCall\('([^']+)'", onclick)
        if match:
            qualy_url = urljoin(main_url, match.group(1))
            break

# --- Get TQ driver from Qualy report ---
tq_driver = None
if qualy_url:
    resp_qualy = requests.get(qualy_url)
    soup_qualy = BeautifulSoup(resp_qualy.text, "html.parser")

    table_q = soup_qualy.find("table", {"id": "data-table"})
    if table_q:
        header_cols = [h.get_text(strip=True) for h in table_q.find_all("th")]
        try:
            pilot_index = header_cols.index("Pilot")
        except ValueError:
            pilot_index = 3  # fallback if header not found

        rows_q = table_q.find_all("tr")[1:]  # skip header
        for row in rows_q:
            cols = row.find_all("td")
            if not cols or len(cols) <= pilot_index:
                continue
            text_cols = [c.get_text(strip=True) for c in cols]
            if text_cols and text_cols[0].isdigit():
                tq_driver = normalize_name(text_cols[pilot_index])  # Pilot name
            break

# --- Fetch Rankinglists Final page ---
resp_final = requests.get(final_url)
soup_final = BeautifulSoup(resp_final.text, "html.parser")

# Build output filename from main page title
title = soup.title.string.strip() if soup.title else "race_results"
title_clean = re.sub(r'[^A-Za-z0-9\\-_ ]+', '', title)
title_abbrev = "_".join(title_clean.split())[:80]
output_file = f"{title_abbrev}_Rankinglists_Final.csv"

# Find the results table
table = soup_final.find("table", {"id": "data-table"})
if not table:
    print("❌ Could not find results table on Rankinglists Final page")
    sys.exit(1)

rows = table.find_all("tr")[1:]  # skip header
results = []

# Points assignment
def get_points(position):
    if position == 1:
        return 50
    elif position == 2:
        return 48
    elif position == 3:
        return 46
    else:
        return max(49 - position, 1)   # 4th=45, 5th=44, ...

# Parse rows
for row in rows:
    # Only keep rows that are direct children of the outer results table
    if row.find_parent("table") != table:
        continue

    cols = row.find_all("td")
    if not cols or len(cols) < 4:
        continue

    text_cols = [c.get_text(strip=True) for c in cols]

    if not text_cols or not text_cols[0].isdigit():
        continue

    position = int(text_cols[0])
    driver = normalize_name(text_cols[3])
    points = get_points(position)

    tq_flag = 1 if tq_driver and driver == tq_driver else 0

    results.append([position, driver, points, tq_flag])

# Save to CSV
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Position", "Driver", "Points", "TQ"])
    writer.writerows(results)

print(f"✅ Results saved to {output_file}")
if tq_driver:
    print(f"ℹ️ TQ identified as: {tq_driver}")
