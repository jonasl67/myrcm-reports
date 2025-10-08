#!/usr/bin/env python3
"""
myrcm_stats.py

Usage:
    python myrcm_stats.py <racefile.csv> [--verbose]

Reads the race CSV, parses driver summary & lap-by-lap data,
extracts track positions from the "(pos)" text in each lap cell,
computes per-driver stats, and writes a PDF report (A4 landscape).
"""
import sys
import csv
import os
import re
import math
import pandas as pd
import numpy as np
import pathlib
import argparse

# ReportLab imports
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Frame, PageTemplate
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.graphics.shapes import Drawing, String, Line
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.lib.colors import black, red, green, blue, orange, violet, pink, gray, Color

# ---------------- Config ----------------
# NOTE: The values below are based on analysis and assumptions to correctly classify
# different in-race events, especially pit stops.
# The core logic assumes a 'clean lap' median as a baseline for time lost.
EVENT_THRESHOLD = 3.0            # minor incident threshold (sec)
FUEL_MIN_LOSS = 4.5                 # minimum seconds lost to call it a fuel lap
MAJOR_EVENT_THRESHOLD = 15.0     # major incident threshold (sec) => auto refuel

# Even though FUEL_WINDOW has an upper bound (360s),
# the classification logic is intentionally more lenient:
#   - Stops later than 6:00 can still be marked as fuel if their lap-time delta
#     matches a "typical" fuel stop (based on previous stops).
#   - A late stop after a "MAJOR EVENT + FUEL" is still considered a fuel stop
#     even if outside the nominal window.
#   - The final stop of a race can be accepted earlier than 3:50 if it plausibly
#     carries the car to the finish without another stop.
#
# This design allows both short on-road nitro stints (~4–5 min) and longer buggy stints (~7–8 min)
# to be classified correctly, without tuning two separate parameter sets.
# So FUEL_WINDOW acts more like a *guideline* than a hard filter.
FUEL_WINDOW = (230, 360)            # 3:50 - 6:00 minutes
STANDARD_FUEL_STINT = 245            # the most common length of a fuel stint

# used to indicate that no fueling laps shall be classified, for electric races
NOFUEL_MODE = False

# used in summary table for fun
# Unicode superscript digits: ¹, ², ³
TROPHIES = ["¹", "²", "³"]


# ---------------- Utilities ----------------

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

def parse_lap_entry(entry):
    """Parse lap cell like '(2) 14.805' or '1:15.523' or '15.523' -> seconds as float or np.nan."""
    if pd.isna(entry):
        return np.nan
    s = str(entry).strip()
    # take text after last ')'
    if ')' in s:
        s = s.split(')')[-1].strip()
    # mm:ss(.xx)
    if ':' in s:
        try:
            parts = s.split(':')
            mins = int(parts[0])
            secs = float(parts[1])
            return mins * 60 + secs
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan    

def parse_pos_from_entry(entry):
    """Extract the integer position inside parentheses like '(2) 03.863' -> 2. Returns None if not found."""
    if pd.isna(entry):
        return None
    s = str(entry)
    m = re.search(r'\((\d+)\)', s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def format_time_mm_ss(seconds):
    if seconds is None or (isinstance(seconds, float) and (math.isnan(seconds) or seconds < 0)):
        return "00:00"
    total = int(round(seconds))
    m = total // 60
    s = total % 60
    return f"{m:02d}:{s:02d}"

def format_time_mm_ss_decimal(seconds):
    if seconds is None or (isinstance(seconds, float) and (math.isnan(seconds) or seconds < 0)):
        return "00:00.00"
    total_seconds = int(math.floor(seconds))
    minutes = total_seconds // 60
    sec = total_seconds % 60
    cent = int((seconds - total_seconds) * 100)
    return f"{minutes:02d}:{sec:02d}.{cent:02d}"

def format_log_time(seconds):
    """Format time as 'ss.ss' for less than 60s, or 'mm:ss.ss' for 60s or more."""
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "00.00"
    if seconds >= 60:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes:02d}:{remaining_seconds:05.2f}"
    else:
        return f"{seconds:.2f}"

def safe_int(v, default=9999):
    try:
        return int(v)
    except Exception:
        return default
    
def get_driver_at_position(lap_num, pos):
    """
    Helper function to find a driver's name based on their position for a given lap.
    
    Args:
        lap_num (int): The lap number.
        pos (int): The track position.

    Returns:
        str: The name of the driver, or None if not found.
    """
    if lap_num < 1 or pos < 1:
        return None
    
    # Iterate through the driver names and their lists of positions
    for driver_name, positions in lap_positions.items():
        # Adjust lap_num to be a 0-based index
        lap_index = lap_num - 1
        
        # Check if the list of positions is long enough
        if lap_index < len(positions):
            # Check if the position at that lap matches the requested position
            if positions[lap_index] == pos:
                return driver_name
    return None

# Decide whether the time delta passed on looks like a fuel pit stop
def looks_like_fuel(d):
    if avg_fuel_delta > 0:
        return (d >= FUEL_MIN_LOSS * 0.9) and (d <= avg_fuel_delta * 1.2)
    else:
        return d >= FUEL_MIN_LOSS

def format_time_ss_decimal(seconds):
    """Format as 'SS.ss' (no minutes)."""
    if seconds is None or (isinstance(seconds, float) and (math.isnan(seconds) or seconds < 0)):
        return "00.00"
    return f"{seconds:.2f}"

def award_trophies(df, column, higher_is_better=True, min_valid_value=0.0):
    """
    Add trophy symbols (¹, ², ³) to the top 3 entries in a given column.

    Args:
        df (pd.DataFrame): Summary DataFrame.
        column (str): Column to rank and modify.
        higher_is_better (bool): Whether larger values rank higher.
        min_valid_value (float): Minimum numeric value to consider valid.

    The column may contain string values like "93.4%" — those are handled automatically.
    """
    if column not in df.columns:
        return df  # nothing to do

    # Create a numeric helper column (strip '%' if present)
    numeric_col = f"{column} (num)"
    df[numeric_col] = (
        df[column]
        .astype(str)
        .str.replace("%", "", regex=False)
        .astype(float)
    )

    valid = df[df[numeric_col].notna() & (df[numeric_col] > min_valid_value)]
    if valid.empty:
        return df

    # Pick direction (best = highest or lowest)
    ranked = (
        valid.nlargest(3, numeric_col)
        if higher_is_better
        else valid.nsmallest(3, numeric_col)
    )

    for i, (idx, row) in enumerate(ranked.iterrows()):
        df.at[idx, column] = f"  {row[column]} {TROPHIES[i]}"

    return df


# ---------------- CLI / main() ----------------
parser = argparse.ArgumentParser(
    description="Generate RC race stats PDF from myrcm CSV"
)
parser.add_argument("csv_file", help="Input race CSV file")
parser.add_argument("pdf_file", nargs="?", help="Output PDF file (optional)")
parser.add_argument("--verbose", action="store_true", help="Enable verbose logging to stdout")
parser.add_argument("--nofuel", action="store_true", help="Disable fuel stop classification (for electric classes)")

args = parser.parse_args()

csv_file = args.csv_file
verbose = args.verbose
NOFUEL_MODE = args.nofuel

# ---------------- Read file & race info lines ----------------
lap_start_line = None
raw_driver_info = []  # tuples (full_name, car_number) in summary order (finishing order)
race_duration_sec = None
race_event = race_name = race_details = ""

try:
    with open(csv_file, newline='') as fh:
        reader = csv.reader(fh)
        all_lines = list(reader)

    # metadata lines
    if len(all_lines) > 0 and len(all_lines[0]) > 0:
        race_event = all_lines[0][0].strip()
    if len(all_lines) > 1 and len(all_lines[1]) > 0:
        race_name = all_lines[1][0].strip()
    if len(all_lines) > 2 and len(all_lines[2]) > 0:
        race_details = all_lines[2][0].strip()

    # find lap start line and race time token
    for i, row in enumerate(all_lines):
        for cell in row:
            if isinstance(cell, str) and "Race time:" in cell:
                try:
                    # take the token after 'Race time:'
                    part = cell.split("Race time:")[-1].strip().split()[0]
                    race_duration_sec = parse_duration_to_seconds(part)
                    if race_duration_sec is None:
                        print(f"⚠️ Could not parse race duration from: '{part}'", file=sys.stderr)
                except Exception as e:
                    print(f"⚠️ Failed parsing race duration: {e}", file=sys.stderr)

        if row and isinstance(row[0], str) and row[0].strip().lower().startswith("laptimes"):
            lap_start_line = i + 1
            break

    # --- Auto-detect NOFUEL mode if not explicitly set 
    if not NOFUEL_MODE:
        lowered_event = race_event.lower()
        if "electric" in lowered_event or "1:12" in lowered_event or "FWD" in lowered_event:
            NOFUEL_MODE = True
            print("⚡ Auto-detected a no-refuel race → disabling fuel stop classification", file=sys.stderr)
        elif race_duration_sec is not None and race_duration_sec <= 300:
            NOFUEL_MODE = True
            print(f"⚡ Auto-detected short race ({race_duration_sec}s ≤ 300s) → disabling fuel stop classification", file=sys.stderr)


    # collect summary block rows: look for rows where col[1] numeric and col[3] name exists
    for row in all_lines:
        if (
            len(row) >= 6
            and row[1].strip().isdigit()
            and row[3].strip()
            and row[5].strip().isdigit()
        ):
            driver_name = row[3].strip()
            car_no = row[1].strip()
            laps = int(row[5].strip())
            raw_driver_info.append((driver_name, car_no, laps))
        #if len(row) >= 4 and row[1].strip().isdigit() and row[3].strip():
        #    raw_driver_info.append((row[3].strip(), row[1].strip()))
except FileNotFoundError:
    print(f"File not found: {csv_file}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print("Failed to read file:", e, file=sys.stderr)
    sys.exit(1)

if lap_start_line is None:
    print("Could not find 'Laptimes' start row in CSV.", file=sys.stderr)
    sys.exit(1)

# load lap table using pandas
try:
    df = pd.read_csv(csv_file, skiprows=lap_start_line)
except Exception as e:
    print("Error loading lap table with pandas:", e, file=sys.stderr)
    sys.exit(1)

# header names for drivers are columns after first
header_names = df.columns[1:].tolist()

# build driver_lookup mapping header_driver -> (full_name, car_number) using the summary order
driver_lookup = {}
for idx, driver in enumerate(header_names):
    if idx < len(raw_driver_info):
        driver_lookup[driver] = raw_driver_info[idx] # (full_name, car_number, laps)
    else:
        driver_lookup[driver] = (driver, "?", 0)


# ---------------- Build driver lap data (include all drivers) ----------------
driver_lap_data = {}
n_rows = df.shape[0]

for col_idx, driver in enumerate(header_names, start=1):
    raw_entries = df.iloc[:, col_idx].tolist()  # list of raw strings (may include NaN)

    # parse lap times ignoring NaN entries
    parsed_times = []
    for e in raw_entries:
        t = parse_lap_entry(e)
        if not (isinstance(t, float) and math.isnan(t)):
            parsed_times.append(t)
    #full_name, car_no = driver_lookup.get(driver, (driver, "?"))
    full_name, car_no, official_laps = driver_lookup.get(driver, (driver, "?", 0))
    driver_lap_data[driver] = {
        "raw_entries": raw_entries,
        "lap_times": np.array(parsed_times),
        "full_name": full_name,
        "car_number": car_no,
        "no_of_laps": official_laps
    }


# --- Find global hot lap (ignore first laps) --------------------------------
all_valid_laps = []
for ddata in driver_lap_data.values():
    laps = ddata["lap_times"]
    if len(laps) > 1:
        all_valid_laps.extend(laps[1:])  # skip first lap
if all_valid_laps:
    global_hot_lap = min(all_valid_laps)
else:
    global_hot_lap = None

print(f"🔥 Global hot lap = {global_hot_lap:.3f}s")  # optional debug


# ---------------- Determine orders ----------------
# finishing order: prefer the summary/top block order (header order). Keep only drivers present.
# Filter out 'Unnamed:' drivers before determining order
filtered_headers = [d for d in header_names if not re.match(r'Unnamed: \d+', d)]
finishing_order = [driver for driver in filtered_headers if driver in driver_lap_data]
starting_order = sorted(finishing_order, key=lambda a: safe_int(driver_lap_data[a]["car_number"], default=9999))

# starting order: sort by numeric car number if possible (car #1 top)
starting_order = sorted(finishing_order, key=lambda a: safe_int(driver_lap_data[a]["car_number"], default=9999))

# color palette mapped by starting order index
base_colors = [black, red, green, blue, orange, violet, pink, gray,
               Color(0.5, 0.2, 0.8), Color(0.8, 0.5, 0.2)]
while len(base_colors) < len(starting_order):
    idx = len(base_colors)
    base_colors.append(Color(((idx*37) % 255)/255.0, ((idx*61) % 255)/255.0, ((idx*97) % 255)/255.0))


# ---------------- Build lap_positions from parentheses (preferred) ----------------
# Parse positions per row for every driver (None if no '(pos)')
parsed_pos_by_driver = {}
max_rows = n_rows
for driver in header_names:
    raw_entries = driver_lap_data[driver]["raw_entries"]
    pos_list = []
    for row_idx in range(max_rows):
        if row_idx < len(raw_entries):
            pos = parse_pos_from_entry(raw_entries[row_idx])
            pos_list.append(pos)
        else:
            pos_list.append(None)
    parsed_pos_by_driver[driver] = pos_list

# Produce lap_positions per driver but **only** up to the last seen position.
# If a driver has no positions at all -> empty list (no line plotted).
lap_positions = {}
for driver in starting_order:  # use starting_order so left labels align with chart
    pos_list = parsed_pos_by_driver.get(driver, [])
    # find last index that contains a real position
    last_pos_idx = -1
    for i, p in enumerate(pos_list):
        if p is not None:
            last_pos_idx = i
    if last_pos_idx == -1:
        # no track positions captured for this driver -> do not plot a line
        lap_positions[driver] = []
        continue

    # build a series 0..last_pos_idx inclusive
    series = []
    last_known = None
    # fallback for initial missing positions: use starting grid slot
    start_slot = starting_order.index(driver) + 1
    for i in range(last_pos_idx + 1):
        p = pos_list[i]
        if p is not None:
            last_known = p
            series.append(p)
        else:
            if last_known is not None:
                series.append(last_known)   # carry forward previous known position
            else:
                series.append(start_slot)   # before first known pos -> starting slot

    lap_positions[driver] = series


# ---------------- Per-driver analysis (iterate in finishing order so Pos aligns with summary) ----------------
results = []
verbose_log = {}

for pos_idx, driver in enumerate(finishing_order, start=1):
    data = driver_lap_data[driver]
    lap_times = data["lap_times"]
    full_name = data["full_name"]
    car_no = data["car_number"]
    n_laps = data["no_of_laps"]
    
    total_time = float(np.sum(lap_times)) if n_laps > 0 else 0.0

    # avg clean lap (median of laps below 90th percentile)
    avg_clean_lap = None
    if n_laps > 0:
        try:
            clean_threshold = np.percentile(lap_times, 90)
            clean_laps = lap_times[lap_times < clean_threshold]
            avg_clean_lap = float(np.median(clean_laps)) if clean_laps.size else float(np.median(lap_times))
        except Exception:
            avg_clean_lap = float(np.median(lap_times))
    else:
        avg_clean_lap = 0.0


    # classification pass (fuel stops, tire changes, incidents)
    fuel_stops_idx = [] # use a list for indices of all fuel stops
    normal_fuel_stop_deltas = [] #  list to store deltas of fuel stops
    tire_changes = []
    major_incidents = []
    minor_incidents = []
    fuel_time_lost = tire_time_lost = major_time_lost = minor_time_lost = 0.0

    current_time = 0.0
    last_fuel_time = 0.0
    last_tire_time = 0.0
    driver_events = []
    
    # use the parsed lap_times sequence for classification if events ------------
    last_event = None
    for i, laptime in enumerate(lap_times):
        prev_time = current_time
        current_time += laptime
        delta = laptime - avg_clean_lap
        event = None
        
        if NOFUEL_MODE: 
            # Skip all fuel logic – just classify by thresholds
            if delta >= MAJOR_EVENT_THRESHOLD:
                event = "MAJOR EVENT"
                major_incidents.append(i)
                major_time_lost += max(0.0, delta)
            elif delta >= EVENT_THRESHOLD:
                event = "MINOR EVENT"
                minor_incidents.append(i)
                minor_time_lost += max(0.0, delta)

        else:
            # re-fueling mode
            # --- Fuel stop classification ---
            # Normally: require ≥ min fuel window (3:50) and ≥4.5s lost
            # But the following exceptions broaden detection:
            #   * After a MAJOR EVENT + FUEL, the next long stop can be tagged as fueling even if >6:00
            #   * A last early splash (within 2–3 min of finish) can count if it looks like a typical fuel stop
            #   * Any stop beyond 6:00 with a fuel-like delta is accepted
            # This flexible approach explains why valid 7–8 min buggy fuel stops
            # are still recognized despite FUEL_WINDOW = (230, 360).

            # Determine average fuel lap time lost for re-classification logic
            avg_normal_fuel_lap_lost_time = np.median(normal_fuel_stop_deltas) if normal_fuel_stop_deltas else 0.0
            time_since_fuel = current_time - last_fuel_time
            avg_fuel_delta = avg_normal_fuel_lap_lost_time if avg_normal_fuel_lap_lost_time else 0.0

            # first capture major events / lap time deltas, default is to assume the driver got fuel during such long delays
            if delta >= MAJOR_EVENT_THRESHOLD:
                race_time_left = race_duration_sec - current_time
                time_to_next_fuel = STANDARD_FUEL_STINT - (current_time - last_fuel_time)

                # Default assumption: fueling happened
                event = "MAJOR EVENT + FUEL"
                do_fuel = True

                # Exception 1: Not enough race left to justify fueling
                if race_time_left < time_to_next_fuel:
                    event = "MAJOR EVENT"
                    do_fuel = False

                # Exception 2: Happens within first minute of race (start carnage)
                elif current_time < 60:
                    event = "MAJOR EVENT"
                    do_fuel = False

                major_incidents.append(i)
                major_time_lost += max(0.0, delta)
                if do_fuel:
                    fuel_stops_idx.append(i)

            # capture fuel stops
            elif (
                # normal fuel stop, time since last stop is in fueling window and minimum time loss for fuel stop is exceeded/satisfied
                (time_since_fuel >= FUEL_WINDOW[0] and delta >= FUEL_MIN_LOSS)
                #(time_since_fuel >= FUEL_WINDOW[0] and time_since_fuel <= FUEL_WINDOW[1] and looks_like_fuel(delta)) # fails to catch long fuel stops

                # Capture stops/events that happens after a MAJOR EVENT + FUEL as fuel stops even if they are outside/longer than the fueling windows.
                # This happens because the car was fueled up leaving the long pit stop (time since last fuel is calculated from start of pit stop/event)
                or (time_since_fuel > FUEL_WINDOW[1] and last_event == "MAJOR EVENT + FUEL" and avg_normal_fuel_lap_lost_time > 0 and looks_like_fuel(delta))

                # This is to capture a last needed fuel stop of a race that may well happen before its strictly needed, but will still take the driver to the finish without another fuel stop
                # # If this stop happens earlier than the minimum fuel window but still at least 2 minutes after the last stop,
                # and the lap’s time loss looks like a normal fuel stop (≥5s but within 20% of the average fuel stop loss),
                # then also count it as a fuel stop.
                or (((race_duration_sec-current_time) < STANDARD_FUEL_STINT) and time_since_fuel < FUEL_WINDOW[0] and time_since_fuel > 120 and avg_normal_fuel_lap_lost_time > 0 and looks_like_fuel(delta))

                # Catch laps that are beyond max fuel stint and looks like a typical fuel lap delta, 
                or ((time_since_fuel > FUEL_WINDOW[1]) and (delta >= (FUEL_MIN_LOSS * 0.9)))
            ):
                #print(f"Last required fuel stop for {driver} - Race time left: {race_duration_sec-current_time} Average fuel stint:  \n") #debug

                event = "FUEL STOP"
                fuel_stops_idx.append(i)
                fuel_time_lost += max(0.0, delta)
                normal_fuel_stop_deltas.append(delta)

            elif delta >= EVENT_THRESHOLD:
                event = "MINOR EVENT"
                minor_incidents.append(i)
                minor_time_lost += max(0.0, delta)

        # --- calculate consistency % ---
        consistency_pct = 0.0
        if n_laps > 1:
            # use only "clean" laps (filter out NaNs)
            lap_times_clean = lap_times[~np.isnan(lap_times)]
            if lap_times_clean.size > 1:
                mean_lap = np.mean(lap_times_clean)
                std_lap = np.std(lap_times_clean)
                if mean_lap > 0:
                    consistency_pct = max(0.0, (1.0 - (std_lap / mean_lap)) * 100.0)

        if event:            
            last_event = event # save last event since we may need it to classify an upcoming event
            before_pos = lap_positions[driver][i-1] if i > 0 else lap_positions[driver][0]
            after_pos = lap_positions[driver][i]
            
            log_line = f"Lap {i+1} completed at={format_time_mm_ss(current_time)}, {event}, lap time={format_log_time(laptime)}, time lost={format_log_time(delta)}"
            
            if "FUEL" in event:
                time_of_fueling = prev_time + (avg_clean_lap / 2.0)
                if len(fuel_stops_idx) == 1:
                    last_fuel_stint_duration = time_of_fueling
                else:
                    last_fuel_stint_duration = time_of_fueling - last_fuel_time
                
                log_line += f", last fuel={format_time_mm_ss(last_fuel_stint_duration)}"
                last_fuel_time = time_of_fueling
            else:
                last_fuel_stint_duration = current_time - last_fuel_time
                log_line += f", last fuel={format_time_mm_ss(last_fuel_stint_duration)}"

            # show track position before/after the event (like a re-fuel lap)
            if before_pos == after_pos:
                log_line += f", track position={before_pos}"
            else:
                log_line += f", track position={before_pos}→{after_pos}"
                
                if before_pos > after_pos:
                    # Position lost, so the drivers who passed are from before_pos+1 to after_pos
                    passed_drivers = []
                    for pos in range(before_pos + 1, after_pos + 1):
                        driver_name = get_driver_at_position(i, pos)
                        if driver_name:
                            surname = driver_name.split()[-1]
                            passed_drivers.append(surname)
                    if passed_drivers:
                        log_line += f" passed: {', '.join(passed_drivers)}"
                
                elif before_pos < after_pos:
                    # Position gained, so the drivers who were passed are from end_pos to start_pos-1
                    passed_drivers = []
                    for pos in range(after_pos, before_pos, -1):
                        driver_name = get_driver_at_position(i, pos)
                        if driver_name:
                            surname = driver_name.split()[-1]
                            passed_drivers.append(surname)
                    if passed_drivers:
                        log_line += f" passed by: {', '.join(passed_drivers)}"
                    
            
            driver_events.append(log_line)

    # Build the summary for this driver ---------------------------------------

    if NOFUEL_MODE:
        avg_fuel_interval = 0
        avg_fuel_stop_time = 0
        fuel_time_lost_str = 0
    else:
        # calc average fuel stop time
        avg_fuel_stop_time = (fuel_time_lost / len(normal_fuel_stop_deltas)) if normal_fuel_stop_deltas else 0.0
        
        # fuel intervals
        fuel_times = []
        accum = 0.0
        for i, laptime in enumerate(lap_times):
            accum += laptime
            if i in fuel_stops_idx:
                fuel_times.append(max(0.0, accum - laptime + avg_clean_lap/2.0))

        fuel_intervals = np.diff([0.0] + fuel_times) if fuel_times else np.array([])
        avg_fuel_interval = float(np.mean(fuel_intervals)) if fuel_intervals.size else 0.0

    total_incident_time_lost = minor_time_lost + major_time_lost
    total_incident_laps = len(minor_incidents) + len(major_incidents)

    display = f"{full_name} ({car_no})"

    # -- consistency for this driver
    consistency = 0.0
    if n_laps > 1:
        mu = np.mean(lap_times)
        sigma = np.std(lap_times)
        if mu > 0:
            consistency = max(0, min(100, 100 * (1 - sigma / mu)))

    # -- Best lap for this driver (ignore first lap) ---
    best_lap = 0.0
    if n_laps > 1:
        valid_laps = lap_times[1:]  # skip first lap
        best_lap = float(np.min(valid_laps)) if valid_laps.size else 0.0
    else:
        best_lap = float(np.min(lap_times)) if lap_times.size else 0.0

    # -- Calculate laps within 3% of global hot lap for this driver ---
    hot_lap_global = np.nanmin([t for d in driver_lap_data.values() for t in d["lap_times"][1:] if not np.isnan(t)]) if driver_lap_data else None
    laps_within_3pct = 0
    if hot_lap_global and n_laps > 1:
        laps_excluding_first = lap_times[1:]
        within_mask = laps_excluding_first <= (hot_lap_global * 1.03)
        laps_within_3pct = round(100.0 * np.sum(within_mask) / len(laps_excluding_first), 1)

    results.append({
        "Pos": pos_idx,
        "Driver (No)": display,
        "Laps": n_laps,
        "Race Time": format_time_mm_ss_decimal(total_time),
        "Best Lap": format_time_ss_decimal(best_lap),
        "90 percentile lap": f"{avg_clean_lap:.2f}" if avg_clean_lap else "0.00",
        "Consistency": f"{consistency:.1f}%",
        "Laps within 3% of hot lap": f"{laps_within_3pct:.0f}%",
        "Fuel Stops": len(fuel_stops_idx),
        "Avg Fuel Interval": format_time_mm_ss(avg_fuel_interval),
        "Avg Fuel Stop Time": format_time_ss_decimal(avg_fuel_stop_time),
        "Events/Incidents": total_incident_laps,
        "Fuel Time Lost": format_time_mm_ss_decimal(fuel_time_lost),
        "Events Time Lost": format_time_mm_ss_decimal(total_incident_time_lost)
    })

    if driver_events:
        verbose_log[display] = driver_events

# Find and mark / trophies top three drivers for some metrics  --------------------
summary_df = pd.DataFrame(results)
if not summary_df.empty:
    # 1. Fastest laps — lower is better
    summary_df = award_trophies(summary_df, "Best Lap", higher_is_better=False)

    # 2. Fastest 90 percentile laps — lower is better
    summary_df = award_trophies(summary_df, "90 percentile lap", higher_is_better=False)

    # 3. Consistency — higher is better
    summary_df = award_trophies(summary_df, "Consistency", higher_is_better=True)

    # 4. % laps within 3% of hot lap — higher is better
    summary_df = award_trophies(summary_df, "Laps within 3% of hot lap", higher_is_better=True)

# ---------------- PDF Report generation ----------------
if len(sys.argv) >= 3 and not sys.argv[2].startswith("--"):
    pdf_path = pathlib.Path(sys.argv[2])
else:
    pdf_path = pathlib.Path(os.path.splitext(os.path.basename(csv_file))[0] + ".pdf")

margin = 20

def add_header_and_footer(canvas, doc):
    """Draws a footer at the bottom right of each page."""
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, doc.bottomMargin, "(C) Balthazar RC")
    canvas.restoreState()

# Create a SimpleDocTemplate instance
doc = SimpleDocTemplate(str(pdf_path), pagesize=landscape(A4),
                        leftMargin=margin, rightMargin=margin,
                        topMargin=margin, bottomMargin=margin)

# Define the frame for the main content
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')

# Create a PageTemplate with the frame and the header/footer function
template = PageTemplate(id='main_page', frames=[frame], onPage=add_header_and_footer)

# Now, use the doc object's pageTemplates attribute to apply the template
doc.pageTemplates = [template]

styles = getSampleStyleSheet()
left_style = ParagraphStyle("Left", parent=styles["Normal"], alignment=TA_LEFT)
center_bold = ParagraphStyle("CenterBold", parent=styles["Normal"], alignment=TA_CENTER, fontName="Helvetica-Bold")

elements = []

#elements.append(Paragraph("Race Summary: An analysis of myrcm race data", styles["Heading1"]))
title = "Race Summary: "
if race_event:
    title = title + race_event + " "
if race_name:
    title = title + race_name + " "
elements.append(Paragraph(title, styles["Heading1"]))

# Add the disclaimer note
custom_note_style = ParagraphStyle(
    "CustomNoteStyle",
    parent=styles["Normal"],
    fontSize=7,
    leading=12,
    spaceAfter=12,
    textColor=colors.grey,
    leftIndent=4,
    fontName="Helvetica"
)

note_text = (
    "<b>Disclaimer - This is not an official myrcm.ch report!</b><br/>"
    "This report is purely based on the data published on myrcm.ch for the race, "
    "automated software has generated the report from the data. The software makes a lot of "
    "assumptions based on typical 1/8 and 1/10 nitro rc car on-road races to arrive "
    "at the information presented in the summary table and event log. "
    "Identifying fuel stops and other events by purely analyzing lap times is unreliable because the analysis relies on a "
    "baseline of lap times from an uneventful race. This means the more incidents a driver has during a race, the less reliable the fuel stop information will be. "
    "The track position chart is based on official myrcm data when available, or by "
    "calculating positions from lap times when track positions are not made avaialble."
)

if NOFUEL_MODE:
    note_text += (
        "<br/><br/><b>Note:</b> Fuel stop detection is <b>disabled</b> in this race report."
        " All events are classified as minor or major incidents only."
    )

elements.append(Spacer(1, 16))
elements.append(Paragraph(note_text, custom_note_style))
elements.append(Spacer(1, 16))

# Race details
if race_details:
    elements.append(Paragraph(race_details, left_style))
elements.append(Spacer(1, 8))


# Summary table
if not summary_df.empty:
    header_row = [
        Paragraph("Pos", center_bold),
        Paragraph("Driver (No)", center_bold),
        Paragraph("Laps", center_bold),
        Paragraph("Race<br/>Time", center_bold),
        Paragraph("Best<br/>Lap", center_bold),
        Paragraph("90 percentile lap", center_bold),
        Paragraph("Consistency", center_bold),
        Paragraph("Laps<br/>within 3%<br/>of hot lap", center_bold),
        Paragraph("Fuel<br/>Stops", center_bold),
        Paragraph("Avg Fuel<br/>Interval", center_bold),
        Paragraph("Avg Fuel<br/>Stop Time", center_bold),
        Paragraph("Events/<br/>Incidents", center_bold),
        Paragraph("Fuel Time Lost", center_bold),
        Paragraph("Events Time Lost", center_bold),
    ]
    table_data = [header_row] + summary_df[[
        "Pos",
        "Driver (No)",
        "Laps",
        "Race Time",
        "Best Lap",
        "90 percentile lap",
        "Consistency",
        "Laps within 3% of hot lap",
        "Fuel Stops",
        "Avg Fuel Interval",
        "Avg Fuel Stop Time",
        "Events/Incidents",
        "Fuel Time Lost",
        "Events Time Lost"
    ]].values.tolist()

    # Leave a bit more margin around the table
    left_right_margin = 30
    available_width = landscape(A4)[0] - 2 * margin - (2 * left_right_margin)

    # Balanced column width ratios (sum ≈ 1.0)
    col_ratios = [
        0.05,  # Pos
        0.24,  # Driver (Nr)
        0.05,  # Laps
        0.09,  # Race Time
        0.07,  # Best lap
        0.07,  # 90 percentile lap
        0.07,  # Consistency
        0.07,  # Laps within 3% of hot lap
        0.05,  # Fuel Stops
        0.07,  # Avg Fuel Interval
        0.05,  # Avg Fuel Stop Time
        0.07,  # Events/Incidents
        0.05,  # Fuel Time Lost
        0.05,  # Events Time Lost
    ]
    col_widths = [r * available_width for r in col_ratios]

    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        # Header formatting
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),

        # Body formatting
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTSIZE', (0,1), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.25, colors.black),

        # Padding tweaks to make the layout tighter
        ('LEFTPADDING', (0,0), (-1,-1), 2),
        ('RIGHTPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))

    # Add horizontal spacing (visual margins)
    elements.append(Spacer(1, 4))
    elements.append(tbl)
    elements.append(Spacer(1, 4))

    # foot note
    note_style = ParagraphStyle(
        "NoteStyle",
        parent=styles["Normal"],
        fontSize=7,
        textColor=colors.grey,
        leftIndent=15  
    )
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "Note: 'Fuel Time Lost' and 'Avg Fuel Stop Time' include pure fuel stops only. "
        "Major events (e.g., flame outs, tire changes) that typically also include fueling are counted separately under 'Events Time Lost'.",
        note_style
    ))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "Note: 'Consistency %' is calculated as 100 × (1 − σ/μ), where σ = standard deviation of lap times and μ = average lap time. "
        "Higher values indicate more consistent driving.",
        note_style
    ))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "Note: 'Laps within 3% of hot lap' is the percentage of a drivers laps that were within 3% of the best lap of the race across all drivers.",
        note_style
    ))


else:
    elements.append(Paragraph("No summary could be computed.", left_style))

elements.append(PageBreak())

# ---------------- Drivers Track Positions chart ----------------
elements.append(Paragraph("Drivers Track Positions", styles["Heading1"]))
elements.append(Spacer(1, 8))

chart_width = landscape(A4)[0] - 2 * margin
chart_height = 360
drawing = Drawing(chart_width, chart_height)

hc = HorizontalLineChart()
hc.x = 100            # leave space for left labels
hc.y = 40
hc.width = chart_width - 260   # room left/right for labels
hc.height = chart_height - 100

# Only consider drivers that have at least one parsed position
drivers_for_chart = [d for d in starting_order if lap_positions.get(d)]
if drivers_for_chart:
    max_laps_to_plot = max(len(lap_positions[d]) for d in drivers_for_chart)
else:
    max_laps_to_plot = 0

if max_laps_to_plot == 0:
    elements.append(Paragraph("No track position data available to draw the chart.", left_style))
    elements.append(PageBreak())
else:
    # Build hc.data from each driver's full lap series (no downsampling!)
    hc.data = []
    n_drivers = len(starting_order)

    for driver in drivers_for_chart:
        series = lap_positions[driver][:]
        def _clamp_val(v):
            try:
                iv = int(v)
            except Exception:
                iv = starting_order.index(driver) + 1
            if iv < 1:
                return 1
            if iv > n_drivers:
                return n_drivers
            return iv
        series = list(map(_clamp_val, series))
        if len(series) == 1:
            series = series * 2
        hc.data.append(series)

    # x categories = all laps, but only show labels every step
    if max_laps_to_plot <= 70:
        step = 1
    elif max_laps_to_plot <= 150:
        step = 5
    else:
        step = 10

    hc.categoryAxis.categoryNames = [
        str(i+1) if (i+1) % step == 0 else ''
        for i in range(max_laps_to_plot)
    ]
    hc.categoryAxis.labels.boxAnchor = 'n'
    hc.categoryAxis.labels.angle = 45
    hc.categoryAxis.labels.fontName = 'Helvetica-Bold'
    hc.categoryAxis.labels.fontSize = 6
    hc.categoryAxis.labels.dy = -8

    # Disable built-in vertical grid
    hc.categoryAxis.visibleGrid = False

    # y axis = position 1..N
    hc.valueAxis.valueMin = 1
    hc.valueAxis.valueMax = n_drivers
    hc.valueAxis.valueStep = 1
    hc.valueAxis.reverseDirection = True
    hc.valueAxis.visibleGrid = True
    hc.valueAxis.labels.fontName = 'Helvetica'
    hc.valueAxis.labels.fontSize = 7

    # assign colors
    for i in range(len(hc.data)):
        try:
            driver = drivers_for_chart[i]
            color_idx = starting_order.index(driver)
            hc.lines[i].strokeColor = base_colors[color_idx % len(base_colors)]
            hc.lines[i].strokeWidth = 1.6
        except Exception:
            pass

    drawing.add(hc)

    # --- Custom vertical grid lines ---
    # Compute scaling of lap → x coord
    x0 = hc.x
    x1 = hc.x + hc.width
    lap_to_x = lambda lap: x0 + (lap-1) * (hc.width / max_laps_to_plot)

    for lap in range(step, max_laps_to_plot+1, step):
        x = lap_to_x(lap)
        line = Line(x, hc.y, x, hc.y + hc.height, strokeColor=colors.lightgrey, strokeWidth=0.25)
        drawing.add(line)

    # Labels and annotations
    def y_for_rank(rank, value_min=1, value_max=None):
        if value_max is None:
            value_max = n_drivers
        if value_max == value_min:
            return hc.y + hc.height / 2.0
        frac = (rank - value_min) / float(value_max - value_min)
        return hc.y + (1.0 - frac) * hc.height

    top_label_y = y_for_rank(1) + 20
    bottom_label_y = hc.y - 30
    left_label_x = 10
    right_label_x = hc.x + hc.width + 20

    drawing.add(String(left_label_x, top_label_y, "Starting pos", fontName="Helvetica-Bold", fontSize=10))
    drawing.add(String(right_label_x, top_label_y, "Finishing pos", fontName="Helvetica-Bold", fontSize=10))
    drawing.add(String(hc.x + hc.width / 2, bottom_label_y, "Lap", fontName="Helvetica-Bold", fontSize=10))

    for idx, driver in enumerate(starting_order):
        start_rank = idx + 1
        y = y_for_rank(start_rank)
        full_name = driver_lap_data[driver]['full_name']
        car_num = driver_lap_data[driver]['car_number']
        color = base_colors[idx % len(base_colors)]
        drawing.add(String(left_label_x, y - 4, f"{car_num} {full_name}", fontName="Helvetica-Bold", fontSize=8, fillColor=color))

    for fin_idx, driver in enumerate(finishing_order):
        fin_rank = fin_idx + 1
        y = y_for_rank(fin_rank)
        full_name = driver_lap_data[driver]['full_name']
        try:
            color_idx = starting_order.index(driver)
            color = base_colors[color_idx % len(base_colors)]
        except ValueError:
            color = black
        drawing.add(String(right_label_x, y - 4, f"{fin_rank} {full_name}", fontName="Helvetica-Bold", fontSize=8, fillColor=color))

    elements.append(drawing)
    elements.append(PageBreak())

# ---------------- Verbose log (always included in PDF) ----------------
if verbose_log:
    elements.append(Paragraph("Drivers Events/Incidents Log", styles["Heading2"]))
    elements.append(Spacer(1, 8))
    for driver_display, logs in verbose_log.items():
        elements.append(Paragraph(driver_display, styles["Heading3"]))
        for line in logs:
            elements.append(Paragraph(line, left_style))
        elements.append(Spacer(1, 8))

# ---------------- Build PDF ----------------
doc.build(elements)
print(f"PDF written to {pdf_path}", file=sys.stderr)

# Print verbose log to stdout if requested
if verbose and verbose_log:
    # Print summary to stdout
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("No summary available.")

    print("\n--- Race Events Log ---")
    for driver, events in verbose_log.items():
        print(f"\n{driver}:")
        for event in events:
            print(f"  {event}")