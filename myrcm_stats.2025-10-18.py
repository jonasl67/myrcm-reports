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

# Driver Performance Index (DPI) weights
DPI_WEIGHTS = {
    "Speed": {
        "weight": 0.4,
        "components": {
            "Best Lap": 0.2,
            "% <3% Laps": 0.5,
            "90% Lap": 0.3,
        },
    },
    "Consistency": {
        "weight": 0.3,
        "components": {
            "Consistency %": 0.40,
            "Clean Lap %": 0.20,
            "# Incidents": 0.15,
            "Time Lost Incidents": 0.15,
            "Fade Resistance": 0.10,
        },
    },
    "Racecraft": {
        "weight": 0.3,
        "components": {
            "Overtakes": 0.25,
            "Losses": 0.25,
            "Overtake Eff": 0.25,
            "Passed Eff": 0.15,
            "Lapping Eff": 0.10,
        },
    },
}

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

def looks_like_fuel(d):
    # Decide whether the time delta passed on looks like a fuel pit stop
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

def detect_overtakes(driver, lap_positions, lap_times, driver_fuel_stops, driver_major_incidents, median_clean, debug_log=None):
    """Detect on-track overtakes (true passes not caused by pit stops or incidents)."""
    if debug_log is None:
        debug_log = []

    overtakes = 0
    overtaken_drivers = []
    overtake_deltas = []

    my_positions = lap_positions.get(driver, [])
    n_laps = len(my_positions)

    print(f"🔍 {driver} driver_fuel_stops: {driver_fuel_stops}")
    print(f"🔍 {driver} driver_major_incidents: {driver_major_incidents}")

    for opponent, opp_positions in lap_positions.items():
        if opponent == driver:
            continue

        max_lap = min(n_laps, len(opp_positions))
        for i in range(1, max_lap):
            pos_prev_d = my_positions[i - 1]
            pos_curr_d = my_positions[i]
            pos_prev_o = opp_positions[i - 1]
            pos_curr_o = opp_positions[i]

            if None in (pos_prev_d, pos_curr_d, pos_prev_o, pos_curr_o):
                continue

            # skip laps with fueling or major incident
            if (
                i in driver_fuel_stops.get(driver, [])
                or i in driver_major_incidents.get(driver, [])
                or i in driver_fuel_stops.get(opponent, [])
                or i in driver_major_incidents.get(opponent, [])
            ):
                continue

            # detect true on-track overtake
            if pos_curr_d < pos_prev_d and pos_curr_o > pos_prev_o and pos_curr_d < pos_curr_o:
                overtakes += 1
                overtaken_drivers.append(opponent)

                if i < len(lap_times):
                    delta = max(0.0, lap_times[i] - median_clean)
                    overtake_deltas.append(delta)

                debug_log.append(
                    f"Lap {i+1}: {driver} overtook {opponent} "
                    f"(Δ {pos_prev_d}->{pos_curr_d} vs {pos_prev_o}->{pos_curr_o})"
                )

    avg_time_lost = float(np.mean(overtake_deltas)) if overtake_deltas else 0.0
    return overtakes, overtaken_drivers, avg_time_lost, debug_log

def detect_losses(driver, lap_positions, lap_times, driver_fuel_stops, driver_major_incidents, median_clean, debug_log=None):
    """Detect on-track losses (being passed, not due to pit stops or incidents)."""
    if debug_log is None:
        debug_log = []

    losses = 0
    lost_to_drivers = []
    loss_deltas = []

    my_positions = lap_positions.get(driver, [])
    n_laps = len(my_positions)

    for opponent, opp_positions in lap_positions.items():
        if opponent == driver:
            continue

        max_lap = min(n_laps, len(opp_positions))
        for i in range(1, max_lap):
            pos_prev_d = my_positions[i - 1]
            pos_curr_d = my_positions[i]
            pos_prev_o = opp_positions[i - 1]
            pos_curr_o = opp_positions[i]

            if None in (pos_prev_d, pos_curr_d, pos_prev_o, pos_curr_o):
                continue

            if (
                i in driver_fuel_stops.get(driver, [])
                or i in driver_major_incidents.get(driver, [])
                or i in driver_fuel_stops.get(opponent, [])
                or i in driver_major_incidents.get(opponent, [])
            ):
                continue

            if pos_curr_d > pos_prev_d and pos_curr_o < pos_prev_o and pos_curr_d > pos_curr_o:
                losses += 1
                lost_to_drivers.append(opponent)

                if i < len(lap_times):
                    delta = max(0.0, lap_times[i] - median_clean)
                    loss_deltas.append(delta)

                debug_log.append(
                    f"Lap {i+1}: {driver} was passed by {opponent} "
                    f"(Δ {pos_prev_d}->{pos_curr_d} vs {pos_prev_o}->{pos_curr_o})"
                )

    avg_time_lost = float(np.mean(loss_deltas)) if loss_deltas else 0.0
    return losses, lost_to_drivers, avg_time_lost, debug_log

def analyze_overtakes_and_losses(driver, lap_positions, lap_times, driver_fuel_stops, driver_major_incidents):
    """Wrapper combining overtakes and losses, returning a single structured dict."""
    median_clean = compute_clean_lap_time(lap_times)
    debug_log = []

    overtakes, overtaken_drivers, avg_loss_ot, log_ot = detect_overtakes(
        driver, lap_positions, lap_times, driver_fuel_stops, driver_major_incidents, median_clean, debug_log
    )
    losses, lost_to_drivers, avg_loss_ls, log_ls = detect_losses(
        driver, lap_positions, lap_times, driver_fuel_stops, driver_major_incidents, median_clean, debug_log
    )

    debug_log.extend(log_ot)
    debug_log.extend(log_ls)

    return {
        "overtakes": overtakes,
        "losses": losses,
        "avg_time_lost_overtakes": avg_loss_ot,
        "avg_time_lost_losses": avg_loss_ls,
        "overtaken_drivers": overtaken_drivers,
        "lost_to_drivers": lost_to_drivers,
        "debug_log": debug_log,
    }

def analyze_lapping_events(driver, lap_positions, lap_times, driver_lap_data,
                           fuel_stops=None, major_incidents=None):
    """
    Identify when this driver laps or gets lapped by others,
    excluding laps affected by fueling or major incidents (for either driver).

    Args:
        driver (str): The driver to analyze.
        lap_positions (dict[str, list[int | None]]): Track positions per driver per lap.
        lap_times (np.ndarray): Array of lap times (seconds) for this driver.
        driver_lap_data (dict[str, dict]): Master data structure with 'no_of_laps' for all drivers.
        fuel_stops (list[int]): Lap indices where this driver fueled.
        major_incidents (list[int]): Lap indices with major incidents.

    Returns:
        dict with keys:
            lappings (int)
            lapped_by (int)
            avg_time_lost_lappings (float)
            avg_time_lost_lapped_by (float)
    """
    fuel_stops = set(fuel_stops or [])
    major_incidents = set(major_incidents or [])
    driver_laps = driver_lap_data.get(driver, {}).get("no_of_laps", len(lap_positions.get(driver, [])))
    median_clean = compute_clean_lap_time(lap_times)

    lappings = 0
    lapped_by = 0
    lapping_deltas = []
    lapped_by_deltas = []

    # For each other driver, compare lap counts
    for other_driver, other_data in driver_lap_data.items():
        if other_driver == driver:
            continue

        other_laps = other_data.get("no_of_laps", len(lap_positions.get(other_driver, [])))

        # Skip drivers with no position data
        other_positions = lap_positions.get(other_driver, [])
        if not other_positions:
            continue

        # Determine laps where both drivers are valid and not in pit/major incident
        valid_laps = [
            i for i in range(min(driver_laps, len(other_positions)))
            if i not in fuel_stops
            and i not in major_incidents
            and other_positions[i] is not None
        ]
        if not valid_laps:
            continue

        lap_diff = driver_laps - other_laps

        if lap_diff >= 1:
            lappings += 1
            lapping_deltas.append(median_clean * 0.02 * lap_diff)  # ~2% per lapped car
        elif lap_diff <= -1:
            lapped_by += 1
            lapped_by_deltas.append(median_clean * 0.05 * abs(lap_diff))  # ~5% penalty

    avg_time_lost_lappings = float(np.mean(lapping_deltas)) if lapping_deltas else 0.0
    avg_time_lost_lapped_by = float(np.mean(lapped_by_deltas)) if lapped_by_deltas else 0.0

    return {
        "lappings": lappings,
        "lapped_by": lapped_by,
        "avg_time_lost_lappings": avg_time_lost_lappings,
        "avg_time_lost_lapped_by": avg_time_lost_lapped_by,
    }

def normalize(value, min_val=-5, max_val=5):
    """
    Normalize a numeric value into the 0–1 range.
    Values below min_val → 0, above max_val → 1.
    """
    if value <= min_val:
        return 0.0
    if value >= max_val:
        return 1.0
    return (value - min_val) / float(max_val - min_val)

def safe_div(a, b, default=0.0):
    """Return a / b but handle division by zero and NaN gracefully."""
    if b is None or b == 0 or np.isnan(b):
        return default
    return a / b

def compute_driver_performance_indices(driver_lap_data, global_hot_lap, lap_positions):
    """ Compute the Driver Performance Index (DPI) and all sub-domain metrics for all drivers. """

    # --- Establish reference laps for normalization ---
    class_ref = {
        "best_lap_ref": global_hot_lap or 1.0,
        "p90_ref": min(
            (d["median_clean"] for d in driver_lap_data.values() if d.get("median_clean")),
            default=global_hot_lap or 1.0
        ),
    }

    dpi_results = []

    for driver, data in driver_lap_data.items():

        # --- Optional: compute Racecraft metrics on the fly ---
        if lap_positions is not None:
            lap_times = data.get("lap_times", np.array([]))
            fuel_stops = data.get("fuel_stops_idx", [])
            major_incidents = data.get("major_incidents", [])

            data["racecraft_basic"] = analyze_overtakes_and_losses(
                driver, lap_positions, lap_times,
                {driver: fuel_stops}, {driver: major_incidents}
            )
            data["racecraft_lap"] = analyze_lapping_events(
                driver, lap_positions, lap_times, driver_lap_data,
                fuel_stops=fuel_stops, major_incidents=major_incidents
            )

        # --- Compute the driver’s DPI ---
        dpi, breakdown = compute_driver_performance_index(data, class_ref)
        full_name = data.get("full_name", driver)
        car_no = data.get("car_number", "?")

        dpi_results.append({
            "Driver": full_name,
            "Car": car_no,
            "Speed": breakdown["speed"],
            "Consistency": breakdown["consistency"],
            "Racecraft": breakdown["racecraft"],
            "DPI": dpi,
            "Components": breakdown.get("components", {}),
        })

        # --- Store back into the driver’s data ---
        data["DPI"] = dpi
        data.update(breakdown)

    return dpi_results

def add_driver_performance_pages(elements, dpi_results):
    """
    Adds a Driver Performance Index (DPI) summary and details pages to the PDF.
    Includes:
      - Main DPI summary (Speed / Consistency / Racecraft / Overall DPI)
      - Subcomponent breakdown table below
    """

    styles = getSampleStyleSheet()
    center_bold = ParagraphStyle(
        "CenterBold",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold"
    )

    # === Page Header ===
    elements.append(PageBreak())
    elements.append(Paragraph("Drivers Performance Index (DPI)", styles["Heading1"]))
    elements.append(Spacer(1, 12))

    # === DPI SUMMARY TABLE ===
    summary_data = [
        ["Pos", "Driver", "Car", "Overall DPI", "Speed", "Consistency", "Racecraft"],
        [
            "", "", "", "Weights",
            f"{DPI_WEIGHTS['Speed']['weight']*100:.0f}%",
            f"{DPI_WEIGHTS['Consistency']['weight']*100:.0f}%",
            f"{DPI_WEIGHTS['Racecraft']['weight']*100:.0f}%",
        ],
    ]

    for pos, r in enumerate(dpi_results, start=1):
        summary_data.append([
            pos,
            r["Driver"],
            r["Car"],
            f"{r['DPI']:.1f}",
            f"{r['Speed']:.1f}",
            f"{r['Consistency']:.1f}",
            f"{r['Racecraft']:.1f}",
        ])

    summary_table = Table(summary_data, repeatRows=2)
    summary_table.setStyle(TableStyle([
        # Header row
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),

        # Weight row
        ('BACKGROUND', (0, 1), (-1, 1), colors.whitesmoke),
        ('TEXTCOLOR', (0, 1), (-1, 1), colors.darkgrey),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Oblique'),
        ('ALIGN', (0, 1), (-1, 1), 'CENTER'),
        ('FONTSIZE', (0, 1), (-1, 1), 7),

        # Data rows
        ('ALIGN', (0, 2), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
        ('FONTSIZE', (0, 2), (-1, -1), 8),
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 20))
    elements.append(PageBreak())

    # === COMPONENT DETAIL TABLE ===
    # Group header row + updated consistency labels
    comp_data = [
        # Row 0: Group headers
        ["", "Speed", "", "", "Consistency", "", "", "", "", "Racecraft", "", "", "", ""],
        # Row 1: Labels
        [
            "Driver",
            "Best Lap",
            "Within 3% Laps",
            "90% Laps",
            "Consistency",
            "Clean Laps",
            "Incident avoidance",
            "Incident Time Lost",
            "Fade Resistance",
            "Overtakes",
            "Losses",
            "Overtake Eff",
            "Passed Eff",
            "Lapping Eff",
        ],
        # Row 2: Weight values
        [
            "Weight",
            *[f"{v:.2f}" for v in DPI_WEIGHTS["Speed"]["components"].values()],
            *[f"{v:.2f}" for v in DPI_WEIGHTS["Consistency"]["components"].values()],
            *[f"{v:.2f}" for v in DPI_WEIGHTS["Racecraft"]["components"].values()],
        ],
    ]

    # --- Data rows ---
    for r in dpi_results:
        c = r.get("Components", r.get("components", {}))
        comp_data.append([
            r["Driver"],

            # --- Speed (best_lap in seconds, within_3pct as % already, p90_lap in seconds) ---
            f"{c.get('best_lap', 0):.3f}",
            f"{c.get('laps_within_3pct', 0):.1f}%",
            f"{c.get('p90_lap', 0):.3f}",

            # --- Consistency (consistency_pct & clean_lap_ratio already in %) ---
            f"{c.get('consistency_pct', 0):.1f}%",
            f"{c.get('clean_lap_ratio', 0):.1f}%",
            f"{c.get('incident_density_score', 0) * 100:.1f}%",
            f"{c.get('time_lost_score', 0) * 100:.1f}%",
            f"{c.get('fade_score', 0) * 100:.1f}%",

            # --- Racecraft (eff_* are 0–1 → show as %) ---
            c.get("overtakes", 0),
            c.get("losses", 0),
            f"{c.get('eff_over', 0) * 100:.1f}%",
            f"{c.get('eff_lost', 0) * 100:.1f}%",
            f"{c.get('eff_lapping', 0) * 100:.1f}%",
        ])

    comp_table = Table(comp_data, repeatRows=3)
    comp_table.setStyle(TableStyle([
        # === Group header spanning (row 0) ===
        ('SPAN', (1, 0), (3, 0)),   # Speed (3 cols)
        ('SPAN', (4, 0), (8, 0)),   # Consistency (5 cols)
        ('SPAN', (9, 0), (13, 0)),  # Racecraft (5 cols)

        # === Group header styling ===
        ('BACKGROUND', (1, 0), (13, 0), colors.lightgrey),
        ('TEXTCOLOR', (1, 0), (13, 0), colors.black),
        ('FONTNAME', (1, 0), (13, 0), 'Helvetica-Bold'),
        ('ALIGN', (1, 0), (13, 0), 'CENTER'),
        ('VALIGN', (1, 0), (13, 0), 'MIDDLE'),
        ('FONTSIZE', (1, 0), (13, 0), 9),
        ('BOTTOMPADDING', (1, 0), (13, 0), 6),
        ('TOPPADDING', (1, 0), (13, 0), 6),

        # === Group separators ===
        ('LINEAFTER', (3, 0), (3, -1), 0.5, colors.black),  # after Speed
        ('LINEAFTER', (8, 0), (8, -1), 0.5, colors.black),  # after Consistency

        # === Column label row (row 1) ===
        ('BACKGROUND', (0, 1), (-1, 1), colors.grey),
        ('TEXTCOLOR', (0, 1), (-1, 1), colors.whitesmoke),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('ALIGN', (0, 1), (-1, 1), 'CENTER'),
        ('FONTSIZE', (0, 1), (-1, 1), 7),

        # === Weight row (row 2) ===
        ('BACKGROUND', (0, 2), (-1, 2), colors.whitesmoke),
        ('TEXTCOLOR', (0, 2), (-1, 2), colors.darkgrey),
        ('FONTNAME', (0, 2), (-1, 2), 'Helvetica-Oblique'),
        ('ALIGN', (0, 2), (-1, 2), 'CENTER'),
        ('FONTSIZE', (0, 2), (-1, 2), 6),

        # === Data body ===
        ('ALIGN', (0, 3), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
        ('FONTSIZE', (0, 3), (-1, -1), 7),
    ]))

    elements.append(Paragraph("Component breakdown", styles["Heading3"]))
    elements.append(Spacer(1, 6))
    elements.append(comp_table)

def compute_speed_index(driver_data, class_ref):
    """
    Compute Speed Index and breakdown.
    Returns (score, breakdown)
    """

    SWC = DPI_WEIGHTS["Speed"]["components"]

    best_lap = driver_data.get("best_lap", np.nan)
    p90_lap = driver_data.get("median_clean", np.nan)
    best_ref = class_ref.get("best_lap_ref", best_lap)
    p90_ref = class_ref.get("p90_ref", p90_lap)

    # Ratios relative to class reference
    best_ratio = best_ref / best_lap if best_lap > 0 else 0
    p90_ratio = p90_ref / p90_lap if p90_lap > 0 else 0
    laps_within_3pct = driver_data.get("laps_within_3pct", 0) / 100.0

    # Weighted normalized speed score
    speed_index = (
        SWC["Best Lap"]   * best_ratio +
        SWC["90% Lap"]    * p90_ratio +
        SWC["% <3% Laps"] * laps_within_3pct
    )
    speed_index = max(0.0, min(speed_index, 1.0))

    breakdown = {
        "best_lap": best_lap,
        "best_ratio": best_ratio,
        "p90_lap": p90_lap,
        "p90_ratio": p90_ratio,
        "laps_within_3pct": laps_within_3pct * 100.0,
    }

    return speed_index, breakdown

def compute_consistency_index(driver_data):
    """
    Compute Consistency Index and breakdown.
    Returns (score, breakdown)
    """

    CWC = DPI_WEIGHTS["Consistency"]["components"]

    consistency_pct = driver_data.get("consistency_pct", 0.0) / 100.0
    clean_lap_ratio = driver_data.get("clean_lap_ratio", 0.0)
    fade_resistance = driver_data.get("fade_resistance", 1.0)
    num_incidents = driver_data.get("num_incidents", 0)
    time_lost_incidents = driver_data.get("time_lost_incidents", 0.0)
    total_raced_time = driver_data.get("total_raced_time", 1.0)

    # --- Normalized scores ---
    fade_score = score_fade_resistance_linear(fade_resistance)
    time_lost_score = score_time_lost_incidents_linear(time_lost_incidents, total_raced_time)
    incident_density_score = score_incident_density_linear(num_incidents, total_raced_time)

    # --- Weighted sum ---
    consistency_index = (
        CWC["Consistency %"]      * consistency_pct +
        CWC["Clean Lap %"]        * clean_lap_ratio +
        CWC["Fade Resistance"]    * fade_score +
        CWC["# Incidents"]        * incident_density_score +
        CWC["Time Lost Incidents"]* time_lost_score
    )
    consistency_index = max(0.0, min(consistency_index, 1.0))

    breakdown = {
        "consistency_pct": consistency_pct * 100.0,
        "clean_lap_ratio": clean_lap_ratio * 100.0,
        "fade_resistance": fade_resistance,
        "fade_score": fade_score,
        "num_incidents": num_incidents,
        "incident_density_score": incident_density_score,
        "time_lost_incidents": time_lost_incidents,
        "time_lost_score": time_lost_score,
        "total_raced_time": total_raced_time,
    }

    return consistency_index, breakdown


    """
    Compute the Driver Performance Index (DPI) and its category breakdown.
    Returns:
      total_dpi, breakdown_dict
    """

    # --- Fetch and normalize top-level weights ---
    SW = float(DPI_WEIGHTS["Speed"]["weight"])
    CW = float(DPI_WEIGHTS["Consistency"]["weight"])
    RW = float(DPI_WEIGHTS["Racecraft"]["weight"])
    tw = SW + CW + RW
    if tw > 0:
        SW, CW, RW = SW / tw, CW / tw, RW / tw

    # --- Compute domain scores using dedicated functions ---
    speed_index = compute_speed_index(driver_data, class_ref)
    consistency_index = max(0.0, min(compute_consistency_index(driver_data), 1.0))
    racecraft_index = compute_racecraft_index(driver_data)

    # --- Combine to total DPI ---
    total_dpi = 100.0 * (SW * speed_index + CW * consistency_index + RW * racecraft_index)

    # === COMPONENT BREAKDOWN ===
    rc_basic = driver_data.get("racecraft_basic", {}) or {}
    rc_lap   = driver_data.get("racecraft_lap", {})   or {}

    breakdown = {
        "speed": round(speed_index * 100, 1),
        "consistency": round(consistency_index * 100, 1),
        "racecraft": round(racecraft_index * 100, 1),
        "components": {
            # --- Speed components ---
            "best_lap": driver_data.get("best_lap", 0.0),
            "laps_3pct": driver_data.get("laps_within_3pct", 0.0),
            "p90": driver_data.get("median_clean", 0.0),

            # --- Consistency components ---
            "consistency_pct": driver_data.get("consistency_pct", 0.0),
            "clean_lap_ratio": (driver_data.get("clean_lap_ratio", 0.0) or 0.0) * 100.0,
            "fade_resistance": driver_data.get("fade_resistance", 1.0),
            "num_incidents": driver_data.get("num_incidents", 0),
            "time_lost_incidents_pct": 100.0 * safe_div(driver_data.get("time_lost_incidents", 0.0), driver_data.get("total_raced_time", 1.0)),

            # --- Racecraft components ---
            "overtakes": rc_basic.get("overtakes", 0),
            "losses": rc_basic.get("losses", 0),
            "eff_over": 1 - safe_div(rc_basic.get("avg_time_lost_overtakes", 0.0), driver_data.get("median_clean", 1.0)),
            "eff_lost": 1 - safe_div(rc_basic.get("avg_time_lost_losses", 0.0), driver_data.get("median_clean", 1.0)),
            "eff_lapping": 1 - safe_div(rc_lap.get("avg_time_lost_lappings", 0.0), driver_data.get("median_clean", 1.0)),
            "avg_loss_overtakes": rc_basic.get("avg_time_lost_overtakes", 0.0),
            "avg_loss_losses": rc_basic.get("avg_time_lost_losses", 0.0),
            "avg_loss_lappings": rc_lap.get("avg_time_lost_lappings", 0.0),
            "median_clean": driver_data.get("median_clean", 1.0),
        },
    }

    return total_dpi, breakdown

def compute_driver_performance_index(driver_data, class_ref):
    """
    Compute the Driver Performance Index (DPI) and its category breakdown.
    Returns total_dpi, breakdown_dict
    """

    # --- Fetch and normalize top-level weights ---
    SW = float(DPI_WEIGHTS["Speed"]["weight"])
    CW = float(DPI_WEIGHTS["Consistency"]["weight"])
    RW = float(DPI_WEIGHTS["Racecraft"]["weight"])
    total_w = SW + CW + RW
    SW, CW, RW = (SW / total_w, CW / total_w, RW / total_w)

    # --- Get each domain's score and breakdown ---
    speed_score, speed_breakdown = compute_speed_index(driver_data, class_ref)
    consistency_score, consistency_breakdown = compute_consistency_index(driver_data)
    racecraft_score, racecraft_breakdown = compute_racecraft_index(driver_data)

    # --- Combine into total DPI ---
    total_dpi = 100.0 * (SW * speed_score + CW * consistency_score + RW * racecraft_score)

    # --- Merge breakdowns ---
    breakdown = {
        "speed": round(speed_score * 100, 1),
        "consistency": round(consistency_score * 100, 1),
        "racecraft": round(racecraft_score * 100, 1),
        "components": {
            **speed_breakdown,
            **consistency_breakdown,
            **racecraft_breakdown,
        },
    }

    return total_dpi, breakdown

def compute_racecraft_index(driver_data):
    """
    Compute Racecraft Index and breakdown.
    Returns (score, breakdown)
    """

    RWC = DPI_WEIGHTS["Racecraft"]["components"]
    rc_basic = driver_data.get("racecraft_basic", {}) or {}
    rc_lap   = driver_data.get("racecraft_lap", {})   or {}

    overtakes = rc_basic.get("overtakes", 0)
    losses = rc_basic.get("losses", 0)
    avg_loss_overtakes = rc_basic.get("avg_time_lost_overtakes", 0.0)
    avg_loss_losses = rc_basic.get("avg_time_lost_losses", 0.0)
    avg_loss_lappings = rc_lap.get("avg_time_lost_lappings", 0.0)
    median_clean = driver_data.get("median_clean", 1.0)

    # Normalized efficiencies (1.0 = perfect, 0.0 = poor)
    eff_over = 1 - safe_div(avg_loss_overtakes, median_clean)
    eff_lost = 1 - safe_div(avg_loss_losses, median_clean)
    eff_lapping = 1 - safe_div(avg_loss_lappings, median_clean)

    # Clamp all between 0–1
    eff_over = max(0.0, min(eff_over, 1.0))
    eff_lost = max(0.0, min(eff_lost, 1.0))
    eff_lapping = max(0.0, min(eff_lapping, 1.0))

    # Weighted combination
    racecraft_index = (
        RWC["Overtakes"]     * normalize(overtakes - losses, -5, 5) +
        RWC["Losses"]        * (1 - normalize(losses, 0, 5)) +
        RWC["Overtake Eff"]  * eff_over +
        RWC["Passed Eff"]    * eff_lost +
        RWC["Lapping Eff"]   * eff_lapping
    )
    racecraft_index = max(0.0, min(racecraft_index, 1.0))

    breakdown = {
        "overtakes": overtakes,
        "losses": losses,
        "eff_over": eff_over,
        "eff_lost": eff_lost,
        "eff_lapping": eff_lapping,
        "avg_loss_overtakes": avg_loss_overtakes,
        "avg_loss_losses": avg_loss_losses,
        "avg_loss_lappings": avg_loss_lappings,
        "median_clean": median_clean,
    }

    return racecraft_index, breakdown

def score_time_lost_incidents_linear(time_lost, total_raced_time, debug=False):
    """
    Convert time lost to incidents into a normalized score between 0.0–1.0.

    0% loss  -> score 1.0 (perfect)
    max_ratio loss -> score 0.0 (worst)
    Beyond max_ratio is clamped to 0.

    Adjust MAX_RATIO to control how strongly penalties apply.
    """
    # --- Configuration ---
    MAX_RATIO = 0.10   # 10% race time lost → score = 0 (tune as desired)
    DURATIONS = [300, 1200, 2700]  # for debug printing: 5min, 20min, 45min
    LOSS_PCTS = [0, 1, 2, 3, 5, 8, 10]

    # --- Core computation ---
    if total_raced_time <= 0:
        return 1.0
    ratio = time_lost / total_raced_time
    score = 1.0 - (ratio / MAX_RATIO)
    score = max(0.0, min(1.0, score))

    # --- Optional debug output ---
    if debug:
        print(f"\n=== Time-lost scoring (MAX_RATIO={MAX_RATIO*100:.0f}% → 0) ===")
        header = f"{'Lost %':>7} | {'Score':>5} | {'5min (s)':>10} | {'20min (s)':>10} | {'45min (s)':>10}"
        print(header)
        print("-" * len(header))
        for pct in LOSS_PCTS:
            r = pct / 100
            s = max(0, min(1, 1 - r / MAX_RATIO))
            row = f"{pct:7.1f} | {s:5.2f} |"
            for dur in DURATIONS:
                lost = dur * r
                row += f" {lost:10.0f} |"
            print(row)
        print("\nTip: Lower MAX_RATIO → harsher penalty, higher → softer.\n")

    return score

def score_incident_density_linear(num_incidents, total_raced_time, debug=False):
    """
    Convert incident density (incidents per race minute) into a normalized 0–1 score.

    - 0 incidents per minute -> 1.0 (perfect)
    - INC_PER_MIN_FOR_0_SCORE incidents/minute -> 0.0 (worst)
    - Linear falloff in between.

    Example: With INC_PER_MIN_FOR_0_SCORE = 1.0,
             1 incident per minute => score 0.0
             0.1 incidents per min (1 every 10min) => score 0.9
    """
    # --- Tunable parameters ---
    INC_PER_MIN_FOR_0_SCORE = 1.0   # 1 incident/minute → 0 score
    SAMPLE_DURATIONS = [300, 1200, 2700]  # 5, 20, 45 min (for debug)
    SAMPLE_INCIDENTS = list(range(0, 11))

    # --- Core computation ---
    if total_raced_time <= 0:
        return 1.0
    minutes = total_raced_time / 60.0
    inc_per_min = num_incidents / minutes
    score = 1.0 - (inc_per_min / INC_PER_MIN_FOR_0_SCORE)
    score = max(0.0, min(1.0, score))

    # --- Optional debug output ---
    if debug:
        print(f"\n=== Incident Density Scoring (≤{1/INC_PER_MIN_FOR_0_SCORE:.1f} incidents per min = 1.0 → 0.0) ===")
        header = f"{'Race (min)':>9} | {'Incidents':>9} | {'Per min':>7} | {'Score':>5}"
        print(header)
        print("-" * len(header))
        for dur in SAMPLE_DURATIONS:
            mins = dur / 60.0
            for inc in [0, 1, 2, 3, 4, 5]:
                ipm = inc / mins
                s = max(0, min(1, 1 - ipm / INC_PER_MIN_FOR_0_SCORE))
                print(f"{mins:9.1f} | {inc:9.0f} | {ipm:7.2f} | {s:5.2f}")
            print("-" * len(header))

    return score

def score_fade_resistance_linear(fade_ratio, debug=False):
    """
    Convert a fade ratio (last_avg / first_avg) into a normalized 0–1 score.

    - fade_ratio <= 1.0 → score = 1.0 (perfect, no fade or improvement)
    - fade_ratio > 1.0  → linearly decays toward 0
    - FADES_TO_ZERO_AT defines how harsh the penalty is (similar to time lost scale)

    Example: FADES_TO_ZERO_AT = 1.50 → 50% fade = score 0
    """
    # --- Tunable parameter ---
    FADES_TO_ZERO_AT = 1.50   # 50% slower → 0.0 score

    # --- Core computation ---
    if fade_ratio <= 1.0:
        score = 1.0
    else:
        score = 1.0 - ((fade_ratio - 1.0) / (FADES_TO_ZERO_AT - 1.0))
    score = max(0.0, min(1.0, score))

    # --- Optional debug output ---
    if debug:
        print(f"\n=== Fade Resistance Scoring (fade_ratio ≤ 1.0 = 1.0, {FADES_TO_ZERO_AT:.2f} = 0.0) ===")
        print(f"{'Fade Ratio':>10} | {'Score':>6}")
        print("-" * 22)
        for r in [0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30, 1.50]:
            s = 1.0 if r <= 1.0 else max(0, min(1, 1 - ((r - 1) / (FADES_TO_ZERO_AT - 1))))
            print(f"{r:10.2f} | {s:6.2f}")
        print("\nTip: Lower FADES_TO_ZERO_AT → harsher penalty.\n")

    return score

def compute_clean_lap_time_filtered(lap_times, fuel_stops_idx, minor_incidents, major_incidents):
    """
    Compute the average clean lap time excluding all incident and fuel laps.

    Falls back to percentile-based computation if too few clean laps remain.

    Args:
        lap_times (array-like): Lap times in seconds.
        fuel_stops_idx (list[int]): Lap indices with fuel stops.
        minor_incidents (list[int]): Lap indices with minor incidents.
        major_incidents (list[int]): Lap indices with major incidents.

    Returns:
        float: Median clean lap time (seconds).
    """
    if lap_times is None or len(lap_times) == 0:
        return 0.0

    # Merge all excluded lap indices into one set
    excluded = set(fuel_stops_idx or []) | set(minor_incidents or []) | set(major_incidents or [])

    try:
        clean_laps = [t for i, t in enumerate(lap_times) if i not in excluded and not np.isnan(t)]

        if len(clean_laps) >= 3:
            # Median of remaining laps — robust to single outliers
            return float(np.median(clean_laps))
        else:
            # Too few “clean” laps — fallback to statistical approach
            return compute_clean_lap_time(lap_times)

    except Exception as e:
        print(f"⚠️ compute_clean_lap_time_filtered() failed: {e}")
        return compute_clean_lap_time(lap_times)

def compute_clean_lap_time(lap_times, percentile=90, verbose=False):
    """
    Compute the median 'clean' lap time based on a percentile threshold.

    Args:
        lap_times (array-like): Lap times in seconds.
        percentile (float): Upper percentile threshold for filtering (default 90).
        verbose (bool): Print warnings if True.

    Returns:
        float: Median clean lap time in seconds (0.0 if invalid input).
    """
    if lap_times is None or len(lap_times) == 0:
        return 0.0

    try:
        lap_times = np.array(lap_times, dtype=float)
        lap_times = lap_times[~np.isnan(lap_times)]
        if lap_times.size == 0:
            return 0.0

        threshold = np.percentile(lap_times, percentile)
        clean_laps = lap_times[lap_times <= threshold]
        if clean_laps.size == 0:
            clean_laps = lap_times

        return float(np.median(clean_laps))

    except Exception as e:
        if verbose:
            print(f"Error: compute_clean_lap_time() failed: {e}")
        return 0.0

def compare_clean_lap_methods(clean_lap_percentile, clean_lap_incidents_masked):
    # --- Sanity check between statistical and incident-filtered clean lap, log a warning if they differ > 2% ---
    if clean_lap_percentile and clean_lap_incidents_masked:  # ensure both are nonzero / valid
        diff = abs(clean_lap_incidents_masked - clean_lap_percentile)
        rel_diff = diff / clean_lap_percentile

        if rel_diff > 0.02:  # >2% difference
            print(f"Warning: Clean lap mismatch for {driver_name}: "
                f"prelim={clean_lap_percentile:.3f}s vs filtered={clean_lap_incidents_masked:.3f}s "
                f"(Δ={diff:.3f}s, {rel_diff*100:.1f}%)")
        else:
            #print(f"Clean lap OK for {driver_name}: Δ={diff:.3f}s ({rel_diff*100:.1f}%)")
            pass
    else:
        print(f"Warning: Clean lap check skipped for {driver_name} (missing data)")

# ===================== CLI / main() / entry point =======================

# ================================================================
#                        ENTRY POINT
# ================================================================

def main(csv_file: str, pdf_file: str | None = None, verbose: bool = False, nofuel: bool = False):
    """
    Main entry point for generating the RC race stats PDF.
    Handles:
        - CSV parsing
        - Race data analysis
        - Driver stats and DPI computation
        - PDF report generation
    """

    # --- Set global config flags ---
    global NOFUEL_MODE
    NOFUEL_MODE = nofuel

    # === STEP 1: Load and parse CSV file ===
    try:
        with open(csv_file, newline='') as fh:
            reader = csv.reader(fh)
            all_lines = list(reader)
    except FileNotFoundError:
        print(f"File not found: {csv_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Failed to read file: {e}", file=sys.stderr)
        sys.exit(1)


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

    # ---------------- Determine starting and finishing orders ----------------
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


    # Per-driver analysis (iterate in finishing order so Pos aligns with summary) ----------------
    results = []
    verbose_log = {}
    driver_fuel_stops = {}
    driver_major_incidents = {}
    global_overtakes = 0 # temporary debug only
    global_losses = 0 # temporary debug only

    # --- loop and analyze all drivers and all their laps 
    for pos_idx, driver in enumerate(finishing_order, start=1):
        data = driver_lap_data[driver]
        lap_times = data["lap_times"]
        full_name = data["full_name"]
        car_no = data["car_number"]
        n_laps = data["no_of_laps"]
        
        total_raced_time = float(np.sum(lap_times)) if n_laps > 0 else 0.0

        # preliminary clean lap purely based on median of 90 percentile laps
        median_clean_lap = compute_clean_lap_time(lap_times)
        #print(f"Preliminary clean lap={median_clean_lap}")

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
        
        # loop all laps for this driver to detect and classification of events ------------
        last_event = None
        for i, laptime in enumerate(lap_times):
            prev_time = current_time
            current_time += laptime
            delta = laptime - median_clean_lap # delta in lap time, used to detect an event(ful) lap
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
                    time_of_fueling = prev_time + (median_clean_lap / 2.0)
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

        # All laps analyzed, build the summary for this driver ---------------------------------------
        driver_fuel_stops[driver] = fuel_stops_idx
        driver_major_incidents[driver] = major_incidents

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
                    fuel_times.append(max(0.0, accum - laptime + median_clean_lap/2.0))

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

        # --- Add metrics needed for DPI computation ---
        if n_laps > 4:
            q25 = int(n_laps * 0.25)
            early_avg = np.mean(lap_times[:q25])
            late_avg = np.mean(lap_times[-q25:])
            fade_resistance = late_avg / early_avg if early_avg > 0 else 1.0
        else:
            fade_resistance = 1.0

        # Clean lap ratio = fraction of laps below 90th percentile
        try:
            clean_threshold = np.percentile(lap_times, 90)
            clean_laps = np.sum(lap_times <= clean_threshold)
            clean_lap_ratio = clean_laps / len(lap_times)
        except Exception:
            clean_lap_ratio = 0.0

        # save data needed for driver performance analysis
        driver_lap_data[driver].update({
            "best_lap": best_lap,
            "laps_within_3pct": laps_within_3pct,
            "consistency_pct": consistency,
            "clean_lap_ratio": clean_lap_ratio,
            "fade_resistance": fade_resistance,
            "median_clean": compute_clean_lap_time(lap_times),
            "fuel_stops_idx": fuel_stops_idx,
            "major_incidents": major_incidents,
            "num_incidents": total_incident_laps,
            "time_lost_incidents": total_incident_time_lost,
            "total_raced_time": total_raced_time
        })

        results.append({
            "Pos": pos_idx,
            "Driver (No)": display,
            "Laps": n_laps,
            "Race Time": format_time_mm_ss_decimal(total_raced_time),
            "Best Lap": format_time_ss_decimal(best_lap),
            "90 percentile lap": f"{median_clean_lap:.2f}" if median_clean_lap else "0.00",
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

        filtered_clean_lap = compute_clean_lap_time_filtered(lap_times, fuel_stops_idx, minor_incidents, major_incidents)
        
        # check that the 90% median lap and the median clean lap based on laps without fueling and incidents are within 2%, or log a warning
        compare_clean_lap_methods(median_clean_lap, filtered_clean_lap)

        #--- end of per driver loop

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

    # --- Compute racecraft metrics for all drivers ---
    #for driver in finishing_order:
    #    compute_racecraft_index(driver_lap_data[driver])

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

    # --- Add Driver Performance Index page  ---
    dpi_results = compute_driver_performance_indices(driver_lap_data, global_hot_lap, lap_positions)
    add_driver_performance_pages(elements, dpi_results)
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

# ================================================================
#                     COMMAND-LINE HANDLER
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate RC race stats PDF from myrcm CSV"
    )
    parser.add_argument("csv_file", help="Input race CSV file")
    parser.add_argument("pdf_file", nargs="?", help="Output PDF file (optional)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging to stdout")
    parser.add_argument("--nofuel", action="store_true", help="Disable fuel stop classification (for electric classes)")

    args = parser.parse_args()

    try:
        main(
            csv_file=args.csv_file,
            pdf_file=args.pdf_file,
            verbose=args.verbose,
            nofuel=args.nofuel
        )
    except KeyboardInterrupt:
        print("\nAborted by user.", file=sys.stderr)
        sys.exit(130)

# Nytt hit

"""
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
"""