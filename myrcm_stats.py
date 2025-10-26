#!/usr/bin/env python3

import sys
import csv
import os
import re
import math
import pandas as pd
import numpy as np
import pathlib
import argparse
from dataclasses import dataclass, field
from typing import List, Optional

# ReportLab imports
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.graphics.shapes import Drawing, String, Line
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.lib.colors import black, red, green, blue, orange, violet, pink, gray, Color
from reportlab.lib.units import cm
import logging

# region logging setup
def setup_logging(level=logging.INFO):
    """Configure the global logger once at import time."""
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s"
    )
    return logging.getLogger(__name__)

logger = setup_logging()
# endregion

class Config:
    """Global configuration constants for race analysis."""
    # Incident thresholds
    EVENT_THRESHOLD = 3.0
    FUEL_MIN_LOSS = 4.5
    MAJOR_EVENT_THRESHOLD = 15.0

    FUEL_WINDOW = (230, 360)
    STANDARD_FUEL_STINT = 245

    # Mode flags
    NOFUEL_MODE = False

    # Cosmetic constants
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
                "Consistency %": 0.50,
                "# Incidents": 0.25,
                "Time Lost Incidents": 0.20,
                "Fade Resistance": 0.05,
            },
        },
        "Racecraft": {
            "weight": 0.3,
            "components": {
                "Overtakes": 0.30,
                "Losses": 0.25,
                "Overtake Eff": 0.20,
                "Passed Eff": 0.15,
                "Lapping Eff": 0.05,
                "Lapped Eff": 0.05,
            },
        },
    }

@dataclass
class DriverData:
    """Encapsulates all data and metrics for a single driver."""
    
    # === Core Identity ===
    driver_key: str
    full_name: str
    car_number: str
    no_of_laps: int
    
    # === Raw Data ===
    raw_entries: List
    lap_times: np.ndarray
    
    # === Position Tracking ===
    positions: List[Optional[int]] = field(default_factory=list)
    
    # === Lap Classification ===
    fuel_stops_idx: List[int] = field(default_factory=list)
    major_incidents: List[int] = field(default_factory=list)
    minor_incidents: List[int] = field(default_factory=list)
    
    # === Basic Metrics ===
    best_lap: float = 0.0
    median_clean: float = 0.0
    total_raced_time: float = 0.0
    
    # === Speed Metrics ===
    laps_within_3pct: float = 0.0
    
    # === Consistency Metrics ===
    consistency_pct: float = 0.0
    fade_resistance: float = 1.0
    num_incidents: int = 0
    time_lost_incidents: float = 0.0
    
    # === Racecraft Metrics ===
    racecraft_basic: dict = field(default_factory=dict)
    racecraft_lap: dict = field(default_factory=dict)
    overtake_score: float = 0.0
    loss_score: float = 1.0
    lappings_count: int = 0 
    lapped_by_count: int = 0
    
    # === DPI Components ===
    speed_index: float = 0.0
    consistency_index: float = 0.0
    racecraft_index: float = 0.0
    dpi: float = 0.0
    
    # === Additional Data ===
    cap_over: float = 1.0
    cap_loss: float = 1.0
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if len(self.lap_times) > 0:
            self.total_raced_time = float(np.sum(self.lap_times))
    
    @classmethod
    def from_csv_data(cls, driver_key: str, full_name: str, car_number: str, 
                      official_laps: int, raw_entries: List, parsed_times: List) -> 'DriverData':
        """Factory method to create DriverData from CSV parsing results."""
        return cls(
            driver_key=driver_key,
            full_name=full_name,
            car_number=car_number,
            no_of_laps=official_laps,
            raw_entries=raw_entries,
            lap_times=np.array(parsed_times)
        )
    
    def compute_best_lap(self, exclude_first: bool = True) -> float:
        """Calculate and store the best lap time."""
        if self.no_of_laps > 1 and exclude_first:
            valid_laps = self.lap_times[1:]
            self.best_lap = float(np.min(valid_laps)) if valid_laps.size else 0.0
        else:
            self.best_lap = float(np.min(self.lap_times)) if self.lap_times.size else 0.0
        return self.best_lap
    
    def compute_median_clean_lap(self, percentile: int = 90) -> float:
        """Calculate and store the median clean lap time."""
        self.median_clean = compute_clean_lap_time(self.lap_times, percentile)
        return self.median_clean
    
    def compute_consistency_metrics(self):
        """Calculate and store consistency percentage"""
        if self.no_of_laps > 1:
            lap_times_clean = self.lap_times[~np.isnan(self.lap_times)]
            if lap_times_clean.size > 1:
                mean_lap = np.mean(lap_times_clean)
                std_lap = np.std(lap_times_clean)
                if mean_lap > 0:
                    self.consistency_pct = max(0.0, (1.0 - (std_lap / mean_lap)) * 100.0)
    
    def compute_fade_resistance(self):
        """Calculate fade resistance (late lap avg / early lap avg)."""
        if self.no_of_laps > 4:
            q25 = int(self.no_of_laps * 0.25)
            early_avg = np.mean(self.lap_times[:q25])
            late_avg = np.mean(self.lap_times[-q25:])
            self.fade_resistance = late_avg / early_avg if early_avg > 0 else 1.0
            #print(f"{self.full_name} early laps: {early_avg} late laps: {late_avg} ratio: {self.fade_resistance}")
        else:
            self.fade_resistance = 1.0
    
    def compute_laps_within_3pct(self, global_hot_lap: float):
        """Calculate percentage of laps within 3% of global hot lap."""
        if global_hot_lap and self.no_of_laps > 1:
            laps_excluding_first = self.lap_times[1:]
            within_mask = laps_excluding_first <= (global_hot_lap * 1.03)
            self.laps_within_3pct = round(100.0 * np.sum(within_mask) / len(laps_excluding_first), 1)
        else:
            self.laps_within_3pct = 0.0
    
    def get_display_name(self) -> str:
        """Get formatted display name with car number."""
        return f"{self.full_name} ({self.car_number})"

def parse_duration_to_seconds(duration_str: str) -> int | None:
    """Convert 'MM:SS' or 'HH:MM:SS' into total seconds."""
    try:
        parts = duration_str.strip().split(":")
        if len(parts) == 2:
            m, s = map(int, parts)
            return m * 60 + s
        elif len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        else:
            return None
    except Exception:
        return None

def parse_lap_entry(entry):
    """Parse lap cell like '(2) 14.805' or '1:15.523' or '15.523'."""
    if pd.isna(entry):
        return np.nan
    s = str(entry).strip()
    if ')' in s:
        s = s.split(')')[-1].strip()
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
    """Extract the integer position inside parentheses like '(2) 03.863' -> 2."""
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

def format_time_ss_decimal(seconds):
    """Format as 'SS.ss' (no minutes)."""
    if seconds is None or (isinstance(seconds, float) and (math.isnan(seconds) or seconds < 0)):
        return "00.00"
    return f"{seconds:.2f}"

def safe_int(v, default=9999):
    try:
        return int(v)
    except Exception:
        return default

def safe_div(a, b, default=0.0):
    """Return a / b but handle division by zero and NaN gracefully."""
    if b is None or b == 0 or np.isnan(b):
        return default
    return a / b

def get_driver_at_position(lap_num, pos, lap_positions):
    """Helper function to find a driver's name based on their position for a given lap."""
    if lap_num < 1 or pos < 1:
        return None
    
    for driver_name, positions in lap_positions.items():
        lap_index = lap_num - 1
        if lap_index < len(positions):
            if positions[lap_index] == pos:
                return driver_name
    return None

def looks_like_fuel(delta_from_clean_lap, avg_fuel_delta):
    """Decide whether the time delta looks like a fuel pit stop."""
    if avg_fuel_delta > 0:
        return (delta_from_clean_lap >= Config.FUEL_MIN_LOSS * 0.9) and (delta_from_clean_lap <= avg_fuel_delta * 1.2)
    else:
        return delta_from_clean_lap >= Config.FUEL_MIN_LOSS

def award_trophies(df, column, higher_is_better=True, min_valid_value=0.0):
    """Add trophy symbols (¹, ², ³) to the top 3 entries in a given column."""
    if column not in df.columns:
        return df

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

    ranked = (
        valid.nlargest(3, numeric_col)
        if higher_is_better
        else valid.nsmallest(3, numeric_col)
    )

    for i, (idx, row) in enumerate(ranked.iterrows()):
        df.at[idx, column] = f"  {row[column]} {Config.TROPHIES[i]}"

    return df

def compute_clean_lap_time(lap_times, percentile=90, verbose=False):
    """Compute the median 'clean' lap time based on a percentile threshold."""
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

def load_race_metadata(csv_file: str) -> tuple[pd.DataFrame, dict]:
    """Read raw CSV, extract race metadata, and load the lap table."""
    try:
        with open(csv_file, newline='') as fh:
            reader = csv.reader(fh)
            all_lines = list(reader)
    except FileNotFoundError:
        logger.error("File not found: %s", csv_file)
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to read file: %s", e)
        sys.exit(1)

    lap_start_line = None
    raw_driver_info = []
    race_duration_sec = None
    race_event = race_name = race_details = ""

    if len(all_lines) > 0 and len(all_lines[0]) > 0:
        race_event = all_lines[0][0].strip()
    if len(all_lines) > 1 and len(all_lines[1]) > 0:
        race_name = all_lines[1][0].strip()
    if len(all_lines) > 2 and len(all_lines[2]) > 0:
        race_details = all_lines[2][0].strip()

    for i, row in enumerate(all_lines):
        for cell in row:
            if isinstance(cell, str) and "Race time:" in cell:
                try:
                    part = cell.split("Race time:")[-1].strip().split()[0]
                    race_duration_sec = parse_duration_to_seconds(part)
                    if race_duration_sec is None:
                        logger.warning("Could not parse race duration from '%s'", part)
                except Exception as e:
                    logger.warning("Failed parsing race duration: %s", e)

        if row and isinstance(row[0], str) and row[0].strip().lower().startswith("laptimes"):
            lap_start_line = i + 1
            break

    if lap_start_line is None:
        logger.error("Could not find 'Laptimes' start row in CSV.")
        sys.exit(1)

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

    try:
        df = pd.read_csv(csv_file, skiprows=lap_start_line)
    except Exception as e:
        logger.error("Error loading lap table with pandas.")
        sys.exit(1)

    header_names = df.columns[1:].tolist()

    # Auto-detect NOFUEL mode
    nofuel_mode = Config.NOFUEL_MODE
    if not nofuel_mode:
        lowered_event = (race_event or "").lower()
        if "1:12" in lowered_event or "fwd" in lowered_event:
            nofuel_mode = True
            logger.info("⚡ Auto-detected a no-refuel race → disabling fuel stop classification")
        elif race_duration_sec is not None and race_duration_sec <= 300:
            nofuel_mode = True
            logger.info("⚡ Auto-detected short race (%.0fs ≤ 300s) → disabling fuel stop classification", race_duration_sec)

    meta = {
        "race_event": race_event,
        "race_name": race_name,
        "race_details": race_details,
        "race_duration_sec": race_duration_sec,
        "lap_start_line": lap_start_line,
        "raw_driver_info": raw_driver_info,
        "header_names": header_names,
        "nofuel_mode": nofuel_mode,
        "csv_file": csv_file,
    }
    return df, meta

def build_driver_lap_data(df, meta):
    """Build driver_lap_data using DriverData class."""
    header_names = meta["header_names"]
    raw_driver_info = meta["raw_driver_info"]
    
    driver_lookup = {}
    for idx, driver in enumerate(header_names):
        if idx < len(raw_driver_info):
            driver_lookup[driver] = raw_driver_info[idx]
        else:
            driver_lookup[driver] = (driver, "?", 0)
    
    driver_data_dict = {}
    n_rows = df.shape[0]
    
    for col_idx, driver_key in enumerate(header_names, start=1):
        raw_entries = df.iloc[:, col_idx].tolist()
        parsed_times = []
        for e in raw_entries:
            t = parse_lap_entry(e)
            if not (isinstance(t, float) and math.isnan(t)):
                parsed_times.append(t)
        
        full_name, car_no, official_laps = driver_lookup.get(driver_key, (driver_key, "?", 0))
        
        driver_data_dict[driver_key] = DriverData.from_csv_data(
            driver_key=driver_key,
            full_name=full_name,
            car_number=car_no,
            official_laps=official_laps,
            raw_entries=raw_entries,
            parsed_times=parsed_times
        )
    
    all_valid_laps = []
    for driver_data in driver_data_dict.values():
        laps = driver_data.lap_times
        if len(laps) > 1:
            all_valid_laps.extend(laps[1:])
    global_hot_lap = min(all_valid_laps) if all_valid_laps else None
    
    valid_drivers = [d for d in header_names if not re.match(r'Unnamed: \d+', d)]
    starting_order = sorted(
        valid_drivers, 
        key=lambda d: safe_int(driver_data_dict[d].car_number, default=9999)
    )
    
    parsed_pos_by_driver = {}
    for driver_key in header_names:
        driver_data = driver_data_dict[driver_key]
        pos_list = []
        for row_idx in range(n_rows):
            if row_idx < len(driver_data.raw_entries):
                pos = parse_pos_from_entry(driver_data.raw_entries[row_idx])
                pos_list.append(pos)
            else:
                pos_list.append(None)
        parsed_pos_by_driver[driver_key] = pos_list
    
    lap_positions = {}
    for driver_key in starting_order:
        pos_list = parsed_pos_by_driver.get(driver_key, [])
        
        last_pos_idx = -1
        for i, p in enumerate(pos_list):
            if p is not None:
                last_pos_idx = i
        
        if last_pos_idx == -1:
            lap_positions[driver_key] = []
            continue
        
        series = []
        last_known = None
        start_slot = starting_order.index(driver_key) + 1
        
        for i in range(last_pos_idx + 1):
            p = pos_list[i]
            if p is not None:
                last_known = p
                series.append(p)
            else:
                series.append(last_known if last_known is not None else start_slot)
        
        lap_positions[driver_key] = series
        driver_data_dict[driver_key].positions = series
    
    return driver_data_dict, lap_positions, global_hot_lap

def classify_driver_events(driver, lap_positions, race_duration_sec):
    """Classify each lap as normal, fuel stop, or incident."""
    fuel_time_lost = 0.0
    major_time_lost = 0.0
    minor_time_lost = 0.0
    normal_fuel_stop_deltas = []
    
    current_time = 0.0
    last_fuel_time = 0.0
    last_event = None
    
    for i, laptime in enumerate(driver.lap_times):
        prev_time = current_time
        current_time += laptime
        delta = laptime - driver.median_clean
        event = None
        
        if Config.NOFUEL_MODE:
            if delta >= Config.MAJOR_EVENT_THRESHOLD:
                event = "MAJOR EVENT"
                driver.major_incidents.append(i)
                major_time_lost += max(0.0, delta)
            elif delta >= Config.EVENT_THRESHOLD:
                event = "MINOR EVENT"
                driver.minor_incidents.append(i)
                minor_time_lost += max(0.0, delta)
        else:
            avg_normal_fuel_lap_lost_time = np.median(normal_fuel_stop_deltas) if normal_fuel_stop_deltas else 0.0
            time_since_fuel = current_time - last_fuel_time
            avg_fuel_delta = avg_normal_fuel_lap_lost_time if avg_normal_fuel_lap_lost_time else 0.0
            
            if delta >= Config.MAJOR_EVENT_THRESHOLD:
                race_time_left = (race_duration_sec or 0) - current_time
                time_to_next_fuel = Config.STANDARD_FUEL_STINT - (current_time - last_fuel_time)
                
                event = "MAJOR EVENT + FUEL"
                do_fuel = True
                if race_time_left < time_to_next_fuel or current_time < 60:
                    event = "MAJOR EVENT"
                    do_fuel = False
                
                driver.major_incidents.append(i)
                major_time_lost += max(0.0, delta)
                if do_fuel:
                    driver.fuel_stops_idx.append(i)
            
            elif (
                (time_since_fuel >= Config.FUEL_WINDOW[0] and delta >= Config.FUEL_MIN_LOSS)
                or (time_since_fuel > Config.FUEL_WINDOW[1] and last_event == "MAJOR EVENT + FUEL" 
                    and avg_normal_fuel_lap_lost_time > 0 and looks_like_fuel(delta, avg_fuel_delta))
                or (((race_duration_sec or 0)-current_time) < Config.STANDARD_FUEL_STINT 
                    and time_since_fuel < Config.FUEL_WINDOW[0] and time_since_fuel > 120 
                    and avg_normal_fuel_lap_lost_time > 0 and looks_like_fuel(delta, avg_fuel_delta))
                or ((time_since_fuel > Config.FUEL_WINDOW[1]) and (delta >= (Config.FUEL_MIN_LOSS * 0.9)))
            ):
                event = "FUEL STOP"
                driver.fuel_stops_idx.append(i)
                fuel_time_lost += max(0.0, delta)
                normal_fuel_stop_deltas.append(delta)
            
            elif delta >= Config.EVENT_THRESHOLD:
                event = "MINOR EVENT"
                driver.minor_incidents.append(i)
                minor_time_lost += max(0.0, delta)
        
        if event:
            last_event = event
            if "FUEL" in event:
                time_of_fueling = prev_time + (driver.median_clean / 2.0)
                last_fuel_time = time_of_fueling
    
    driver.num_incidents = len(driver.minor_incidents) + len(driver.major_incidents)
    driver.time_lost_incidents = minor_time_lost + major_time_lost

def build_summary_row(driver, pos_idx):
    """Build a summary table row for a driver."""
    if Config.NOFUEL_MODE:
        avg_fuel_interval = 0
        avg_fuel_stop_time = 0
        fuel_time_lost = 0
    else:
        fuel_time_lost = 0.0
        normal_fuel_deltas = []
        current_time = 0.0
        for i, laptime in enumerate(driver.lap_times):
            current_time += laptime
            if i in driver.fuel_stops_idx:
                delta = laptime - driver.median_clean
                if i not in driver.major_incidents:
                    fuel_time_lost += max(0.0, delta)
                    normal_fuel_deltas.append(delta)
        
        avg_fuel_stop_time = (fuel_time_lost / len(normal_fuel_deltas)) if normal_fuel_deltas else 0.0
        
        fuel_times = []
        accum = 0.0
        for i, laptime in enumerate(driver.lap_times):
            accum += laptime
            if i in driver.fuel_stops_idx:
                fuel_times.append(max(0.0, accum - laptime + driver.median_clean/2.0))
        fuel_intervals = np.diff([0.0] + fuel_times) if fuel_times else np.array([])
        avg_fuel_interval = float(np.mean(fuel_intervals)) if fuel_intervals.size else 0.0
    
    return {
        "Pos": pos_idx,
        "Driver (No)": driver.get_display_name(),
        "Laps": driver.no_of_laps,
        "Race Time": format_time_mm_ss_decimal(driver.total_raced_time),
        "Best Lap": format_time_ss_decimal(driver.best_lap),# PART 3: Continuation and remaining functions
# This continues build_summary_row from Part 2

        "90 percentile lap": f"{driver.median_clean:.2f}" if driver.median_clean else "0.00",
        "Consistency": f"{driver.consistency_pct:.1f}%",
        "Laps within 3% of hot lap": f"{driver.laps_within_3pct:.0f}%",
        "Fuel Stops": len(driver.fuel_stops_idx),
        "Avg Fuel Interval": format_time_mm_ss(avg_fuel_interval),
        "Avg Fuel Stop Time": format_time_ss_decimal(avg_fuel_stop_time),
        "Events/Incidents": driver.num_incidents,
        "Fuel Time Lost": format_time_mm_ss_decimal(fuel_time_lost),
        "Events Time Lost": format_time_mm_ss_decimal(driver.time_lost_incidents)
    }

def build_event_log(driver, lap_positions):
    """Build verbose event log for a driver."""
    log_lines = []
    current_time = 0.0
    last_fuel_time = 0.0
    
    for i, laptime in enumerate(driver.lap_times):
        prev_time = current_time
        current_time += laptime
        delta = laptime - driver.median_clean
        
        event = None
        if i in driver.major_incidents:
            if i in driver.fuel_stops_idx:
                event = "MAJOR EVENT + FUEL"
            else:
                event = "MAJOR EVENT"
        elif i in driver.fuel_stops_idx:
            event = "FUEL STOP"
        elif i in driver.minor_incidents:
            event = "MINOR EVENT"
        
        if not event:
            continue
        
        log_line = (f"Lap {i+1} completed at={format_time_mm_ss(current_time)}, "
                   f"{event}, lap time={format_log_time(laptime)}, "
                   f"time lost={format_log_time(delta)}")
        
        if "FUEL" in event:
            time_of_fueling = prev_time + (driver.median_clean / 2.0)
            if len([x for x in driver.fuel_stops_idx if x <= i]) == 1:
                last_fuel_stint_duration = time_of_fueling
            else:
                last_fuel_stint_duration = time_of_fueling - last_fuel_time
            log_line += f", last fuel={format_time_mm_ss(last_fuel_stint_duration)}"
            last_fuel_time = time_of_fueling
        else:
            last_fuel_stint_duration = current_time - last_fuel_time
            log_line += f", last fuel={format_time_mm_ss(last_fuel_stint_duration)}"
        
        positions = lap_positions.get(driver.driver_key, [])
        if i < len(positions):
            before_pos = positions[i-1] if i > 0 else positions[0]
            after_pos = positions[i]
            
            if before_pos is not None and after_pos is not None:
                if before_pos == after_pos:
                    log_line += f", track position={before_pos}"
                else:
                    log_line += f", track position={before_pos}→{after_pos}"
                    
                    if before_pos > after_pos:
                        passed_drivers = []
                        for pos in range(after_pos, before_pos):
                            dn = get_driver_at_position(i+1, pos, lap_positions)
                            if dn:
                                surname = dn.split()[-1]
                                passed_drivers.append(surname)
                        if passed_drivers:
                            log_line += f" passed: {', '.join(passed_drivers)}"
                    elif before_pos < after_pos:
                        passed_by = []
                        for pos in range(before_pos + 1, after_pos + 1):
                            dn = get_driver_at_position(i+1, pos, lap_positions)
                            if dn:
                                surname = dn.split()[-1]
                                passed_by.append(surname)
                        if passed_by:
                            log_line += f" passed by: {', '.join(passed_by)}"
        
        log_lines.append(log_line)
    
    return log_lines

def analyze_driver_events(driver_data_dict, lap_positions, meta, global_hot_lap):
    """Per-driver loop: classify events, compute metrics, build summary rows and verbose logs."""
    finishing_order = [d for d in driver_data_dict.keys() if d in lap_positions]
    if "header_names" in meta:
        filtered = [d for d in meta["header_names"] 
                   if d in driver_data_dict and d in lap_positions]
        if filtered:
            finishing_order = filtered
    
    results = []
    verbose_log = {}
    
    race_duration_sec = meta.get("race_duration_sec")
    Config.NOFUEL_MODE = meta.get("nofuel_mode", Config.NOFUEL_MODE)
    
    for pos_idx, driver_key in enumerate(finishing_order, start=1):
        driver = driver_data_dict[driver_key]
        
        driver.compute_median_clean_lap()
        classify_driver_events(driver, lap_positions, race_duration_sec)
        driver.compute_best_lap(exclude_first=True)
        driver.compute_laps_within_3pct(global_hot_lap)
        driver.compute_consistency_metrics()
        driver.compute_fade_resistance()
        
        row = build_summary_row(driver, pos_idx)
        results.append(row)
        
        event_log = build_event_log(driver, lap_positions)
        if event_log:
            verbose_log[driver.get_display_name()] = event_log
    
    summary_df = pd.DataFrame(results)
    if not summary_df.empty:
        summary_df = award_trophies(summary_df, "Best Lap", higher_is_better=False)
        summary_df = award_trophies(summary_df, "90 percentile lap", higher_is_better=False)
        summary_df = award_trophies(summary_df, "Consistency", higher_is_better=True)
        summary_df = award_trophies(summary_df, "Laps within 3% of hot lap", higher_is_better=True)
    
    return summary_df, verbose_log

def detect_overtakes(driver, lap_positions, lap_times, drivers_fuel_stops, drivers_major_incidents, median_clean, debug_log=None):
    """Detect on-track overtakes (true passes not caused by pit stops or incidents)."""
    if debug_log is None:
        debug_log = []

    overtakes = 0
    overtaken_drivers = []
    overtake_deltas = []

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
                i in drivers_fuel_stops.get(driver, [])
                or i in drivers_major_incidents.get(driver, [])
                or i in drivers_fuel_stops.get(opponent, [])
                or i in drivers_major_incidents.get(opponent, [])
            ):
                continue

            if pos_curr_d < pos_prev_d and pos_curr_o > pos_prev_o and pos_curr_d < pos_curr_o and pos_prev_d > pos_prev_o:
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

def detect_losses(driver, lap_positions, lap_times, drivers_fuel_stops, drivers_major_incidents, median_clean, debug_log=None):
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
                i in drivers_fuel_stops.get(driver, [])
                or i in drivers_major_incidents.get(driver, [])
                or i in drivers_fuel_stops.get(opponent, [])
                or i in drivers_major_incidents.get(opponent, [])
            ):
                continue

            if pos_curr_d > pos_prev_d and pos_curr_o < pos_prev_o and pos_curr_d > pos_curr_o and pos_prev_d < pos_prev_o:
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

def analyze_overtakes_and_losses(driver, all_drivers, lap_positions, racecraft_debug_list):
    """Analyze overtakes and losses for a driver using DriverData objects."""
    debug_log = []
    
    drivers_fuel_stops = {
        key: d.fuel_stops_idx for key, d in all_drivers.items()
    }
    drivers_major_incidents = {
        key: d.major_incidents for key, d in all_drivers.items()
    }
    
    overtakes, overtaken_drivers, avg_loss_ot, log_ot = detect_overtakes(
        driver.driver_key, 
        lap_positions, 
        driver.lap_times, 
        drivers_fuel_stops, 
        drivers_major_incidents, 
        driver.median_clean, 
        debug_log
    )
    
    losses, lost_to_drivers, avg_loss_ls, log_ls = detect_losses(
        driver.driver_key,
        lap_positions, 
        driver.lap_times, 
        drivers_fuel_stops, 
        drivers_major_incidents, 
        driver.median_clean, 
        debug_log
    )
    
    debug_log.extend(log_ot)
    debug_log.extend(log_ls)
    
    racecraft_debug_list.append({
        "driver": driver.driver_key,
        "overtakes": overtakes,
        "losses": losses,
        "log_ot": log_ot,
        "log_ls": log_ls,
    })
    
    return {
        "overtakes": overtakes,
        "losses": losses,
        "avg_time_lost_overtakes": avg_loss_ot,
        "avg_time_lost_losses": avg_loss_ls,
        "overtaken_drivers": overtaken_drivers,
        "lost_to_drivers": lost_to_drivers,
        "debug_log": debug_log,
    }

def check_overtake_losses_balance(global_racecraft_debug, tolerance=0):
    """Compare total on-track overtakes vs. losses across all drivers."""
    total_overtakes = sum(entry.get("overtakes", 0) for entry in global_racecraft_debug)
    total_losses    = sum(entry.get("losses", 0) for entry in global_racecraft_debug)

    diff = abs(total_overtakes - total_losses)
    
    # Print detailed breakdown if verbose logging is enabled
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("=== Racecraft Balance Check ===")
        for entry in global_racecraft_debug:
            driver = entry.get("driver", "Unknown")
            overtakes = entry.get("overtakes", 0)
            losses = entry.get("losses", 0)
            logger.debug(f"  {driver:<25} Overtakes: {overtakes:>3}  Losses: {losses:>3}")
        logger.debug(f"  {'TOTAL':<25} Overtakes: {total_overtakes:>3}  Losses: {total_losses:>3}")
        logger.debug(f"  Difference: {diff} (tolerance: {tolerance})")
        logger.debug("=" * 50)
    
    if diff > tolerance:
        logger.warning(
            f"Warning: overtake/loss mismatch detected: {total_overtakes} overtakes vs {total_losses} losses "
            f"(diff={diff}, tolerance={tolerance})"
        )

def analyze_lapping_events(driver, all_drivers, lap_positions):
    """Analyze lapping events using DriverData objects."""
    fuel_stops = set(driver.fuel_stops_idx)
    major_incidents = set(driver.major_incidents)
    driver_laps = driver.no_of_laps
    median_clean = driver.median_clean
    
    lappings = 0
    lapped_by = 0
    lapping_deltas = []
    lapped_by_deltas = []
    
    for other_key, other_driver in all_drivers.items():
        if other_key == driver.driver_key:
            continue
        
        other_laps = other_driver.no_of_laps
        other_positions = lap_positions.get(other_key, [])
        
        if not other_positions:
            continue
        
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
            lapping_deltas.append(median_clean * 0.02 * lap_diff)
        elif lap_diff <= -1:
            lapped_by += 1
            lapped_by_deltas.append(median_clean * 0.05 * abs(lap_diff))
    
    avg_time_lost_lappings = float(np.mean(lapping_deltas)) if lapping_deltas else 0.0
    avg_time_lost_lapped = float(np.mean(lapped_by_deltas)) if lapped_by_deltas else 0.0
    
    # Store lapped_by count in driver object
    driver.lappings_count = lappings
    driver.lapped_by_count = lapped_by
    
    return {
        "lappings": lappings,
        "lapped_by": lapped_by,
        "avg_time_lost_lappings": avg_time_lost_lappings,
        "avg_time_lost_lapped": avg_time_lost_lapped,
    }

def compute_racecraft_caps(driver_data_dict, over_pct=95, loss_pct=95):
    """Compute percentile caps for overtakes/losses normalization."""
    over_list = []
    loss_list = []
    
    for driver in driver_data_dict.values():
        rc = driver.racecraft_basic or {}
        over_list.append(rc.get("overtakes", 0) or 0)
        loss_list.append(rc.get("losses", 0) or 0)
    
    cap_over = float(np.percentile(over_list, over_pct)) if any(over_list) else 1.0
    cap_loss = float(np.percentile(loss_list, loss_pct)) if any(loss_list) else 1.0
    
    if cap_over <= 0:
        cap_over = 1.0
    if cap_loss <= 0:
        cap_loss = 1.0
    
    return cap_over, cap_loss

def compute_overtake_loss_scores(overtakes, losses, cap_over, cap_loss):
    """Normalize overtake and loss scores."""
    over_norm = min(overtakes / cap_over, 1.0) if cap_over > 0 else 0.0
    loss_pen  = min(losses   / cap_loss, 1.0)  if cap_loss > 0 else 1.0
    loss_score = 1.0 - loss_pen
    return over_norm, loss_score

def score_time_lost_incidents_linear(time_lost, total_raced_time, debug=False):
    """Convert time lost to incidents into a normalized score between 0.0–1.0."""
    MAX_RATIO = 0.07 # this % of race time lost to incidents will give 0% score

    if total_raced_time <= 0:
        return 1.0
    ratio = time_lost / total_raced_time 
    score = 1.0 - (ratio / MAX_RATIO)  
    score = max(0.0, min(1.0, score))

    return score

def score_incident_density_linear(num_incidents, total_raced_time, debug=False):
    """Convert incident density (incidents per race minute) into a normalized 0–1 score."""
    INC_PER_MIN_FOR_0_SCORE = 1.0

    if total_raced_time <= 0:
        return 1.0
    minutes = total_raced_time / 60.0
    inc_per_min = num_incidents / minutes
    score = 1.0 - (inc_per_min / INC_PER_MIN_FOR_0_SCORE)
    score = max(0.0, min(1.0, score))

    return score

def score_fade_resistance_linear(fade_ratio, debug=False):
    """Convert a fade ratio (last_avg / first_avg) into a normalized 0–1 score."""
    FADES_TO_ZERO_AT = 1.25 # % of initial pace/lap times that will yield a 0% score

    if fade_ratio <= 1.0:
        score = 1.0
    else:
        score = 1.0 - ((fade_ratio - 1.0) / (FADES_TO_ZERO_AT - 1.0))
    score = max(0.0, min(1.0, score))

    return score

def compute_speed_index(driver, class_ref):
    """Compute Speed Index using DriverData object."""
    SWC = Config.DPI_WEIGHTS["Speed"]["components"]
    
    best_ref = class_ref.get("best_lap_ref", driver.best_lap)
    p90_ref = class_ref.get("p90_ref", driver.median_clean)
    
    best_ratio = best_ref / driver.best_lap if driver.best_lap > 0 else 0
    p90_ratio = p90_ref / driver.median_clean if driver.median_clean > 0 else 0
    laps_within_3pct = driver.laps_within_3pct / 100.0
    
    speed_index = (
        SWC["Best Lap"]   * best_ratio +
        SWC["90% Lap"]    * p90_ratio +
        SWC["% <3% Laps"] * laps_within_3pct
    )
    speed_index = max(0.0, min(speed_index, 1.0))
    
    breakdown = {
        "best_lap": driver.best_lap,
        "best_ratio": best_ratio,
        "p90_lap": driver.median_clean,
        "p90_ratio": p90_ratio,
        "laps_within_3pct": driver.laps_within_3pct,
    }
    
    return speed_index, breakdown

def compute_consistency_index(driver):
    """Compute Consistency Index using DriverData object."""
    CWC = Config.DPI_WEIGHTS["Consistency"]["components"]
    
    consistency_pct = driver.consistency_pct / 100.0
    fade_resistance = driver.fade_resistance
    num_incidents = driver.num_incidents
    time_lost_incidents = driver.time_lost_incidents
    total_raced_time = driver.total_raced_time
    
    fade_score = score_fade_resistance_linear(fade_resistance)
    time_lost_score = score_time_lost_incidents_linear(time_lost_incidents, total_raced_time)
    incident_density_score = score_incident_density_linear(num_incidents, total_raced_time)
    
    consistency_index = (
        CWC["Consistency %"]      * consistency_pct +
        CWC["Fade Resistance"]    * fade_score +
        CWC["# Incidents"]        * incident_density_score +
        CWC["Time Lost Incidents"]* time_lost_score
    )
    consistency_index = max(0.0, min(consistency_index, 1.0))
    
    breakdown = {
        "consistency_pct": driver.consistency_pct,
        "fade_resistance": driver.fade_resistance,
        "fade_score": fade_score,
        "num_incidents": driver.num_incidents,
        "incident_density_score": incident_density_score,
        "time_lost_incidents": driver.time_lost_incidents,
        "time_lost_score": time_lost_score,
        "total_raced_time": driver.total_raced_time,
    }
    
    return consistency_index, breakdown

def compute_racecraft_index(driver):
    """Compute Racecraft Index using DriverData object."""
    RWC = Config.DPI_WEIGHTS["Racecraft"]["components"]
    
    rc_basic = driver.racecraft_basic or {}
    rc_lap = driver.racecraft_lap or {}
    
    overtakes = rc_basic.get("overtakes", 0)
    losses = rc_basic.get("losses", 0)
    over_score = driver.overtake_score
    loss_score = driver.loss_score
    
    avg_loss_overtakes = rc_basic.get("avg_time_lost_overtakes", 0.0)
    avg_loss_losses = rc_basic.get("avg_time_lost_losses", 0.0)
    avg_loss_lappings = rc_lap.get("avg_time_lost_lappings", 0.0)
    avg_loss_lapped = rc_lap.get("avg_time_lost_lapped", 0.0)
    
    eff_over = max(0.0, min(1.0, 1 - safe_div(avg_loss_overtakes, driver.median_clean)))
    eff_lost = max(0.0, min(1.0, 1 - safe_div(avg_loss_losses, driver.median_clean)))
    eff_lapping = max(0.0, min(1.0, 1 - safe_div(avg_loss_lappings, driver.median_clean)))
    eff_lapped = max(0.0, min(1.0, 1 - safe_div(avg_loss_lapped, driver.median_clean)))

    
    racecraft_index = (
        RWC["Overtakes"]    * over_score +
        RWC["Losses"]       * loss_score +
        RWC["Overtake Eff"] * eff_over +
        RWC["Passed Eff"]   * eff_lost +
        RWC["Lapping Eff"]  * eff_lapping +
        RWC["Lapped Eff"]  * eff_lapped
    )
    racecraft_index = max(0.0, min(racecraft_index, 1.0))
    
    breakdown = {
        "overtakes": overtakes,
        "losses": losses,
        "overtake_score": over_score,
        "loss_score": loss_score,
        "eff_over": eff_over,
        "eff_lost": eff_lost,
        "eff_lapping": eff_lapping,
        "eff_lapped": eff_lapped,
        "avg_loss_overtakes": avg_loss_overtakes,
        "avg_loss_losses": avg_loss_losses,
        "avg_loss_lappings": avg_loss_lappings,
        "avg_loss_lapped": avg_loss_lapped,
        "median_clean": driver.median_clean,
    }
    
    return racecraft_index, breakdown

def compute_driver_performance_indices(driver_data_dict, global_hot_lap, lap_positions):
    """Compute DPI for all drivers using DriverData objects."""
    class_ref = {
        "best_lap_ref": global_hot_lap or 1.0,
        "p90_ref": min(
            (d.median_clean for d in driver_data_dict.values() if d.median_clean),
            default=global_hot_lap or 1.0
        ),
    }
    
    racecraft_debug_list = []
    dpi_results = []
    
    for driver_key, driver in driver_data_dict.items():
        if lap_positions is None:
            continue
        
        driver.racecraft_basic = analyze_overtakes_and_losses(
            driver,
            driver_data_dict,
            lap_positions,
            racecraft_debug_list
        )
        
        driver.racecraft_lap = analyze_lapping_events(
            driver,
            driver_data_dict,
            lap_positions
        )
    
    check_overtake_losses_balance(racecraft_debug_list)
    
    cap_over, cap_loss = compute_racecraft_caps(driver_data_dict)
    
    for driver_key, driver in driver_data_dict.items():
        rc_basic = driver.racecraft_basic or {}
        overtakes = rc_basic.get("overtakes", 0) or 0
        losses = rc_basic.get("losses", 0) or 0
        
        over_score, loss_score = compute_overtake_loss_scores(
            overtakes, losses, cap_over, cap_loss
        )
        
        driver.overtake_score = over_score
        driver.loss_score = loss_score
        driver.cap_over = cap_over
        driver.cap_loss = cap_loss
        
        speed_score, speed_breakdown = compute_speed_index(driver, class_ref)
        consistency_score, consistency_breakdown = compute_consistency_index(driver)
        racecraft_score, racecraft_breakdown = compute_racecraft_index(driver)
        
        SW = Config.DPI_WEIGHTS["Speed"]["weight"]
        CW = Config.DPI_WEIGHTS["Consistency"]["weight"]
        RW = Config.DPI_WEIGHTS["Racecraft"]["weight"]
        total_w = SW + CW + RW
        SW, CW, RW = (SW / total_w, CW / total_w, RW / total_w)
        
        total_dpi = 100.0 * (SW * speed_score + CW * consistency_score + RW * racecraft_score)
        
        driver.speed_index = speed_score
        driver.consistency_index = consistency_score
        driver.racecraft_index = racecraft_score
        driver.dpi = total_dpi
        
        components = {
            **speed_breakdown,
            **consistency_breakdown,
            **racecraft_breakdown,
            "overtakes": overtakes,
            "losses": losses,
            "overtake_score": over_score,
            "loss_score": loss_score,
            "cap_over": cap_over,
            "cap_loss": cap_loss,
            "lappings_count": driver.lappings_count,
            "lapped_by_count": driver.lapped_by_count,
        }
        
        dpi_results.append({
            "Driver": driver.full_name,
            "Car": driver.car_number,
            "Speed": round(speed_score * 100, 1),
            "Consistency": round(consistency_score * 100, 1),
            "Racecraft": round(racecraft_score * 100, 1),
            "DPI": total_dpi,
            "Components": components,
        })
    
    return dpi_results

def print_pdf_report(summary_df, driver_data_dict, lap_positions, meta, verbose_log, csv_file, pdf_file, dpi_results):
    """Generate a full multi-page race report PDF."""
    if pdf_file and not str(pdf_file).startswith("--"):
        pdf_path = pathlib.Path(pdf_file)
    else:
        pdf_path = pathlib.Path(os.path.splitext(os.path.basename(csv_file))[0] + ".pdf")

    margin = 20

    def add_header_and_footer(canvas, doc):
        """Add footer with attribution on each page."""
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        y = 0.75 * cm
        x = doc.pagesize[0] - doc.rightMargin
        canvas.drawRightString(x, y, "(C) Balthazar RC")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(A4),
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=margin
    )

    styles = getSampleStyleSheet()
    elements = []

    print_pdf_cover_header(elements, meta, styles)
    print_pdf_summary_table(elements, summary_df, styles)
    print_pdf_lapchart(elements, driver_data_dict, lap_positions, meta, styles)
    print_pdf_driver_perf_summary(elements, dpi_results)
    print_pdf_driver_perf_components(elements, dpi_results)
    print_pdf_event_log_pages(elements, verbose_log, styles)

    doc.build(elements, onFirstPage=add_header_and_footer, onLaterPages=add_header_and_footer)
    logger.info("PDF written to %s", pdf_path)

def print_pdf_cover_header(elements, meta, styles):
    """Add race title, disclaimer, and notes page."""
    elements.append(Paragraph("The Balthazar RC Race Report", styles["Heading3"]))
    elements.append(Spacer(1, 12))

    race_event = meta.get("race_event", "")
    race_name = meta.get("race_name", "")
    race_details = meta.get("race_details", "")

    title = ""
    if race_event:
        title += race_event + " "
    if race_name:
        title += race_name + " "
    elements.append(Paragraph(title, styles["Heading1"]))
    elements.append(Spacer(1, 4))

    if race_details:
        left_style = ParagraphStyle("Left", parent=styles["Normal"], alignment=TA_LEFT)
        elements.append(Paragraph(race_details, left_style))

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
        "assumptions based on typical rc car races to arrive at the information presented in the summary table and event log. "
        "For fuel classes, identifying fuel stops and other events by purely analyzing lap times is unreliable because the analysis relies on a "
        "baseline of uneventful consistent laps. This means the more incidents a driver has during a race, the less reliable the fuel stop information will be. "
        "The track position chart is based on official myrcm data when available, or by "
        "calculating positions from lap times when track positions are not made avaialble."
    )

    if meta.get("nofuel_mode"):
        note_text += (
            "<br/><br/><b>Note:</b> Fuel stop detection is <b>disabled</b> for this race."
        )

    elements.append(Spacer(1, 16))
    elements.append(Paragraph(note_text, custom_note_style))

def print_pdf_summary_table(elements, summary_df, styles):
    """Add summary table page to the PDF."""
    left_style = ParagraphStyle("Left", parent=styles["Normal"], alignment=TA_LEFT)
    center_bold = ParagraphStyle("CenterBold", parent=styles["Normal"], alignment=TA_CENTER, fontName="Helvetica-Bold")

    elements.append(Paragraph("Race Summary", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    if summary_df.empty:
        elements.append(Paragraph("No summary could be computed.", left_style))
        elements.append(PageBreak())
        return

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

    margin = 20
    available_width = landscape(A4)[0] - 2 * margin - 60
    col_ratios = [0.05, 0.24, 0.05, 0.09, 0.07, 0.07, 0.07, 0.07, 0.05, 0.07, 0.05, 0.07, 0.05, 0.05]
    col_widths = [r * available_width for r in col_ratios]

    table_data = [header_row] + summary_df[[
        "Pos","Driver (No)","Laps","Race Time","Best Lap","90 percentile lap","Consistency",
        "Laps within 3% of hot lap","Fuel Stops","Avg Fuel Interval","Avg Fuel Stop Time",
        "Events/Incidents","Fuel Time Lost","Events Time Lost"
    ]].values.tolist()

    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.25, colors.black),
        ('FONTSIZE', (0,1), (-1,-1), 8),
    ]))

    elements.append(tbl)
    elements.append(Spacer(1, 8))
    
    note_style = ParagraphStyle(
        "NoteStyle",
        parent=styles["Normal"],
        fontSize=7,
        textColor=colors.grey,
        leftIndent=15  
    )
    elements.append(Paragraph("Note: 'Fuel Time Lost' and 'Avg Fuel Stop Time' include pure fuel stops only. "
        "Major events (e.g., flame outs, tire changes) that typically also include fueling are counted separately under 'Events Time Lost'.",
        note_style
    ))
    elements.append(Spacer(1, 2))
    elements.append(Paragraph(
        "Note: 'Consistency %' is calculated as 100 × (1 − σ/μ), where σ = standard deviation of lap times and μ = average lap time. "
        "Higher values indicate more consistent driving.",
        note_style))
    elements.append(Spacer(1, 2))
    elements.append(Paragraph(
        "Note: 'Laps within 3% of hot lap' is the percentage of a drivers laps that were within 3% of the best lap of the race across all drivers.",
        note_style
    ))
    
    elements.append(Spacer(1, 2))
    elements.append(PageBreak())

def print_pdf_lapchart(elements, driver_data_dict, lap_positions, meta, styles):
    """Add a page showing the drivers' track position chart."""
    header_names = meta.get("header_names", [])
    left_style = ParagraphStyle("Left", parent=styles["Normal"], alignment=TA_LEFT)

    # Determine driver order using DriverData objects
    finishing_order = [d for d in header_names if d in driver_data_dict]
    starting_order = sorted(
        finishing_order, 
        key=lambda d: safe_int(driver_data_dict[d].car_number, default=9999)
    )

    elements.append(Paragraph("Race Lap Chart", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    # Color palette
    base_colors = [black, red, green, blue, orange, violet, pink, gray,
                   Color(0.5, 0.2, 0.8), Color(0.8, 0.5, 0.2)]
    while len(base_colors) < len(starting_order):
        idx = len(base_colors)
        base_colors.append(Color(((idx * 37) % 255) / 255.0,
                                 ((idx * 61) % 255) / 255.0,
                                 ((idx * 97) % 255) / 255.0))

    # Chart setup
    margin = 20
    chart_width = landscape(A4)[0] - 2 * margin
    chart_height = 440
    drawing = Drawing(chart_width, chart_height)

    hc = HorizontalLineChart()
    hc.x = 100
    hc.y = 40
    hc.width = chart_width - 260
    hc.height = chart_height - 100

    # Data preparation
    drivers_for_chart = [d for d in starting_order if lap_positions.get(d)]
    max_laps_to_plot = max((len(lap_positions[d]) for d in drivers_for_chart), default=0)

    if max_laps_to_plot == 0:
        elements.append(Paragraph("No track position data available to draw the chart.", left_style))
        elements.append(PageBreak())
        return

    hc.data = []
    n_drivers = len(starting_order)

    for driver_key in drivers_for_chart:
        series = lap_positions[driver_key][:]
        def _clamp_val(v):
            try:
                iv = int(v)
            except Exception:
                iv = starting_order.index(driver_key) + 1
            return 1 if iv < 1 else (n_drivers if iv > n_drivers else iv)
        series = list(map(_clamp_val, series))
        if len(series) == 1:
            series = series * 2
        hc.data.append(series)

    # Axis scaling
    if max_laps_to_plot <= 70:
        step = 1
    elif max_laps_to_plot <= 150:
        step = 5
    else:
        step = 10

    hc.categoryAxis.categoryNames = [str(i + 1) if (i + 1) % step == 0 else '' for i in range(max_laps_to_plot)]
    hc.categoryAxis.labels.boxAnchor = 'n'
    hc.categoryAxis.labels.angle = 45
    hc.categoryAxis.labels.fontName = 'Helvetica-Bold'
    hc.categoryAxis.labels.fontSize = 6
    hc.categoryAxis.labels.dy = -8
    hc.categoryAxis.visibleGrid = False

    hc.valueAxis.valueMin = 1
    hc.valueAxis.valueMax = n_drivers
    hc.valueAxis.valueStep = 1
    hc.valueAxis.reverseDirection = True
    hc.valueAxis.visibleGrid = True
    hc.valueAxis.labels.fontName = 'Helvetica'
    hc.valueAxis.labels.fontSize = 7

    # Line styling
    for i in range(len(hc.data)):
        try:
            driver_key = drivers_for_chart[i]
            color_idx = starting_order.index(driver_key)
            hc.lines[i].strokeColor = base_colors[color_idx % len(base_colors)]
            hc.lines[i].strokeWidth = 1.6
        except Exception:
            pass

    drawing.add(hc)

    # Custom grid lines
    x0 = hc.x
    lap_to_x = lambda lap: x0 + (lap - 1) * (hc.width / max_laps_to_plot)
    for lap in range(step, max_laps_to_plot + 1, step):
        x = lap_to_x(lap)
        drawing.add(Line(x, hc.y, x, hc.y + hc.height, strokeColor=colors.lightgrey, strokeWidth=0.25))

    def y_for_rank(rank, value_min=1, value_max=None):
        if value_max is None:
            value_max = n_drivers
        if value_max == value_min:
            return hc.y + hc.height / 2.0
        frac = (rank - value_min) / float(value_max - value_min)
        return hc.y + (1.0 - frac) * hc.height

    # Labels
    top_label_y = y_for_rank(1) + 20
    bottom_label_y = hc.y - 30
    left_label_x = 10
    right_label_x = hc.x + hc.width + 20

    drawing.add(String(left_label_x, top_label_y, "Starting pos", fontName="Helvetica-Bold", fontSize=10))
    drawing.add(String(right_label_x, top_label_y, "Finishing pos", fontName="Helvetica-Bold", fontSize=10))
    drawing.add(String(hc.x + hc.width / 2, bottom_label_y, "Lap", fontName="Helvetica-Bold", fontSize=10))

    # Starting position labels (left side)
    for idx, driver_key in enumerate(starting_order):
        start_rank = idx + 1
        y = y_for_rank(start_rank)
        driver = driver_data_dict[driver_key]
        full_name = driver.full_name
        car_num = driver.car_number
        color = base_colors[idx % len(base_colors)]
        drawing.add(String(left_label_x, y - 4, f"{car_num} {full_name}", fontName="Helvetica-Bold", fontSize=8, fillColor=color))

    # Finishing position labels (right side)
    for fin_idx, driver_key in enumerate(finishing_order):
        fin_rank = fin_idx + 1
        y = y_for_rank(fin_rank)
        driver = driver_data_dict[driver_key]
        full_name = driver.full_name
        try:
            color_idx = starting_order.index(driver_key)
            color = base_colors[color_idx % len(base_colors)]
        except ValueError:
            color = black
        drawing.add(String(right_label_x, y - 4, f"{fin_rank} {full_name}", fontName="Helvetica-Bold", fontSize=8, fillColor=color))

    elements.append(drawing)
    elements.append(Spacer(1, 2))
    elements.append(PageBreak())

def print_pdf_driver_perf_summary(elements, dpi_results):
    """Add the first DPI summary page (overall Speed / Consistency / Racecraft)."""
    styles = getSampleStyleSheet()
    elements.append(Paragraph("Drivers Performance Index (DPI) - Driver Rankings", styles["Heading2"]))
    elements.append(Spacer(1, 12))

    # Filter out "Unnamed" placeholder drivers
    dpi_results = [r for r in dpi_results if not re.match(r'Unnamed: \d+', r['Driver'])]

    # Sort drivers by DPI descending (highest first)
    sorted_results = sorted(dpi_results, key=lambda x: x['DPI'], reverse=True)

    # Build a map of driver names to their finishing position
    finish_positions = {r["Driver"]: idx + 1 for idx, r in enumerate(dpi_results)}
    
    # Build a map of driver names to their starting position (sorted by car number)
    drivers_by_car = sorted(dpi_results, key=lambda x: int(x['Car']) if x['Car'].isdigit() else 9999)
    start_positions = {r["Driver"]: idx + 1 for idx, r in enumerate(drivers_by_car)}

    summary_data = [
        ["Driver Rank", "Driver (Start pos, Finish pos)", "Overall DPI", "Speed", "Consistency", "Racecraft"],
        [
            "", "", "",
            f"{Config.DPI_WEIGHTS['Speed']['weight']*100:.0f}%",
            f"{Config.DPI_WEIGHTS['Consistency']['weight']*100:.0f}%",
            f"{Config.DPI_WEIGHTS['Racecraft']['weight']*100:.0f}%",
        ],
    ]

    # Add data rows with driver rank based on DPI
    for rank, r in enumerate(sorted_results, start=1):
        driver_name = r['Driver']
        start_pos = start_positions.get(driver_name, "?")
        finish_pos = finish_positions.get(driver_name, "?")
        driver_with_positions = f"{driver_name} ({start_pos},{finish_pos})"
        
        summary_data.append([
            rank,
            driver_with_positions,
            f"{r['DPI']:.1f}%",
            f"{r['Speed']:.1f}%",
            f"{r['Consistency']:.1f}%",
            f"{r['Racecraft']:.1f}%",
        ])

    # Create table with hAlign='LEFT' to force left alignment
    summary_table = Table(summary_data, repeatRows=2, hAlign='LEFT')
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),

        ('BACKGROUND', (0, 1), (-1, 1), colors.whitesmoke),
        ('TEXTCOLOR', (0, 1), (-1, 1), colors.darkgrey),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Oblique'),
        ('ALIGN', (0, 1), (-1, 1), 'CENTER'),
        ('FONTSIZE', (0, 1), (-1, 1), 7),

        ('ALIGN', (0, 2), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
        ('FONTSIZE', (0, 2), (-1, -1), 8),
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 12))
    
    # Add explanatory note
    note_style = ParagraphStyle(
        "NoteStyle",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.grey,
        leftIndent=15,
        spaceAfter=6
    )
    
    note_text = [
        "<b>Understanding the Driver Performance Index (DPI):</b>",
        "",
        "<b>Speed</b> measures how fast you are compared to other drivers. It looks at your best lap time, how many of your laps were within 3% of the fastest lap in the race, and your best 90% of laps (excluding your worst 10% of laps). A higher Speed score means you were consistently quick.",
        "",
        "<b>Consistency</b> measures how steady and clean your driving was. It considers how similar your lap times were to each other, how many incidents (mistakes, crashes, or delays) you had, how much time you lost due to those incidents, and whether you maintained your pace throughout the race (fade resistance). A higher Consistency score means you drove smoothly with fewer mistakes.",
        "",
        "<b>Racecraft</b> measures your racing skills in traffic. It looks at how many overtakes you made, how often you were passed, and how efficiently you handled these situations without losing much time. It also considers how well you handled lapping slower cars or being lapped by faster cars. A higher Racecraft score means you're good at racing wheel-to-wheel and managing traffic.",
        "",
        "<b>Overall DPI</b> is calculated by combining these three scores with the following weights: Speed (40%), Consistency (30%), and Racecraft (30%). This gives you a single number that represents your overall performance in the race, where 100% would be perfect in all areas.",
        "",
        "<b>Note</b> Driver performance Index is based on actual raced laps for the driver, hence an early retirement will not impact the index, but very much the race result, which is what matters in the end :-)"
    ]
    
    for line in note_text:
        if line:  # Only add paragraph if line is not empty
            elements.append(Paragraph(line, note_style))
        else:  # Add smaller spacer for empty lines
            elements.append(Spacer(1, 4))
    
    elements.append(Spacer(1, 20))
    elements.append(PageBreak())

def print_pdf_driver_perf_components(elements, dpi_results):
    """Add the Driver Performance Index - Component Breakdown page."""
    styles = getSampleStyleSheet()
    elements.append(Paragraph("Driver Performance Index - Component Breakdown", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    # Filter out "Unnamed" placeholder drivers
    dpi_results = [r for r in dpi_results if not re.match(r'Unnamed: \d+', r['Driver'])]

    # Sort drivers by DPI descending (same order as summary page)
    sorted_results = sorted(dpi_results, key=lambda x: x['DPI'], reverse=True)

    # Header rows with wrapped column names
    comp_data = [
        ["", "Speed", "", "", "Consistency", "", "", "", "Racecraft", "", "", "", "", ""],
        [
            "Driver",
            "Best Lap",
            "Within 3%\nLaps",
            "90% Lap",
            "Consistency",
            "Incident\nAvoidance",
            "Incident\nTime Lost",
            "Fade\nResistance",
            "On-track\nOvertakes",
            "On-track\nLosses",
            "Overtake\nEfficiency",
            "Losses\nEfficiency",
            "Lapping\nEfficiency",
            "Lapped\nEfficiency",
        ],
        [
            "Weight",
            *[f"{v:.2f}" for v in Config.DPI_WEIGHTS["Speed"]["components"].values()],
            *[f"{v:.2f}" for v in Config.DPI_WEIGHTS["Consistency"]["components"].values()],
            *[f"{v:.2f}" for v in Config.DPI_WEIGHTS["Racecraft"]["components"].values()],
        ],
    ]

    # Extract caps (same for all drivers)
    cap_over = None
    cap_loss = None
    if sorted_results:
        c0 = sorted_results[0].get("Components", {})
        cap_over = c0.get("cap_over", None)
        cap_loss = c0.get("cap_loss", None)

    # Data rows per driver (now sorted by DPI)
    for r in sorted_results:
        c = r.get("Components", {})
        overtake_score = c.get("overtake_score", 0) * 100
        loss_score = c.get("loss_score", 0) * 100
        overtakes = c.get("overtakes", 0)
        losses = c.get("losses", 0)
        lappings_count = c.get("lappings_count", 0) 
        lapped_by_count = c.get("lapped_by_count", 0) 
        lappings_eff     = c.get("eff_lapping", 0) * 100
        lapped_eff      = c.get("eff_lapped", 0) * 100

        comp_data.append([
            f"{r['Driver']} ({r['Car']})",
            f"{c.get('best_ratio', 0)*100:.1f}% ({c.get('best_lap', 0):.3f})",
            f"{c.get('laps_within_3pct', 0):.1f}%",
            f"{c.get('p90_ratio', 0)*100:.1f}% ({c.get('p90_lap', 0):.3f})",
            f"{c.get('consistency_pct', 0):.1f}%",
            f"{c.get('incident_density_score', 0)*100:.1f}%",
            f"{c.get('time_lost_score', 0)*100:.1f}%",
            f"{c.get('fade_score', 0)*100:.1f}%",
            f"{overtake_score:.1f}% ({overtakes})",
            f"{loss_score:.1f}% ({losses})",
            f"{c.get('eff_over', 0)*100:.1f}%",
            f"{c.get('eff_lost', 0)*100:.1f}%",
            f"{lappings_eff:.1f}% ({lappings_count})",  
            f"{lapped_eff:.1f}% ({lapped_by_count})",
        ])

    # Build table with LEFT alignment
    comp_table = Table(comp_data, repeatRows=3, hAlign='LEFT') 
    comp_table.setStyle(TableStyle([
        # Group headers
        ('SPAN', (1, 0), (3, 0)),   # Speed (3 cols)
        ('SPAN', (4, 0), (7, 0)),   # Consistency (now 4 cols)
        ('SPAN', (8, 0), (13, 0)),  # Racecraft (remaining 6 cols)

        ('BACKGROUND', (1, 0), (13, 0), colors.lightgrey),
        ('TEXTCOLOR', (1, 0), (13, 0), colors.black),
        ('FONTNAME', (1, 0), (13, 0), 'Helvetica-Bold'),
        ('ALIGN', (1, 0), (13, 0), 'CENTER'),
        ('VALIGN', (1, 0), (13, 0), 'MIDDLE'),
        ('FONTSIZE', (1, 0), (13, 0), 9),

        ('LINEAFTER', (3, 0), (3, -1), 0.5, colors.black),  # after Speed
        ('LINEAFTER', (7, 0), (7, -1), 0.5, colors.black),  # after Consistency

        # Column headers
        ('BACKGROUND', (0, 1), (-1, 1), colors.grey),
        ('TEXTCOLOR', (0, 1), (-1, 1), colors.whitesmoke),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('ALIGN', (0, 1), (-1, 1), 'CENTER'),
        ('VALIGN', (0, 1), (-1, 1), 'MIDDLE'),
        ('FONTSIZE', (0, 1), (-1, 1), 7),

        # Weight row
        ('BACKGROUND', (0, 2), (-1, 2), colors.whitesmoke),
        ('TEXTCOLOR', (0, 2), (-1, 2), colors.darkgrey),
        ('FONTNAME', (0, 2), (-1, 2), 'Helvetica-Oblique'),
        ('ALIGN', (0, 2), (-1, 2), 'CENTER'),
        ('FONTSIZE', (0, 2), (-1, 2), 6),

        # Data rows
        ('ALIGN', (0, 3), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
        ('FONTSIZE', (0, 3), (-1, -1), 7),
    ]))

    elements.append(comp_table)
    elements.append(Spacer(1, 12))

    # Notes section
    note_style = ParagraphStyle(
        "NoteStyle",
        parent=styles["Normal"],
        fontSize=7,
        textColor=colors.grey,
        leftIndent=15
    )

    note_lines = [
        "<b>Best Lap</b>: 100% equals the race’s fastest lap, while 98% means the lap was 2% slower than the hot lap",
        "<b>Within 3% Laps</b>: % of driver’s laps within 3% of the overall fastest lap.",
        "<b>90% Lap</b>: 100% means the driver's best 90% of laps were the fastest across all drivers, while 96% means that driver's best 90% of laps were 4% slower.",
        "<b>Consistency</b>: Measures how evenly paced the driver’s laps are. 100% means every lap time is identical; more variation lowers the score ((1 − σ/μ) × 100)",
        "<b>Incident Avoidance</b>: Having no incidents gives 100%; averaging one or more incidents per race minute gives 0%.",
        "<b>Incident Time Lost</b>: Losing no time to incidents gives 100%; losing 7% or more of raced time gives 0%.",
        "<b>Fade Resistance</b>: Matching early pace in the final quarter gives 100%; being 25% slower gives 0%",
        "<b>On-track Overtakes</b>: Making more on-track overtakes than 95% of drivers gives 100%; fewer gives a lower score.",
        "<b>On-track Losses</b>: Being passed on-track more times than 95% of drivers gives 0%; fewer than 5% gives 100%. ",
        "<b>Overtake Efficiency</b>: Losing no time doing overtakes compared to a clean lap gives 100%; losing a full clean-lap’s worth of time gives 0%.",
        "<b>Losses Efficiency</b>: Losing no time being overtaken compared to a clean lap gives 100%; losing a full clean-lap’s worth of time gives 0%.",
        "<b>Lapping Efficiency</b>: Losing no time lapping others compared to a clean lap gives 100%; losing a full clean-lap’s worth of time gives 0%.",
        "<b>Lapped Efficiency</b>: Losing no time being lapped compared to a clean lap gives 100%; losing a full clean-lap’s worth of time gives 0%.",
    ]

    """
    if cap_over is not None and cap_loss is not None:
        elements.append(Spacer(1, 12))
        note_lines.append(
            f"For this race: {cap_over:.1f} overtakes = 100%, {cap_loss:.1f} losses = 0% "
        )
    """

    for line in note_lines:
        elements.append(Paragraph(line, note_style))

    elements.append(PageBreak())

def print_pdf_event_log_pages(elements, verbose_log, styles):
    """Add the Drivers Events/Incidents log page to the PDF."""
    left_style = ParagraphStyle("Left", parent=styles["Normal"], alignment=TA_LEFT)

    if not verbose_log:
        return

    elements.append(Paragraph("Drivers Events/Incidents Log", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    for driver_display, logs in verbose_log.items():
        elements.append(Paragraph(driver_display, styles["Heading3"]))
        for line in logs:
            elements.append(Paragraph(line, left_style))
        elements.append(Spacer(1, 8))

    elements.append(Spacer(1, 2))
    elements.append(PageBreak())

def main(csv_file, pdf_file=None, verbose=False, nofuel=False):
    """Main entry point for race analysis and PDF generation."""
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug / Verbose logging enabled")
    
    if nofuel:
        Config.NOFUEL_MODE = True
    
    logger.info("Loading race data from %s", csv_file)
    df, meta = load_race_metadata(csv_file)
    
    logger.info("Processing driver lap data...")
    driver_data_dict, lap_positions, global_hot_lap = build_driver_lap_data(df, meta)
    
    logger.info("Analyzing race events and computing metrics...")
    summary_df, verbose_log = analyze_driver_events(
        driver_data_dict,
        lap_positions,
        meta,
        global_hot_lap
    )
    
    logger.info("Computing Driver Performance Index...")
    dpi_results = compute_driver_performance_indices(
        driver_data_dict,
        global_hot_lap,
        lap_positions
    )
    
    logger.info("Generating PDF report...")
    print_pdf_report(
        summary_df,
        driver_data_dict,  # Pass DriverData objects directly
        lap_positions,
        meta,
        verbose_log,
        csv_file,
        pdf_file,
        dpi_results
    )
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RC race stats PDF from myrcm CSV")
    parser.add_argument("csv_file", help="Input race CSV file")
    parser.add_argument("pdf_file", nargs="?", help="Output PDF file (optional)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--nofuel", action="store_true", help="Disable fuel stop classification")

    args = parser.parse_args()

    try:
        main(csv_file=args.csv_file, pdf_file=args.pdf_file, verbose=args.verbose, nofuel=args.nofuel)
    except KeyboardInterrupt:
        print("\nAborted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception:
        logger.exception("Fatal error:")
        sys.exit(1) 
  
"""    except Exception as e:
        logger.error("Fatal error: %s", e)
        if args.verbose:
            logger.exception("Stack trace:")
        sys.exit(1)"""