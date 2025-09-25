#!/usr/bin/env python3
"""
myrcm_runner.py
Wrapper to run scraper + stats in one command.

Usage:
    python myrcm_runner.py <url> <search terms...> [--verbose]

Example:
    python myrcm_runner.py https://myrcm.ch/myrcm/report/en/88811/365280# Final A
"""

import sys
import subprocess
import pathlib

BASE_DIR = pathlib.Path(__file__).parent.resolve()

def main():
    if len(sys.argv) < 3:
        print("Usage: python myrcm_runner.py <url> <search terms...> [--verbose]")
        sys.exit(1)

    url = sys.argv[1]
    args = sys.argv[2:]

    # Absolute paths to the other scripts
    scraper_path = BASE_DIR / "myrcm_scraper.py"
    stats_path = BASE_DIR / "myrcm_stats.py"

    cwd = pathlib.Path.cwd()  # current working directory (Flask sets this to run_dir)

    # Step 1: Run scraper
    scraper_cmd = ["python", str(scraper_path), url] + args
    result = subprocess.run(scraper_cmd, capture_output=True, text=True, cwd=cwd)

    stdout_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    stderr_lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]

    # --- Handle scraper failures ---
    if result.returncode == 2:
        # Distinct exit for "no results tables found"
        print("⚠️ No results available for this final.")
        if stderr_lines:
            print("stderr:\n" + "\n".join(stderr_lines))
        sys.exit(2)

    if result.returncode != 0:
        print("❌ Scraper failed with non-zero exit code.")
        if stderr_lines:
            print("stderr:\n" + "\n".join(stderr_lines))
        sys.exit(1)

    if not stdout_lines:
        print("❌ Scraper did not return a CSV filename.")
        if stderr_lines:
            print("stderr:\n" + "\n".join(stderr_lines))
        sys.exit(1)

    csv_file = stdout_lines[-1]

    # Step 2: Handle scraper error messages
    if csv_file.startswith("ERROR:"):
        print(f"❌ {csv_file}")
        if stderr_lines:
            print("stderr:\n" + "\n".join(stderr_lines))
        sys.exit(1)

    print(f"✅ Scraper produced CSV: {csv_file}")

    # Step 3: Run stats on CSV
    pdf_name = pathlib.Path(csv_file).with_suffix(".pdf").name
    pdf_path = cwd / pdf_name  # ensure inside cwd

    stats_cmd = ["python", str(stats_path), csv_file, str(pdf_path)]
    if "--verbose" in args:
        stats_cmd.append("--verbose")

    stats_result = subprocess.run(stats_cmd, capture_output=True, text=True, cwd=cwd)

    if stats_result.returncode != 0:
        print("❌ Stats script failed.")
        if stats_result.stderr.strip():
            print("stderr:\n" + stats_result.stderr)
        sys.exit(1)

    if stats_result.stdout.strip():
        print(stats_result.stdout.strip())

    print("✅ PDF report generation complete.")
    print(f"PDF_FILE:{pdf_path.name}")  # marker for Flask app

    return str(pdf_path)

if __name__ == "__main__":
    main()
