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

def main():
    if len(sys.argv) < 3:
        print("Usage: python myrcm_runner.py <url> <search terms...> [--verbose]")
        sys.exit(1)

    url = sys.argv[1]
    args = sys.argv[2:]

    # Step 1: Run scraper
    scraper_cmd = ["python", "myrcm_scraper.py", url] + args
    result = subprocess.run(scraper_cmd, capture_output=True, text=True)

    stdout_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    stderr_lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]

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
    stats_cmd = ["python", "myrcm_stats.py", csv_file]
    if "--verbose" in args:
        stats_cmd.append("--verbose")

    stats_result = subprocess.run(stats_cmd, capture_output=True, text=True)

    if stats_result.returncode != 0:
        print("❌ Stats script failed.")
        if stats_result.stderr.strip():
            print("stderr:\n" + stats_result.stderr)
        sys.exit(1)

    print(stats_result.stdout.strip())
    print("✅ PDF report generation complete.")

if __name__ == "__main__":
    main()
