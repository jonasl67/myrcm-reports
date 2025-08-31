#!/usr/bin/env python3
"""
test_myrcm.py

Runs myrcm_runner.py against all test cases listed in myrcm_test_cases.txt.
Reports success/failure for each case, along with the descriptive comment
from the test file for easier manual checking of the generated PDFs.
"""

import subprocess
import shlex

RUNNER_SCRIPT = "myrcm_runner.py"
TEST_FILE = "myrcm_test_cases.txt"

def run_test_case(parts):
    """Run one test case with given CLI parts (list of args)."""
    cmd = ["python", RUNNER_SCRIPT] + parts
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        success = result.returncode == 0
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        return success, stdout, stderr
    except Exception as e:
        return False, "", f"Exception: {e}"

def main():
    results = []
    last_comment = None

    with open(TEST_FILE, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("#"):
                last_comment = line.lstrip("#").strip()
                continue

            # Use shlex.split so quoted params work correctly
            parts = shlex.split(line)

            description = last_comment if last_comment else "(no description)"
            print(f"\n=== Running: {description} ===")
            print(f"Command args: {parts}")

            ok, stdout, stderr = run_test_case(parts)
            results.append((description, line, ok, stdout, stderr))

            status = "✅ SUCCESS" if ok else "❌ FAIL"
            print(f"{status}")
            if stdout:
                print("stdout:", stdout)
            if stderr:
                print("stderr:", stderr)

            last_comment = None

    print("\n\n=== SUMMARY ===")
    for desc, cmd_line, ok, stdout, stderr in results:
        status = "✅ SUCCESS" if ok else "❌ FAIL"
        print(f"{status} | {desc}")
        print(f"    Command: {cmd_line}")
        if not ok:
            first_err = stderr.splitlines()[0] if stderr else "(no stderr)"
            print(f"    -> {first_err}")

if __name__ == "__main__":
    main()
