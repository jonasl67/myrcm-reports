#!/usr/bin/env python3
"""
test_myrcm.py

Runs myrcm_runner.py against all test cases listed in myrcm_test_cases.txt.
Supports optional expected return codes with syntax: ; expect=N
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
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return -999, "", f"Exception: {e}"

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

            # Split off expected return code if present
            expected_rc = 0
            if "; expect=" in line:
                line, rc_part = line.split("; expect=", 1)
                try:
                    expected_rc = int(rc_part.strip())
                except ValueError:
                    expected_rc = 0
            parts = shlex.split(line)

            description = last_comment if last_comment else "(no description)"
            print(f"\n=== Running: {description} ===")
            print(f"Command args: {parts} (expect {expected_rc})")

            rc, stdout, stderr = run_test_case(parts)
            ok = (rc == expected_rc)

            results.append((description, line, expected_rc, rc, ok, stdout, stderr))

            status = "✅ PASS" if ok else f"❌ FAIL (got {rc}, expected {expected_rc})"
            print(status)
            if stdout:
                print("stdout:", stdout)
            if stderr:
                print("stderr:", stderr)

            last_comment = None

    print("\n\n=== SUMMARY ===")
    for desc, cmd_line, exp_rc, rc, ok, stdout, stderr in results:
        status = "✅ PASS" if ok else f"❌ FAIL (got {rc}, expected {exp_rc})"
        print(f"{status} | {desc}")
        print(f"    Command: {cmd_line}")
        if not ok:
            first_err = stderr.splitlines()[0] if stderr else "(no stderr)"
            print(f"    -> {first_err}")

if __name__ == "__main__":
    main()
