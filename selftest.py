"""
selftest.py -- runs the full unit test suite programmatically and writes
the result to selftest_results.json.

This exists for the UNATTENDED services (see scripts/install_services.ps1),
not as a replacement for normal development testing
(`python -m unittest discover -s tests -v` is still the right tool for
that). A long-running, unattended deployment can silently drift out from
under you -- a corrupted venv, a bad dependency upgrade, an accidental
edit to a file while the services keep running on the old in-memory code.
Running the real test suite on a schedule (and once at system startup)
catches that kind of drift instead of only discovering it the next time a
human happens to look. This is a health check, not a substitute for
preflight.py -- preflight checks the ACCOUNT is safe to trade against;
this checks the CODE still behaves the way its own tests say it should.

Exit code is 0 if the full suite passed, 1 otherwise, so a scheduled task
runner can flag a failure loudly (see Task Scheduler's "last run result").
"""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

RESULTS_FILE = "selftest_results.json"
_PROJECT_ROOT = Path(__file__).resolve().parent


def run_selftest() -> dict:
    # Match exactly how `python -m unittest discover -s tests` behaves from
    # the project root -- tests/ has no __init__.py, so discovery only
    # works when start_dir and top_level_dir coincide (the default when cwd
    # IS the project root). selftest.py can be invoked from anywhere (e.g.
    # a scheduled task with a different working directory), so pin cwd here.
    os.chdir(_PROJECT_ROOT)
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests")
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=1)
    result = runner.run(suite)

    def _fmt(entries):
        return [{"test": str(test), "message": msg.strip()[-2000:]} for test, msg in entries]

    return {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "total": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "passed": result.wasSuccessful(),
        "failure_details": _fmt(result.failures),
        "error_details": _fmt(result.errors),
    }


def main() -> int:
    summary = run_selftest()
    out_path = _PROJECT_ROOT / RESULTS_FILE
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    status = "PASS" if summary["passed"] else "FAIL"
    print(
        f"[selftest] {status}: {summary['total']} tests, "
        f"{summary['failures']} failures, {summary['errors']} errors"
    )
    if not summary["passed"]:
        for f in summary["failure_details"] + summary["error_details"]:
            print(f"  - {f['test']}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
