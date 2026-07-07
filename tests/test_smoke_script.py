from __future__ import annotations

import subprocess
import sys


def test_smoke_script_help_runs_from_checkout() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/smoke_deepseek.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--text-only" in result.stdout
