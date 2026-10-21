"""Run dictation lifecycle checks with a fake browser speech engine (no audio or network)."""

from pathlib import Path
import subprocess


def test_template_dictation_lifecycle():
    result = subprocess.run(
        ["node", "--test", str(Path(__file__).with_name("template_dictation.test.cjs"))],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
