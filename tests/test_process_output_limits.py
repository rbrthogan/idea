import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from idea.evaluation.processes import run_bounded, ProcessOutputLimitError


def test_bounded_capture_drains_both_streams():
    result = run_bounded([sys.executable, '-c', "import os;os.write(1,b'ok');os.write(2,b'error')"],
                         timeout=3, stdout_limit=2, stderr_limit=5)
    assert result.stdout == b'ok'
    assert result.stderr == b'error'
    assert result.returncode == 0


@pytest.mark.parametrize('descriptor', [1, 2])
def test_infinite_output_is_terminated_at_byte_cap(descriptor):
    with pytest.raises(ProcessOutputLimitError):
        run_bounded([sys.executable, '-c', f'import os\nwhile True: os.write({descriptor},b"x"*4096)'],
                    timeout=3, stdout_limit=128, stderr_limit=128)


def test_silent_process_timeout_is_reaped():
    with pytest.raises(subprocess.TimeoutExpired):
        run_bounded([sys.executable, '-c', 'import time;time.sleep(20)'], timeout=0.1, stdout_limit=128)


def test_wrapper_does_not_expose_saved_stdout_descriptors(tmp_path):
    runner = Path(__file__).resolve().parents[1] / 'docker/artifact-runner/runner.py'
    source = tmp_path / 'attack.py'
    source.write_text('import os\nos.write(3,b"LEAKED")\n')
    result = subprocess.run([sys.executable, str(runner), str(source)], capture_output=True, timeout=3)
    assert b'LEAKED' not in result.stdout
    assert result.returncode == 1


def test_result_file_is_bounded_before_read(tmp_path):
    ns = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'docker/artifact-runner/runner.py'))
    path = tmp_path / 'result.json'
    with path.open('wb') as stream:
        stream.truncate(ns['MAX_RESULT_BYTES'] + 1)
    with pytest.raises(ValueError, match='exceeds'):
        ns['_validate_result'](path)


def test_container_wrapper_has_its_own_deadline(tmp_path):
    import os
    runner = Path(__file__).resolve().parents[1] / 'docker/artifact-runner/runner.py'
    source = tmp_path / 'loop.py'
    source.write_text('while True: pass\n')
    result = subprocess.run([sys.executable, str(runner), str(source)],
        env={**os.environ, 'IDEA_EXECUTION_TIMEOUT': '1'}, capture_output=True, timeout=3)
    assert result.returncode == 124
    assert b'execution deadline' in result.stderr
