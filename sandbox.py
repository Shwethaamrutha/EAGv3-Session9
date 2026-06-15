"""Sandbox executor — runs Python code in an isolated subprocess.

This is a usability boundary, not a security one. The subprocess runs with
most of the host's privileges. Environment variables are scrubbed, the working
directory is a temp directory, and a timeout applies, but the subprocess can
read filesystem paths and call out to the network. Appropriate for student code;
not appropriate for untrusted sources.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass


ALLOWED_ENV = {"PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE"}
TIMEOUT_SECONDS = 10
MAX_OUTPUT_BYTES = 1_000_000


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


def run_code(code: str) -> SandboxResult:
    env = {k: v for k, v in os.environ.items() if k in ALLOWED_ENV}

    with tempfile.TemporaryDirectory(prefix="sandbox_") as tmpdir:
        script_path = os.path.join(tmpdir, "script.py")
        with open(script_path, "w") as f:
            f.write(code)

        try:
            result = subprocess.run(
                ["python3", script_path],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                cwd=tmpdir,
                env=env,
            )
            return SandboxResult(
                stdout=result.stdout[:MAX_OUTPUT_BYTES],
                stderr=result.stderr[:MAX_OUTPUT_BYTES],
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                stdout="",
                stderr=f"[TIMEOUT] Script exceeded {TIMEOUT_SECONDS}s limit.",
                exit_code=-1,
                timed_out=True,
            )
        except Exception as e:
            return SandboxResult(
                stdout="",
                stderr=f"[ERROR] Sandbox error: {e}",
                exit_code=-1,
            )
