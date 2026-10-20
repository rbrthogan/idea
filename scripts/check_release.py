#!/usr/bin/env python3
"""Check public source for accidental credentials and local-only files.

The default checks tracked and unignored source in the working tree. Use --index
before committing to check the exact staged files, or --history to audit the
current branch's ancestry. Findings show only paths and rule names, never values.
This is a focused safety check, not a guarantee that every secret is detectable.
"""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import re
import subprocess


LOCAL_DIRECTORIES = {
    ".codex-jobs", ".deploy", ".firebase", ".playwright-cli", ".pytest_cache",
    ".ruff_cache", ".superpowers", ".terraform", ".venv", "__pycache__",
    "artifacts", "build", "data", "dist", "htmlcov", "node_modules", "output", "venv",
}
SECRET_PATTERNS = {
    "private key": rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "Google API key": rb"AIza[0-9A-Za-z_-]{35}",
    "provider secret": rb"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{32,}",
    "GitHub token": rb"\bgh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{60,}",
    "AWS access key": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "OAuth client secret": rb"GOCSPX-[A-Za-z0-9_-]{20,}",
    "Slack token": rb"\bxox[baprs]-[A-Za-z0-9-]{20,}",
}


def local_only(path: str) -> bool:
    parts = PurePosixPath(path).parts
    name = parts[-1] if parts else ""
    if (
        bool(parts and parts[0] in LOCAL_DIRECTORIES)
        or path.startswith("archive/internal-docs/")
        or any(part.startswith("firebase-export-") for part in parts)
        or any(part in {".terraform", "__pycache__", "node_modules"}
               or part.endswith(".egg-info") for part in parts)
    ):
        return True
    if name.endswith(".example"):
        return False
    return (
        name == ".env"
        or name.startswith(".env.")
        or path == "idea/static/js/auth.js"
        or name in {".DS_Store", ".coverage", ".cloudignore", "backend.hcl",
                    "application_default_credentials.json"}
        or name.endswith((".pem", ".key", ".tfvars", ".tfvars.json", ".tfplan", ".log", ".pyc"))
        or ".tfstate" in name
        or bool(re.search(r"service[-_]account.*\.json$", name))
    )


def findings(path: str, content: bytes) -> list[str]:
    result = ["local-only file"] if local_only(path) else []
    # Image/other binary assets are valid source. Text credentials are not.
    if b"\0" not in content:
        result.extend(
            label for label, pattern in SECRET_PATTERNS.items()
            if re.search(pattern, content)
        )
    return result


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def source_files(root: Path, *, index: bool = False):
    args = ["ls-files", "-z", "--cached"]
    if not index:
        args.extend(["--others", "--exclude-standard"])
    for path in sorted(set(git(root, *args).decode().split("\0")) - {""}):
        if index:
            yield path, git(root, "show", f":{path}")
        elif (root / path).is_file():
            yield path, (root / path).read_bytes()


def history_files(root: Path, revision: str = "HEAD"):
    objects = git(root, "rev-list", "--objects", revision).splitlines()
    # One Git process avoids spawning twice for every historical object.
    with subprocess.Popen(
        ["git", "-C", str(root), "cat-file", "--batch"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    ) as process:
        for entry in objects:
            oid, _, path = entry.partition(b" ")
            if not path:
                continue
            process.stdin.write(oid + b"\n")
            process.stdin.flush()
            _, kind, size = process.stdout.readline().split()
            content = process.stdout.read(int(size))
            process.stdout.read(1)  # Batch records end with a newline.
            if kind == b"blob":
                yield path.decode(errors="replace"), content
        process.stdin.close()
        if process.wait():
            raise RuntimeError("Git history read failed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--index", action="store_true", help="check exact staged content")
    mode.add_argument("--history", action="store_true", help="audit current branch ancestry")
    args = parser.parse_args()
    root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())
    files = history_files(root) if args.history else source_files(root, index=args.index)
    checked, failures = 0, set()
    for path, content in files:
        checked += 1
        for rule in findings(path, content):
            failures.add((path, rule))
    for path, rule in sorted(failures):
        print(f"{path}: {rule}")
    print(f"Checked {checked} files; {len(failures)} findings. Secret values are never printed.")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
