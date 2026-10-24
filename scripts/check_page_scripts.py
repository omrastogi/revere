"""
Syntax-check the inline <script> blocks in Revere's static HTML pages.

Usage:
    python scripts/check_page_scripts.py                 # all static/*.html
    python scripts/check_page_scripts.py static/x.html   # specific files

Uses Node if available, otherwise macOS JavaScriptCore (osascript -l JavaScript).
Exits 1 if any script fails to parse. Used by the "Check page scripts on save" Kiro hook.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

SCRIPT_RE = re.compile(r"<script>(.*?)</script>", re.S)  # inline scripts only (no src=)


def check_js(src: str) -> str | None:
    """Return None if `src` parses, otherwise an error message."""
    if shutil.which("node"):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(src)
            path = f.name
        try:
            r = subprocess.run(["node", "--check", path], capture_output=True, text=True)
            return None if r.returncode == 0 else (r.stderr.strip() or "parse error")
        finally:
            os.unlink(path)
    if shutil.which("osascript"):
        probe = "try{new Function(" + json.dumps(src) + ");'OK'}catch(e){'ERR '+e}"
        r = subprocess.run(["osascript", "-l", "JavaScript", "-e", probe], capture_output=True, text=True)
        out = (r.stdout or r.stderr).strip()
        return None if out == "OK" else out
    return "no JavaScript engine found (install Node.js)"


def main(argv: list[str]) -> int:
    files = argv or sorted(glob.glob("static/*.html"))
    failed = False
    for path in files:
        if not path.endswith(".html") or not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            blocks = SCRIPT_RE.findall(f.read())
        for i, block in enumerate(blocks, 1):
            err = check_js(block)
            if err:
                failed = True
                print(f"FAIL {path} script #{i}: {err}")
            else:
                print(f"ok   {path} script #{i}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
