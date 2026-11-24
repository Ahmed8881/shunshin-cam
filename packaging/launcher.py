"""Entry point for the standalone executable (PyInstaller needs a plain script)."""

import os
import sys

from shunshin.__main__ import main

if __name__ == "__main__":
    # Dropping an image onto the .exe starts it with that picture as the virtual background.
    if len(sys.argv) == 2 and os.path.isfile(sys.argv[1]):
        sys.argv.insert(1, "--background")
    try:
        main()
    except SystemExit as e:
        if e.code not in (None, 0):
            print(e.code)
    finally:
        # Double-clicked windows close instantly; keep them open so messages can be read.
        if getattr(sys, "frozen", False) and sys.stdin and sys.stdin.isatty():
            try:
                input("\nPress Enter to close...")
            except EOFError:
                pass
