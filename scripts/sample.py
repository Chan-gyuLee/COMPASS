"""Run after `python -m pip install -e '.[bioemu]'` from the repository root."""

from compass.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
