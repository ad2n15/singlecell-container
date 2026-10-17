"""``python -m scpipe run|resume|status|doctor ...`` (see scpipe.runner)."""

from .runner import main

if __name__ == "__main__":
    raise SystemExit(main())
