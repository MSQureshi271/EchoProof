"""Allow ``python -m echoproof`` to use the same CLI entry point."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())

