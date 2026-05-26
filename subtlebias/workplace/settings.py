from pathlib import Path

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        raise SystemExit("pip install tomli  (required for Python < 3.11)")

DEFAULT_CONFIG = Path(__file__).parent.parent / "config.toml"


def load(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.exists():
        raise SystemExit(f"Config file not found: {p}")
    with open(p, "rb") as f:
        return tomllib.load(f)
