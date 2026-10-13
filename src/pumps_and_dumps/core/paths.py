import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

DATA_DIR_ENV = "PUMPS_AND_DUMPS_DATA_DIR"
DATA_CONFIG_NAME = ".pumps-and-dumps.json"
DEFAULT_DATA_DIR = Path("~/dev/pumps_and_dumps/data").expanduser()


def get_root_dir() -> Path:
    """Find this checkout from the installed source, independent of the caller's cwd.

    Editable installs resolve through ``src/pumps_and_dumps``. A wheel installed
    into a separate environment can also use a checkout as its working directory.
    Requiring the package directory avoids selecting another project's pyproject.
    """
    cwd = Path.cwd().resolve()
    for directory in (*Path(__file__).resolve().parents, cwd, *cwd.parents):
        if (directory / "pyproject.toml").is_file() and (
            directory / "src" / "pumps_and_dumps"
        ).is_dir():
            return directory
    raise RuntimeError(
        "Project checkout not found; run poetry install in the pumps_and_dumps repository"
    )


def get_data_dir() -> Path:
    """Use an environment override, the installer's saved directory, or the original default."""
    if override := os.environ.get(DATA_DIR_ENV):
        return Path(override).expanduser().resolve()
    try:
        config_path = get_root_dir() / DATA_CONFIG_NAME
    except RuntimeError:
        # Importing the installed package outside a checkout should still work.
        return DEFAULT_DATA_DIR
    if not config_path.exists():
        return DEFAULT_DATA_DIR
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        value = config["data_dir"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("data_dir must be a nonempty path")
        directory = Path(value).expanduser()
        if not directory.is_absolute():
            directory = config_path.parent / directory
        return directory.resolve()
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Invalid data directory configuration in {config_path}: {exc}"
        ) from exc


def save_data_dir(directory: Path) -> Path:
    """Atomically persist the selected directory for subsequent commands and notebooks."""
    config_path = get_root_dir() / DATA_CONFIG_NAME
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=config_path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            json.dump(
                {"data_dir": str(directory.expanduser().resolve())}, temporary, indent=2
            )
            temporary.write("\n")
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        temporary_path.replace(config_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return config_path


DATA_DIR: Path = get_data_dir()
# Each split policy has its own studies so previous held-out periods cannot leak into tuning.
SQLITE_URL: str = f"sqlite:///{DATA_DIR / 'studies_split_60_10_20_10_v1.db'}"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
TRANSFORMED_DATA_DIR: Path = DATA_DIR / "transformed"

# Raw zip files
BINANCE_SPOT_RAW_TRADES: Path = RAW_DATA_DIR / "binance" / "spot" / "trades"
BINANCE_SPOT_RAW_KLINES: Path = RAW_DATA_DIR / "binance" / "spot" / "klines"
# HIVE locations
BINANCE_SPOT_HIVE_TRADES: Path = TRANSFORMED_DATA_DIR / "binance" / "spot" / "trades"
FEATURE_DIR: Path = DATA_DIR.joinpath("features")
