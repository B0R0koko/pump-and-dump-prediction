import json
from pathlib import Path

import pytest

from pumps_and_dumps.core.paths import get_root_dir
from pumps_and_dumps.core import paths


def test_get_root_dir_raises_when_no_checkout_is_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        paths, "__file__", str(tmp_path / "site-packages/pumps_and_dumps/core/paths.py")
    )
    with pytest.raises(RuntimeError, match="Project checkout not found"):
        get_root_dir()


def test_get_root_dir_finds_project_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """When run from anywhere under the project, get_root_dir returns the folder with pyproject.toml."""
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    assert project_root.joinpath("pyproject.toml").exists()

    monkeypatch.chdir(project_root)
    assert get_root_dir() == project_root

    # Also works from a nested directory.
    nested: Path = project_root / "src" / "pumps_and_dumps" / "core"
    monkeypatch.chdir(nested)
    assert get_root_dir() == project_root


def test_get_root_dir_uses_installed_source_outside_the_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A notebook or CLI in an unrelated project must not read that project's resources.
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "unrelated"\n')
    monkeypatch.chdir(tmp_path)
    assert get_root_dir() == Path(__file__).resolve().parents[2]


def test_wheel_install_can_find_a_checkout_from_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(
        paths, "__file__", str(tmp_path / "site-packages/pumps_and_dumps/core/paths.py")
    )
    monkeypatch.chdir(project_root / "notebooks")
    assert get_root_dir() == project_root


def test_saved_data_directory_is_independent_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(paths.DATA_DIR_ENV, raising=False)
    monkeypatch.setattr(paths, "get_root_dir", lambda: tmp_path)
    directory = tmp_path / "market data with spaces"
    config = paths.save_data_dir(directory)
    monkeypatch.chdir(tmp_path.parent)

    assert json.loads(config.read_text()) == {"data_dir": str(directory)}
    assert paths.get_data_dir() == directory
    # Reading and saving a preference must not create a dataset.
    assert not directory.exists()


def test_environment_takes_precedence_over_saved_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paths, "get_root_dir", lambda: tmp_path)
    paths.save_data_dir(tmp_path / "saved")
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "override"))
    assert paths.get_data_dir() == tmp_path / "override"


def test_data_directory_defaults_when_no_configuration_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(paths.DATA_DIR_ENV, raising=False)
    monkeypatch.setattr(paths, "get_root_dir", lambda: tmp_path)
    assert paths.get_data_dir() == paths.DEFAULT_DATA_DIR


def test_relative_saved_directory_is_resolved_from_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(paths.DATA_DIR_ENV, raising=False)
    monkeypatch.setattr(paths, "get_root_dir", lambda: tmp_path)
    (tmp_path / paths.DATA_CONFIG_NAME).write_text('{"data_dir": "custom/data"}')
    monkeypatch.chdir(tmp_path.parent)
    assert paths.get_data_dir() == tmp_path / "custom" / "data"


@pytest.mark.parametrize(
    "contents", ['{"data_dir": null}', '{"data_dir": ""}', "[]", "not json"]
)
def test_invalid_saved_directory_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contents: str
) -> None:
    monkeypatch.delenv(paths.DATA_DIR_ENV, raising=False)
    monkeypatch.setattr(paths, "get_root_dir", lambda: tmp_path)
    (tmp_path / paths.DATA_CONFIG_NAME).write_text(contents)
    with pytest.raises(ValueError, match="Invalid data directory configuration"):
        paths.get_data_dir()


def test_failed_config_replace_preserves_previous_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paths, "get_root_dir", lambda: tmp_path)
    config = paths.save_data_dir(tmp_path / "previous")
    original = config.read_bytes()

    def fail_replace(*_args):
        raise OSError("write failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="write failed"):
        paths.save_data_dir(tmp_path / "new")
    assert config.read_bytes() == original
    assert list(tmp_path.iterdir()) == [config]
