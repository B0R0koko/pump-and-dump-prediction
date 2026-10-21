"""Exercise environment setup through the installer without touching a checkout.

The environment helper is replaced with a recorder: these tests check command
selection, argument forwarding and the absence of unrelated dataset operations.
Separate helper tests cover Git, package installation and environment contents.
"""

import pytest

from installer import cli, menu
from installer.menu import Selection, SetupMenu


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("update", [False, True])
def test_explicit_environment_setup_bypasses_data_and_docker(tmp_path, monkeypatch, dry_run, update):
    """An explicit setup request dispatches only the research environment helper.

    A nonexistent checkout stand-in makes filesystem side effects observable.
    Every data-related operation, Docker executable lookup and terminal probe
    is replaced with a failing stub. The helper must receive exactly this root,
    the requested interpreter, update policy and dry-run flag. Its nonzero status must
    reach the caller unchanged. No data directory prompt is allowed, even when
    neither --data-dir nor --no-tui was supplied.
    """
    root = tmp_path / "checkout"
    calls = []

    def forbidden(*_args, **_kwargs):
        pytest.fail("environment setup reached the data pipeline or terminal probing")

    def setup(path, **kwargs):
        calls.append((path, kwargs))
        return 17

    monkeypatch.setattr(cli, "get_root_dir", lambda: root)
    monkeypatch.setattr(cli, "setup_environment", setup)
    for name in (
        "get_data_dir",
        "_prepare_data_dir",
        "_validate_selection",
        "_build_stages",
        "run_tui",
        "tui_available",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(cli.shutil, "which", forbidden)
    monkeypatch.setattr("builtins.input", forbidden)
    arguments = ["--setup-environment", "--python", "/custom/python3.13", "--tui"]
    if dry_run:
        arguments.append("--dry-run")
    if not update:
        arguments.append("--no-update")
    assert cli.main(arguments) == 17
    assert calls == [(root, {"python": "/custom/python3.13", "dry_run": dry_run, "update": update})]
    assert not root.exists()


def test_menu_environment_setup_returns_to_console_without_data_work(tmp_path, monkeypatch):
    """Choosing setup in the terminal menu runs the same helper as the CLI flag.

    The menu returns an empty stage list plus the explicit environment marker.
    The installer must close the menu and dispatch setup with the selected
    Python instead of validating nonexistent data, persisting its path, building
    containers or launching the data progress dashboard. Unlike explicit CLI
    setup, this covers the normal interactive selection branch.
    """
    root = tmp_path / "checkout"
    data = tmp_path / "not-created"
    calls = []

    def forbidden(*_args, **_kwargs):
        pytest.fail("menu environment selection reached data work")

    def choose(data_dir, dates, configured_stages):
        assert data_dir == data
        assert configured_stages == set()
        return Selection(data_dir, [], setup_environment=True)

    monkeypatch.setattr(cli, "get_root_dir", lambda: root)
    monkeypatch.setattr(cli, "get_data_dir", lambda: data)
    monkeypatch.setattr(cli, "tui_available", lambda: True)
    monkeypatch.setattr(cli, "choose_installation", choose)
    monkeypatch.setattr(cli, "setup_environment", lambda path, **kwargs: calls.append((path, kwargs)) or 0)
    for name in ("_prepare_data_dir", "_validate_selection", "_build_stages", "run_tui"):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr("builtins.input", forbidden)
    assert cli.main(["--python", "python3.13"]) == 0
    assert calls == [(root, {"python": "python3.13", "dry_run": False, "update": True})]
    assert not data.exists()


def test_menu_third_choice_skips_dataset_prerequisites(tmp_path, monkeypatch):
    """Two downward moves select setup, even when there is no market data.

    The first two positions remain full installation and custom stages. The
    third must return a marked environment-only selection immediately; a failing
    prerequisite-validator stub proves it cannot reject setup for absent Hive
    files or archives. Further navigation must wrap through all three choices.
    """

    def forbidden(*_args, **_kwargs):
        pytest.fail("environment menu choice validated dataset inputs")

    monkeypatch.setattr(menu, "validate_inputs", forbidden)
    chooser = SetupMenu(tmp_path, "Dates")
    chooser.handle("j")
    chooser.handle("j")
    assert chooser.focus == 2
    assert chooser.handle("\n") == Selection(tmp_path, [], setup_environment=True)
    chooser.handle("j")
    assert chooser.focus == 0
    chooser.handle("k")
    assert chooser.focus == 2


def test_environment_choice_fits_minimum_terminal_height(tmp_path):
    """The new third card remains readable in the supported 72 by 24 terminal.

    A screen recorder exposes exactly what the real Canvas would draw. The
    setup title must appear above the footer, and existing data-stage guidance
    must not overwrite it. This catches a menu that works by keyboard while
    silently placing its new choice below the visible terminal area.
    """

    class Screen:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 24, 72

        def erase(self):
            self.writes.clear()

        def refresh(self):
            pass

        def addnstr(self, y, x, text, size, style):
            self.writes.append((y, x, text[:size]))

    screen = Screen()
    chooser = SetupMenu(tmp_path, "Dates")
    chooser.focus = 2
    chooser.draw(screen)
    title = [(y, text) for y, _x, text in screen.writes if "Set up research environment" in text]
    assert title == [(18, "  Set up research environment  ")]
    assert any(y == 23 and "Navigate" in text for y, _x, text in screen.writes)


@pytest.mark.parametrize("pipeline", [["--all"], ["--stages", "features"]])
def test_environment_flag_cannot_be_combined_with_pipeline(pipeline):
    """One invocation cannot both update its checkout and launch data stages.

    Argparse must reject setup together with either the full pipeline or an
    explicit stage list. Keeping those scopes mutually exclusive makes it
    clear which work will run before any Git update, downloads or config writes.
    """
    with pytest.raises(SystemExit) as raised:
        cli._build_parser().parse_args(["--setup-environment", *pipeline])
    assert raised.value.code == 2


@pytest.mark.parametrize("environment_options", [["--python", "python3.13"], ["--no-update"]])
def test_environment_options_are_not_silently_ignored_for_pipeline(monkeypatch, capsys, environment_options):
    """A custom interpreter or code-update policy applies only to environment setup.

    A console pipeline invocation with either option must fail before prompting for
    data or launching Docker. This prevents a plausible-looking command from
    silently running container stages when the user intended local setup.
    """

    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid interpreter scope started installation")

    monkeypatch.setattr(cli, "_install", forbidden)
    with pytest.raises(SystemExit) as raised:
        cli.main(["--all", "--no-tui", *environment_options])
    assert raised.value.code == 2
    assert "require --setup-environment" in capsys.readouterr().err
