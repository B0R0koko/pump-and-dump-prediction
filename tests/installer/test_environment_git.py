"""Exercise environment setup updates against disposable, local Git repositories.

Every test uses its own bare remote and two tiny checkouts. No command contacts
GitHub or modifies the working repository. Literal file contents and commit IDs
make it possible to verify both the expected update and preservation of local
work without mocking the Git commands that enforce those guarantees.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
import subprocess

import pytest

from installer import environment
from installer.environment import _update_checkout

LOGGER = logging.getLogger("installer.tests.environment.git")


def _git(root: Path, *arguments: str) -> str:
    """Run Git only in a test-created directory and return its stripped output."""
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.stdout.strip()


def _identify(root: Path) -> None:
    """Give a disposable checkout an identity without touching global Git config."""
    _git(root, "config", "user.name", "Installer Test")
    _git(root, "config", "user.email", "installer-test@example.invalid")
    _git(root, "config", "commit.gpgsign", "false")


@dataclass
class GitCheckouts:
    """The consumer being updated and the independent author of remote commits."""

    consumer: Path
    publisher: Path

    def publish_update(self) -> str:
        """Publish one unmistakable source change and return its new commit ID."""
        (self.publisher / "project.txt").write_text("remote version 2\n", encoding="utf-8")
        _git(self.publisher, "add", "project.txt")
        _git(self.publisher, "commit", "-m", "Publish version 2")
        _git(self.publisher, "push", "origin", "main")
        return _git(self.publisher, "rev-parse", "HEAD")


@pytest.fixture
def git_checkouts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GitCheckouts:
    """Create a clean tracking checkout and a local remote with one small commit.

    System and user configuration are disabled so signing, pull policy, hooks,
    and a developer's identity cannot change these tests. Both environment and
    log directories are ignored, mirroring installer-generated local files. A
    space in the checkout path also checks argument handling without a shell.
    """
    if shutil.which("git") is None:
        pytest.skip("Git is needed for local repository integration tests")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "absent-global-config"))
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    remote = tmp_path / "remote.git"
    publisher = tmp_path / "publisher"
    consumer = tmp_path / "consumer checkout"
    _git(tmp_path, "init", "--bare", "--initial-branch=main", str(remote))
    _git(tmp_path, "clone", str(remote), str(publisher))
    _identify(publisher)
    (publisher / ".gitignore").write_text("/.venv/\n/logs/\n", encoding="utf-8")
    (publisher / "project.txt").write_text("initial version\n", encoding="utf-8")
    _git(publisher, "add", ".")
    _git(publisher, "commit", "-m", "Initial version")
    _git(publisher, "push", "-u", "origin", "main")
    _git(tmp_path, "clone", str(remote), str(consumer))
    _identify(consumer)
    return GitCheckouts(consumer=consumer, publisher=publisher)


def test_checkout_fast_forwards_to_latest_tracking_commit(git_checkouts: GitCheckouts) -> None:
    """A clean checkout receives exactly the newly published tracked commit.

    The consumer begins with ``initial version`` and the remote advances once
    to ``remote version 2``. Setup must return True, move HEAD to that exact
    remote commit, and expose the new file contents on the same main branch.
    Matching the commit ID also rules out silently creating a merge commit.
    """
    expected_commit = git_checkouts.publish_update()

    assert _update_checkout(git_checkouts.consumer, LOGGER) is True

    assert _git(git_checkouts.consumer, "rev-parse", "HEAD") == expected_commit
    assert _git(git_checkouts.consumer, "branch", "--show-current") == "main"
    assert (git_checkouts.consumer / "project.txt").read_text(encoding="utf-8") == "remote version 2\n"


def test_current_checkout_returns_false_without_new_commit(git_checkouts: GitCheckouts) -> None:
    """Updating an already current branch reports that no restart is necessary.

    Both local and remote HEAD initially refer to the same single commit. The
    return value must be False and HEAD and source contents must stay unchanged,
    so callers can distinguish a real update from an ordinary successful pull.
    """
    before = _git(git_checkouts.consumer, "rev-parse", "HEAD")

    assert _update_checkout(git_checkouts.consumer, LOGGER) is False

    assert _git(git_checkouts.consumer, "rev-parse", "HEAD") == before
    assert (git_checkouts.consumer / "project.txt").read_text(encoding="utf-8") == "initial version\n"


@pytest.mark.parametrize("local_change", ["unstaged", "staged", "untracked"])
def test_dirty_checkout_preserves_work_and_does_not_update(
    git_checkouts: GitCheckouts,
    local_change: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Uncommitted work skips the pull while allowing local installation to proceed.

    The three cases cover modified tracked content, an explicitly staged edit,
    and a new untracked file. Although version 2 exists remotely, setup must
    return False and explain the skipped pull in both console output and a log
    warning. HEAD, the remote-tracking ref, status, staged bytes, and working-file
    bytes must remain unchanged. Absence of FETCH_HEAD also proves that no fetch
    occurred before the decision; there is no stash, reset, or silent update.
    """
    root = git_checkouts.consumer
    git_checkouts.publish_update()
    changed_path = root / ("local-notes.txt" if local_change == "untracked" else "project.txt")
    changed_path.write_text("keep my unfinished local work\n", encoding="utf-8")
    if local_change == "staged":
        _git(root, "add", "project.txt")
    before_head = _git(root, "rev-parse", "HEAD")
    before_remote = _git(root, "rev-parse", "origin/main")
    before_status = _git(root, "status", "--porcelain", "--untracked-files=all")
    before_index = _git(root, "diff", "--cached", "--binary")
    before_contents = changed_path.read_bytes()

    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        assert _update_checkout(root, LOGGER) is False

    warning = "Skipping Git pull: the checkout has local changes. Installing the current local code."
    assert warning in capsys.readouterr().out
    assert [(record.levelname, record.getMessage()) for record in caplog.records] == [("WARNING", warning)]
    assert _git(root, "rev-parse", "HEAD") == before_head
    assert _git(root, "rev-parse", "origin/main") == before_remote
    assert _git(root, "status", "--porcelain", "--untracked-files=all") == before_status
    assert _git(root, "diff", "--cached", "--binary") == before_index
    assert changed_path.read_bytes() == before_contents
    assert not (root / ".git" / "FETCH_HEAD").exists()


def test_default_setup_installs_dirty_local_checkout_and_logs_success(
    git_checkouts: GitCheckouts, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Default setup completes from local files when a Git update cannot be applied safely.

    The remote has a newer commit, while the consumer has different staged and
    unstaged versions of its tracked file plus an untracked research note. Only
    dependency installation is replaced with a recorder: actual setup dispatch,
    Git inspection, and .log file creation run unchanged. The recorder must see
    the local working content, setup must return success, and the log must record
    both the skipped pull and completion. Git refs, configuration, index contents,
    and all user files must retain their original values, with no remote fetch.
    """
    root = git_checkouts.consumer
    git_checkouts.publish_update()
    source = root / "project.txt"
    source.write_text("staged local version\n", encoding="utf-8")
    _git(root, "add", "project.txt")
    source.write_text("unfinished local version\n", encoding="utf-8")
    note = root / "research-notes.txt"
    note.write_text("keep this untracked note\n", encoding="utf-8")
    before_head = _git(root, "rev-parse", "HEAD")
    before_remote = _git(root, "rev-parse", "origin/main")
    before_status = _git(root, "status", "--porcelain", "--untracked-files=all")
    before_index = _git(root, "diff", "--cached", "--binary")
    before_config = (root / ".git" / "config").read_bytes()
    before_source = source.read_bytes()
    before_note = note.read_bytes()
    installed_sources = []

    def record_install(checkout: Path, requested: str | None, logger: logging.Logger) -> None:
        assert checkout == root
        assert requested is None
        installed_sources.append((checkout / "project.txt").read_text(encoding="utf-8"))

    monkeypatch.setattr(environment, "_install_environment", record_install)

    assert environment.setup_environment(root) == 0

    assert installed_sources == ["unfinished local version\n"]
    console = capsys.readouterr()
    assert "Skipping Git pull:" in console.out
    assert "Research environment ready:" in console.out
    assert console.err == ""
    log_files = list((root / "logs").glob("setup-environment-*.log"))
    assert len(log_files) == 1
    log = log_files[0].read_text(encoding="utf-8")
    assert "WARNING" in log
    assert "Skipping Git pull: the checkout has local changes. Installing the current local code." in log
    assert "Research environment setup completed." in log
    assert _git(root, "rev-parse", "HEAD") == before_head
    assert _git(root, "rev-parse", "origin/main") == before_remote
    assert _git(root, "status", "--porcelain", "--untracked-files=all") == before_status
    assert _git(root, "diff", "--cached", "--binary") == before_index
    assert (root / ".git" / "config").read_bytes() == before_config
    assert source.read_bytes() == before_source
    assert note.read_bytes() == before_note
    assert not (root / ".git" / "FETCH_HEAD").exists()


def test_divergence_does_not_merge_or_rebase_local_commits(git_checkouts: GitCheckouts) -> None:
    """Independent local and remote commits cannot be resolved automatically.

    The consumer commits a local research note while the publisher updates a
    different source file. An ordinary merge would succeed without conflicts,
    so rejecting this case specifically proves the fast-forward-only policy.
    The local commit, its parent, and both local file contents must survive.
    """
    root = git_checkouts.consumer
    (root / "local-research.txt").write_text("committed local research\n", encoding="utf-8")
    _git(root, "add", "local-research.txt")
    _git(root, "commit", "-m", "Keep local research")
    before_history = _git(root, "log", "--format=%H", "-2")
    git_checkouts.publish_update()

    with pytest.raises((RuntimeError, ValueError)):
        _update_checkout(root, LOGGER)

    assert _git(root, "log", "--format=%H", "-2") == before_history
    assert (root / "local-research.txt").read_text(encoding="utf-8") == "committed local research\n"
    assert (root / "project.txt").read_text(encoding="utf-8") == "initial version\n"
    assert _git(root, "status", "--porcelain") == ""


def test_detached_checkout_is_not_switched_to_a_guessed_branch(git_checkouts: GitCheckouts) -> None:
    """A pinned detached revision is preserved instead of guessing main/master.

    Even when the remote has a newer main commit, setup must reject detached
    HEAD with a useful message. The original commit remains checked out and no
    branch becomes active, preserving a deliberately selected historical state.
    """
    root = git_checkouts.consumer
    before = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "--detach", before)
    git_checkouts.publish_update()

    with pytest.raises((RuntimeError, ValueError), match="(?i)detached|branch"):
        _update_checkout(root, LOGGER)

    assert _git(root, "rev-parse", "HEAD") == before
    assert _git(root, "branch", "--show-current") == ""
    assert (root / "project.txt").read_text(encoding="utf-8") == "initial version\n"


def test_missing_upstream_is_not_repaired_by_guessing(git_checkouts: GitCheckouts) -> None:
    """A named local branch still needs an explicit configured tracking branch.

    Removing only main's upstream leaves an origin remote and origin/main ref
    available, so a simplistic installer could guess them. Setup must instead
    report the missing tracking configuration and leave HEAD and source intact.
    """
    root = git_checkouts.consumer
    before = _git(root, "rev-parse", "HEAD")
    _git(root, "branch", "--unset-upstream")
    git_checkouts.publish_update()

    with pytest.raises((RuntimeError, ValueError), match="(?i)upstream|tracking"):
        _update_checkout(root, LOGGER)

    assert _git(root, "rev-parse", "HEAD") == before
    assert (root / "project.txt").read_text(encoding="utf-8") == "initial version\n"


def test_ignored_environment_and_logs_do_not_block_updates(git_checkouts: GitCheckouts) -> None:
    """A normal existing environment and setup logs are not treated as user edits.

    Two files under the explicitly ignored .venv and logs directories simulate
    repeat setup. A real source update must still succeed, while both generated
    files retain their exact content. This verifies that dirty detection respects
    Git ignore rules rather than scanning every file in the checkout.
    """
    root = git_checkouts.consumer
    environment_marker = root / ".venv" / "environment.txt"
    log_marker = root / "logs" / "setup.log"
    environment_marker.parent.mkdir()
    log_marker.parent.mkdir()
    environment_marker.write_text("existing environment\n", encoding="utf-8")
    log_marker.write_text("previous setup output\n", encoding="utf-8")
    expected_commit = git_checkouts.publish_update()

    assert _update_checkout(root, LOGGER) is True

    assert _git(root, "rev-parse", "HEAD") == expected_commit
    assert environment_marker.read_text(encoding="utf-8") == "existing environment\n"
    assert log_marker.read_text(encoding="utf-8") == "previous setup output\n"
