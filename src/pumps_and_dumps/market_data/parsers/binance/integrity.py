"""Bounded-memory ZIP validation and atomic installation of downloaded archives."""

from io import BytesIO
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import threading
from typing import BinaryIO
import zipfile
import zlib

from pumps_and_dumps.core.progress import ProgressReporter


class ArchiveIntegrityError(ValueError):
    """An archive is incomplete, corrupt, or does not contain CSV data."""


def _archive_size(source: Path | BinaryIO) -> int:
    if isinstance(source, Path):
        return source.stat().st_size
    source.seek(0, os.SEEK_END)
    size = source.tell()
    source.seek(0)
    return size


def _verify_member(
    archive: zipfile.ZipFile, member: zipfile.ZipInfo, progress: ProgressReporter
) -> None:
    read_size = 0
    with archive.open(member) as stream:
        while chunk := stream.read(1024 * 1024):
            read_size += len(chunk)
            progress.update(advance=len(chunk))
    if read_size != member.file_size:
        raise ArchiveIntegrityError(f"incomplete member: {member.filename}")


def _verify_contents(source: Path | BinaryIO, progress: ProgressReporter) -> None:
    with zipfile.ZipFile(source) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if not any(member.filename.lower().endswith(".csv") for member in members):
            raise ArchiveIntegrityError("archive contains no CSV files")
        progress.update(total=sum(member.file_size for member in members), force=True)
        for member in members:
            _verify_member(archive, member, progress)


def verify_archive(
    source: Path | BinaryIO, *, expected_size: int | None = None, detail: str = ""
) -> None:
    """Read every member to EOF, checking decompression, lengths, and ZIP CRCs.

    Filesystem errors propagate: an unreadable file should fail the stage rather
    than be treated as an archive that can be repaired by downloading it again.
    """
    with ProgressReporter(
        f"verify:{threading.get_ident()}", unit="bytes", detail=detail
    ) as progress:
        size = _archive_size(source)
        if expected_size is not None and size != expected_size:
            raise ArchiveIntegrityError(
                f"size is {size:,} bytes; Binance lists {expected_size:,}"
            )
        try:
            _verify_contents(source, progress)
        except (
            zipfile.BadZipFile,
            EOFError,
            zlib.error,
            RuntimeError,
            NotImplementedError,
        ) as exc:
            raise ArchiveIntegrityError(str(exc)) from exc


def install_archive(
    path: Path, body: bytes, *, expected_size: int | None = None, detail: str = ""
) -> None:
    """Validate before replacing the destination; interrupted writes stay temporary."""
    verify_archive(BytesIO(body), expected_size=expected_size, detail=detail)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".part", delete=False
    )
    temporary = Path(stream.name)
    try:
        with stream:
            stream.write(body)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
