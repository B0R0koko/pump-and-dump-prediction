"""Read archive sizes and pagination markers from Binance's S3 listings."""

from xml.etree import ElementTree


def _archive_sizes(listing: ElementTree.Element) -> dict[str, int]:
    sizes = {}
    for entry in listing.findall("{*}Contents"):
        key = entry.findtext("{*}Key")
        if not key:
            raise ValueError("Binance archive listing contains an empty key")
        size = entry.findtext("{*}Size", default="")
        if not size.isdecimal():
            raise ValueError(f"Invalid archive size for {key}: {size!r}")
        sizes[key] = int(size)
    return sizes


def _next_marker(
    listing: ElementTree.Element, sizes: dict[str, int], previous: str | None
) -> str | None:
    truncated = listing.findtext("{*}IsTruncated")
    if truncated == "false":
        return None
    if truncated != "true":
        raise ValueError("Binance archive listing is missing a valid IsTruncated flag")
    marker = listing.findtext("{*}NextMarker") or next(reversed(sizes), None)
    if not marker:
        raise ValueError(
            "Truncated Binance archive listing contains no pagination marker"
        )
    if marker == previous:
        raise ValueError("Binance archive listing did not advance")
    return marker


def parse_listing(
    body: bytes, previous_marker: str | None = None
) -> tuple[dict[str, int], str | None]:
    """Return object sizes and the next marker, or raise for an invalid listing."""
    listing = ElementTree.fromstring(body)
    if listing.tag.rsplit("}", 1)[-1] != "ListBucketResult":
        raise ValueError("Invalid Binance archive listing")
    sizes = _archive_sizes(listing)
    return sizes, _next_marker(listing, sizes, previous_marker)
