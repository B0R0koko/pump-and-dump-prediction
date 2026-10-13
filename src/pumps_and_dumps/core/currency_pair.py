"""Currency-pair value object and compatibility imports for older callers."""

from dataclasses import dataclass
from pathlib import Path

from pumps_and_dumps.core.time_utils import Bounds


@dataclass
class CurrencyPair:
    base: str
    term: str

    @classmethod
    def from_string(cls, symbol: str):
        """Parse CurrencyPair from string formatted like this: ADA-USDT"""
        base, term = symbol.split("-")
        return cls(base=base, term=term)  # type: ignore

    def __str__(self) -> str:
        return f"{self.base}-{self.term}"

    @property
    def name(self) -> str:
        return f"{self.base}-{self.term}"

    @property
    def binance_name(self) -> str:
        return f"{self.base}{self.term}"

    def __hash__(self) -> int:
        return hash(self.name)


def collect_all_spot_currency_pairs() -> list[CurrencyPair]:
    """Compatibility wrapper; new callers should use market_data.symbols."""
    from pumps_and_dumps.market_data.symbols import (
        collect_all_spot_currency_pairs as collect,
    )

    return collect()


def get_cross_section_currencies(hive_dir: Path, bounds: Bounds) -> list[CurrencyPair]:
    """Compatibility wrapper; new callers should use market_data.hive_catalog."""
    from pumps_and_dumps.market_data.hive_catalog import (
        get_cross_section_currencies as collect,
    )

    return collect(hive_dir, bounds)
