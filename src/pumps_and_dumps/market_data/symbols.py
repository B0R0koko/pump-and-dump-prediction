"""Binance HTTP adapter for discovering currently listed spot currency pairs."""

from typing import Any, Dict, List
import requests

from pumps_and_dumps.core.currency_pair import CurrencyPair


def collect_all_spot_currency_pairs() -> List[CurrencyPair]:
    """Collect a set of all CurrencyPairs traded on Binance"""
    resp = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=30)
    resp.raise_for_status()
    data: Dict[str, Any] = resp.json()
    return [
        CurrencyPair(base=entry["baseAsset"], term=entry["quoteAsset"])
        for entry in data["symbols"]
    ]
