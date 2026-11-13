"""Past-only regression tests for the ordinary execution-impact provider."""

from datetime import datetime, timedelta
from unittest.mock import Mock

import pandas as pd
import pytest

from pumps_and_dumps.backtest.portfolio.impact_provider import (
    LookbackImpactModelProvider,
    quote_to_usdt_or_default,
)
from pumps_and_dumps.core.columns import IS_BUYER_MAKER, PRICE, QUANTITY, TRADE_TIME
from pumps_and_dumps.core.currency_pair import CurrencyPair
from pumps_and_dumps.core.exchange import Exchange
from pumps_and_dumps.core.pump_event import PumpEvent
from pumps_and_dumps.core.time_utils import Bounds


@pytest.mark.parametrize(
    "failure", [OSError("missing prices"), ValueError("bad rate"), RuntimeError("provider failed")]
)
def test_conversion_fallback_logs_provider_errors(failure, caplog):
    """Provider errors keep the established fallback and its diagnostic."""
    provider = Mock()
    provider.get_quote_to_usdt_indicative_price.side_effect = failure
    pair = CurrencyPair.from_string("AAA-BTC")
    timestamp = datetime(2021, 6, 1, 12)

    assert quote_to_usdt_or_default(provider, pair, timestamp) == 1.0
    provider.get_quote_to_usdt_indicative_price.assert_called_once_with(
        quote_asset=pair.term, ts=timestamp
    )
    assert "Falling back to 1.0 quote-to-USDT rate for AAA-BTC" in caplog.text


def test_conversion_fallback_preserves_valid_rates_and_cancellation(caplog):
    """Successful lookups stay silent; cancellation must propagate."""
    pair = CurrencyPair.from_string("AAA-BTC")
    timestamp = datetime(2021, 6, 1, 12)
    provider = Mock()
    provider.get_quote_to_usdt_indicative_price.return_value = 25000.0
    assert quote_to_usdt_or_default(provider, pair, timestamp) == 25000.0
    assert quote_to_usdt_or_default(None, pair, timestamp) == 1.0
    assert not caplog.records

    provider.get_quote_to_usdt_indicative_price.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        quote_to_usdt_or_default(provider, pair, timestamp)


def test_entry_impact_fit_ends_at_entry_and_excludes_preannouncement_future() -> None:
    cp = CurrencyPair.from_string("AAA-BTC")
    pump_time = datetime(2021, 6, 1, 12, 0)
    entry_time = pump_time - timedelta(minutes=15)
    pump = PumpEvent(currency_pair=cp, time=pump_time, exchange=Exchange.BINANCE_SPOT)
    observed_bounds: list[Bounds] = []

    def load_trades(bounds: Bounds, currency_pair: CurrencyPair) -> pd.DataFrame:
        observed_bounds.append(bounds)
        return pd.DataFrame(
            {
                TRADE_TIME: [entry_time - timedelta(minutes=5)],
                PRICE: [100.0],
                QUANTITY: [1.0],
                IS_BUYER_MAKER: [False],
            }
        )

    provider = LookbackImpactModelProvider(load_trades=load_trades, lookback_days=14)
    provider.get_impact_model(pump=pump, currency_pair=cp, end_exclusive=entry_time)

    assert observed_bounds[-1] == Bounds(
        start_inclusive=entry_time - timedelta(days=14),
        end_exclusive=entry_time,
    )


def test_pre_pump_impact_provider_rejects_post_announcement_cutoff() -> None:
    cp = CurrencyPair.from_string("AAA-BTC")
    pump_time = datetime(2021, 6, 1, 12, 0)
    pump = PumpEvent(currency_pair=cp, time=pump_time, exchange=Exchange.BINANCE_SPOT)
    provider = LookbackImpactModelProvider(
        load_trades=lambda bounds, currency_pair: pd.DataFrame(),
        lookback_days=14,
    )

    with pytest.raises(ValueError, match="cannot be fitted past"):
        provider.get_impact_model(
            pump=pump,
            currency_pair=cp,
            end_exclusive=pump_time + timedelta(seconds=1),
        )
