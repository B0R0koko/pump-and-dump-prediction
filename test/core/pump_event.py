from datetime import datetime

from core.currency_pair import CurrencyPair
from core.exchange import Exchange
from core.pump_event import PumpEvent


def test_pump_event():
    pump: PumpEvent = PumpEvent(
        currency_pair=CurrencyPair.from_string("ACM-BTC"),
        time=datetime.strptime("2021-06-05 18:00:13", "%Y-%m-%d %H:%M:%S"),
        exchange=Exchange.BINANCE_SPOT
    )

    inferred_pump: PumpEvent = PumpEvent.from_pump_hash(str(pump))
    assert inferred_pump == pump
