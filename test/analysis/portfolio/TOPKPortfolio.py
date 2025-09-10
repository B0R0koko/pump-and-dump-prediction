from datetime import datetime

import numpy as np
import pandas as pd

from analysis.models.BaseModel import ImplementsRank
from analysis.portfolio.BasePortfolio import Portfolio
from analysis.portfolio.TOPKPortfolio import TOPKPortfolio
from analysis.utils.columns import COL_CURRENCY_PAIR
from core.currency_pair import CurrencyPair
from core.exchange import Exchange
from core.pump_event import PumpEvent
from core.utils import configure_logging

test_pump: PumpEvent = PumpEvent(
    currency_pair=CurrencyPair.from_string(symbol="ACM-BTC"),
    time=datetime.strptime("2021-06-05 18:00:13", "%Y-%m-%d %H:%M:%S"),
    exchange=Exchange.BINANCE_SPOT
)

cross_section: pd.DataFrame = pd.DataFrame(
    data=[
        ("ADA-BTC",),
        ("ETH-BTC",),
        ("ACM-BTC",),
    ],
    columns=[COL_CURRENCY_PAIR]
)


class DummyTestModel(ImplementsRank):

    def rank(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicted ranks for ADA-BTC, ETH-BTC, ACM-BTC, so our portfolio should only contain ACM-BTC as it has the
        highest logit
        """
        return pd.Series([1, 2, 3])



def test_topk_portfolio():
    configure_logging()
    portfolio_manager: TOPKPortfolio = TOPKPortfolio(model=DummyTestModel(), portfolio_size=1)
    expected_return: float = 0.0001981 / 0.0001953 - 1 # portfolio should only contain

    portfolio_return: float
    portfolio: Portfolio

    portfolio_return, portfolio = portfolio_manager.evaluate_cross_section(cross_section=cross_section, pump=test_pump)

    assert np.abs(portfolio_return - expected_return) < 1e-10, "Returns do not match"
    assert portfolio.currency_pairs == [CurrencyPair.from_string("ACM-BTC")]

