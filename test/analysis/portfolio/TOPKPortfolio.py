from datetime import datetime

import numpy as np
import pandas as pd

from backtest.pipelines.BaseModel import ImplementsRank
from backtest.portfolio.BasePortfolio import Portfolio
from backtest.portfolio.TOPKPortfolio import TOPKPortfolio
from backtest.utils.columns import COL_CURRENCY_PAIR, COL_PUMP_HASH
from backtest.utils.sample import Dataset, DatasetType
from backtest.utils.feature_set import FeatureSet
from core.currency_pair import CurrencyPair
from core.exchange import Exchange
from core.pump_event import PumpEvent
from core.utils import configure_logging

test_pump: PumpEvent = PumpEvent(
    currency_pair=CurrencyPair.from_string(symbol="ACM-BTC"),
    time=datetime.strptime("2021-06-05 18:00:13", "%Y-%m-%d %H:%M:%S"),
    exchange=Exchange.BINANCE_SPOT,
)

cross_section: pd.DataFrame = pd.DataFrame(
    data=[
        ("ADA-BTC", test_pump.as_pump_hash()),
        ("ETH-BTC", test_pump.as_pump_hash()),
        ("ACM-BTC", test_pump.as_pump_hash()),
    ],
    columns=[COL_CURRENCY_PAIR, COL_PUMP_HASH],
)


class DummyTestModel(ImplementsRank):

    def rank(self, dataset: Dataset) -> pd.Series:
        """
        Predicted ranks for ADA-BTC, ETH-BTC, ACM-BTC, so our portfolio should only contain ACM-BTC as it has the
        highest logit
        """
        return pd.Series([1, 2, 3])


def test_topk_portfolio():
    configure_logging()
    portfolio_manager: TOPKPortfolio = TOPKPortfolio(model=DummyTestModel(), portfolio_size=1)
    expected_return: float = 0.0001981 / 0.0001953 - 1  # portfolio should only contain

    portfolio_return: float
    portfolio: Portfolio

    dataset: Dataset = Dataset(
        data=cross_section,
        feature_set=FeatureSet(
            numeric_features=[],
            categorical_features=[],
            target="",
            eval_fields=[COL_CURRENCY_PAIR, COL_PUMP_HASH],
        ),
        ds_type=DatasetType.TEST,
    )

    portfolio_return, portfolio = portfolio_manager.evaluate_cross_section(dataset=dataset, pump=test_pump)
    print(portfolio_return, portfolio)

    assert np.abs(portfolio_return - expected_return) < 1e-10, "Returns do not match"
    assert portfolio.currency_pairs == [CurrencyPair.from_string("ACM-BTC")]
