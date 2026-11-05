import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.feature_sets import BOOKING_TIME_FEATURES


def make_bookings(rows: int = 300, duplicates: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "LeadTime": rng.integers(0, 400, rows),
        "ArrivalDateYear": rng.choice([2015, 2016, 2017], rows),
        "ArrivalDateMonth": rng.choice(["January", "July", "August"], rows),
        "ArrivalDateWeekNumber": rng.integers(1, 53, rows),
        "ArrivalDateDayOfMonth": rng.integers(1, 29, rows),
        "StaysInWeekendNights": rng.integers(0, 4, rows),
        "StaysInWeekNights": rng.integers(0, 8, rows),
        "Adults": rng.integers(1, 4, rows),
        "Children": rng.choice([0.0, 1.0, np.nan], rows),
        "Babies": rng.integers(0, 2, rows),
        "Meal": rng.choice(["BB       ", "HB       ", "SC       "], rows),
        "Country": rng.choice(["PRT", "GBR", "FRA"], rows),
        "MarketSegment": rng.choice(["Online TA", "Direct", "Groups"], rows),
        "DistributionChannel": rng.choice(["TA/TO", "Direct"], rows),
        "IsRepeatedGuest": rng.integers(0, 2, rows),
        "PreviousCancellations": rng.integers(0, 3, rows),
        "PreviousBookingsNotCanceled": rng.integers(0, 3, rows),
        "ReservedRoomType": rng.choice(["A               ", "D               "], rows),
        "DepositType": rng.choice(["No Deposit     ", "Non Refund     "], rows),
        "Agent": rng.choice(["         9", "       NULL", "        14"], rows),
        "Company": rng.choice(["       NULL", "        40"], rows, p=[0.9, 0.1]),
        "CustomerType": rng.choice(["Transient", "Contract"], rows),
        "ADR": rng.uniform(40, 250, rows).round(1),
        "IsCanceled": rng.integers(0, 2, rows),
    }
    df = pd.DataFrame(data)
    return pd.concat([df, df.head(duplicates)], ignore_index=True)


@pytest.fixture
def bookings() -> pd.DataFrame:
    return make_bookings()


@pytest.fixture
def example_booking() -> dict:
    booking = {name: None for name in BOOKING_TIME_FEATURES}
    booking.update({
        "LeadTime": 100,
        "ArrivalDateYear": 2016,
        "ArrivalDateMonth": "July",
        "ArrivalDateWeekNumber": 28,
        "ArrivalDateDayOfMonth": 10,
        "StaysInWeekendNights": 1,
        "StaysInWeekNights": 2,
        "Adults": 2,
        "Children": 0.0,
        "Babies": 0,
        "Meal": "BB",
        "Country": "PRT",
        "MarketSegment": "Online TA",
        "DistributionChannel": "TA/TO",
        "IsRepeatedGuest": 0,
        "PreviousCancellations": 0,
        "PreviousBookingsNotCanceled": 0,
        "ReservedRoomType": "A",
        "DepositType": "No Deposit",
        "Agent": "9",
        "Company": None,
        "CustomerType": "Transient",
    })
    return booking
