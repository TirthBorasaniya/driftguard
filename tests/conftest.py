"""Pytest fixtures and markers for the fraud detection pipeline test suite."""

import numpy as np
import pandas as pd
import pytest

from src.config import TRAIN_CSV


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_data: marks tests that need source CSVs (skipped in CI)"
    )


requires_data = pytest.mark.skipif(
    not TRAIN_CSV.exists(),
    reason="fraudTrain.csv not found. Download from Kaggle.",
)


@pytest.fixture
def sample_df():
    """Minimal Sparkov-schema dataframe for unit tests."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "trans_date_trans_time": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "cc_num": [f"4532{i:012d}" for i in range(n)],
        "merchant": np.random.choice(["merchant_A", "merchant_B", "merchant_C"], n),
        "category": np.random.choice(
            ["grocery_pos", "gas_transport", "misc_net", "shopping_net",
             "food_dining", "entertainment", "travel", "health_fitness",
             "personal_care", "home", "kids_pets"], n
        ),
        "amt": np.abs(np.random.exponential(80, n)) + 1.0,
        "gender": np.random.choice(["M", "F"], n),
        "city": np.random.choice(["Austin", "Houston", "Dallas"], n),
        "state": np.random.choice(
            ["TX", "CA", "NY", "FL", "WA", "OR", "NV", "AZ", "CO", "GA",
             "NC", "VA", "OH", "IL", "PA", "MI", "NJ", "MN", "WI", "MO",
             "TN", "IN", "MD", "AK", "AL", "AR", "CT", "DE", "HI", "IA",
             "ID", "KS", "KY", "LA", "MA", "ME", "MS", "MT", "NE", "NH",
             "NM", "ND", "OK", "RI", "SC", "SD", "UT", "VT", "WV", "WY"], n
        ),
        "zip": np.random.choice(["78701", "90210", "10001"], n),
        "lat": np.random.uniform(25, 48, n),
        "long": np.random.uniform(-120, -70, n),
        "city_pop": np.random.randint(5000, 2000000, n),
        "job": np.random.choice(["Engineer", "Teacher", "Doctor", "Artist"], n),
        "dob": "1985-06-15",
        "merch_lat": np.random.uniform(25, 48, n),
        "merch_long": np.random.uniform(-120, -70, n),
        "trans_num": [f"txn_{i:06d}" for i in range(n)],
        "is_fraud": np.random.choice([0, 1], n, p=[0.992, 0.008]),
    })


@pytest.fixture
def sample_transaction_request():
    return {
        "cc_num": "4532015112830366",
        "merchant": "merchant_A",
        "category": "grocery_pos",
        "amt": 149.62,
        "gender": "F",
        "city": "Henderson",
        "state": "TX",
        "zip": "76054",
        "lat": 36.0788,
        "long": -81.1781,
        "city_pop": 35550,
        "job": "Engineer",
        "dob": "1987-01-01",
        "merch_lat": 36.011293,
        "merch_long": -82.048315,
        "trans_date_trans_time": "2020-06-21 12:14:25",
    }
