from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = {
    "Country": "country",
    "ISO3 Code": "iso3_code",
    "Datetime (UTC)": "timestamp_utc",
    "Datetime (Local)": "timestamp_local",
    "Price (EUR/MWhe)": "price_eur_mwh",
}


def load_prices(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {sorted(missing_cols)}")

    df = df.rename(columns=EXPECTED_COLUMNS)

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="raise")
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"], errors="coerce")
    df["price_eur_mwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce")

    df = df.dropna(subset=["timestamp_utc", "price_eur_mwh"]).copy()
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    if df["timestamp_utc"].duplicated().any():
        duplicates = df.loc[df["timestamp_utc"].duplicated(), "timestamp_utc"]
        raise ValueError(f"Duplicate UTC timestamps found, e.g. {duplicates.iloc[:5].tolist()}")

    full_index = pd.date_range(
        start=df["timestamp_utc"].iloc[0],
        end=df["timestamp_utc"].iloc[-1],
        freq="h",
        tz="UTC",
    )
    n_missing = len(full_index) - len(df)
    if n_missing > 0:
        warnings.warn(
            f"Source data has {n_missing} missing hour(s). Rows inserted with NaN price.",
            UserWarning,
            stacklevel=2,
        )

    df = (
        df.set_index("timestamp_utc").reindex(full_index).rename_axis("timestamp_utc").reset_index()
    )
    df[["country", "iso3_code"]] = df[["country", "iso3_code"]].ffill()

    return df
