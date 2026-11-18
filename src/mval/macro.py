"""Monthly macro series from the FRED CSVs in data_external/fred (no network access needed)."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

FRED = Path(__file__).resolve().parents[2] / "data_external" / "fred"

def _read(series_id: str) -> pd.Series:
    df = pd.read_csv(FRED / f"{series_id}.csv", na_values=["."])
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df["observation_date"]), name=series_id).dropna()
    return s

def monthly_rates(start: str = "1980-01-01") -> pd.DataFrame:
    """Month-start indexed table: pmms30, pmms15 (monthly means of weekly PMMS), dgs10, unrate."""
    out = pd.DataFrame({
        "pmms30": _read("MORTGAGE30US").resample("MS").mean(),
        "pmms15": _read("MORTGAGE15US").resample("MS").mean(),
        "dgs10": _read("DGS10").resample("MS").mean(),
        "unrate": _read("UNRATE").resample("MS").mean(),
    })
    out.index.name = "month"
    return out.loc[start:].interpolate(limit_area="inside")

def fhfa_ca_hpi() -> pd.Series:
    """FHFA all-transactions HPI for California, quarterly -> month-start, linear interpolation."""
    s = _read("CASTHPI")
    s.index = s.index.to_period("M").to_timestamp()
    return s.resample("MS").mean().interpolate(limit_area="inside").rename("fhfa_ca_hpi")
