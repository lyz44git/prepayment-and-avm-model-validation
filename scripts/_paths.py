"""Shared paths.

The licensed inputs are not in this repository. Scripts 04-17, 21-24 and 31-32 read de-identified working tables from
MVAL_WORK (default: data_work/, git-ignored); the figure scripts read reports/ only and run from a fresh clone.
"""
import os
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("MVAL_WORK", WS / "data_work"))   # de-identified working tables (not distributed)
TMP = Path(os.environ.get("MVAL_TMP", WS / "tmp"))           # DuckDB spill directory
REPORTS = WS / "reports"


def duck(threads=24, mem="160GB"):
    import duckdb
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{TMP}'")
    con.execute(f"SET threads={threads}")
    con.execute(f"SET memory_limit='{mem}'")
    con.execute("SET preserve_insertion_order=false")
    return con
