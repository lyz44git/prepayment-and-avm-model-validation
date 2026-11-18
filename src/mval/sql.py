"""Shared SQL fragments: arm's-length sale rules and the residential / owner-occupied universe."""
# Vendor C deed (document) types treated as arm's-length conveyances. Excluded on purpose:
# QC quitclaim, TD trustee's deed (foreclosure / trust), AF affidavit, GF gift, EX/PR/FI estate and
# fiduciary deeds, CS/CN contracts, CV/CR/SU/LM (unknown without the codebook -> excluded, conservative).
ARMS_DOCS = ("WD", "SW", "GD", "BA", "WJ", "DD")
ARMS_SQL = "(" + ",".join(f"'{d}'" for d in ARMS_DOCS) + ")"
RES_SQL = "proptype0 IN ('10','11')"          # 10 single-family, 11 condominium (Vendor C property indicator)

def arms_new(alias=""):
    a = f"{alias}." if alias else ""
    return f"({a}tt1='1' AND {a}doc1 IN {ARMS_SQL} AND {a}sa1 BETWEEN 30000 AND 20000000)"

def arms_old(alias=""):
    a = f"{alias}." if alias else ""
    return f"({a}tt0='1' AND {a}doc0 IN {ARMS_SQL} AND {a}sa0 BETWEEN 30000 AND 20000000)"

def ym_to_idx(col):
    """YYYYMMDD (or YYYYMM00) integer -> months since 1990-01 (0-based)."""
    return f"((({col})//10000 - 1990)*12 + (({col})//100)%100 - 1)"
