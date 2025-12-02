import os
from pathlib import Path
from typing import Optional
import pandas as pd

try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None


def _read_jsonl(path: Path):
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(__import__('json').loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)


def upload_to_sheet(spreadsheet_id: str, service_account_file: str, sheet_name: str = "snapshots") -> bool:
    """
    Upload `data/market_snapshots.jsonl` to the given Google Sheets spreadsheet.
    Requires a service account JSON file path and that the spreadsheet is shared with the service account email.
    Returns True on success.
    """
    if gspread is None:
        raise RuntimeError("gspread/google-auth not installed")

    sa_file = service_account_file or os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
    if not sa_file:
        raise RuntimeError("Service account file not provided. Set GOOGLE_SERVICE_ACCOUNT_FILE in env or pass path.")

    sa_path = Path(sa_file)
    if not sa_path.exists():
        raise RuntimeError(f"Service account file not found: {sa_path}")

    # read data
    base = Path(__file__).resolve().parents[1] / "data"
    df = _read_jsonl(base / "market_snapshots.jsonl")
    if df.empty:
        # nothing to upload
        return False

    # prepare credentials
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
    ]
    creds = Credentials.from_service_account_file(sa_path, scopes=scopes)
    client = gspread.authorize(creds)

    sh = client.open_by_key(spreadsheet_id)
    try:
        worksheet = sh.worksheet(sheet_name)
    except Exception:
        worksheet = sh.add_worksheet(title=sheet_name, rows=str(len(df) + 10), cols=str(len(df.columns) + 5))

    # write header if sheet empty
    header = df.columns.tolist()
    values = [header] + df.fillna("").astype(str).values.tolist()
    worksheet.clear()
    worksheet.update(values)
    return True
