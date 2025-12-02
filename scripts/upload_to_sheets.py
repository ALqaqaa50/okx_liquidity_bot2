import os
from pathlib import Path
from utils.google_sheets import upload_to_sheet

def main():
    # spreadsheet id: prefer env SHEET_ID, else use hard-coded id from user
    sheet_id = os.getenv("SHEET_ID") or "1v6Xszy0p6KWCyrj3Q9ANxnQ9r4IaBWYnhAzyg4MIXAo"
    sa_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
    if not sa_file:
        print("Set GOOGLE_SERVICE_ACCOUNT_FILE environment variable to the service account JSON path.")
        return

    try:
        ok = upload_to_sheet(sheet_id, sa_file, sheet_name="snapshots")
        if ok:
            print("Uploaded snapshots to Google Sheet.")
        else:
            print("No snapshots to upload or upload skipped.")
    except Exception as e:
        print(f"Upload failed: {e}")


if __name__ == "__main__":
    main()
