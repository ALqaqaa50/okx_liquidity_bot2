import os
import json
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Optional
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def generate_excel_report(out_path: Path):
    """Generate an Excel workbook with sheets: snapshots, orders, metrics."""
    snapshots = _read_jsonl(DATA_DIR / "market_snapshots.jsonl")
    orders = _read_jsonl(DATA_DIR / "orders.jsonl")
    metrics = _read_jsonl(DATA_DIR / "metrics.jsonl")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        if snapshots:
            pd.DataFrame(snapshots).to_excel(writer, sheet_name="snapshots", index=False)
        else:
            pd.DataFrame([]).to_excel(writer, sheet_name="snapshots", index=False)

        if orders:
            pd.DataFrame(orders).to_excel(writer, sheet_name="orders", index=False)
        else:
            pd.DataFrame([]).to_excel(writer, sheet_name="orders", index=False)

        if metrics:
            pd.DataFrame(metrics).to_excel(writer, sheet_name="metrics", index=False)
        else:
            pd.DataFrame([]).to_excel(writer, sheet_name="metrics", index=False)

    return out_path


def send_email_with_attachment(smtp_host: str, smtp_port: int, smtp_user: str, smtp_pass: str,
                               to_email: str, subject: str, body: str, attachment_path: Path) -> bool:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg.set_content(body)

    with open(attachment_path, "rb") as f:
        data = f.read()
    msg.add_attachment(data, maintype="application", subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       filename=attachment_path.name)

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.send_message(msg)
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


def send_report_via_env(out_dir: Optional[Path] = None) -> Optional[Path]:
    """Generate report and send if SMTP env vars present. Returns path if sent/generated."""
    out_dir = out_dir or DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"report_{int(__import__('time').time())}.xlsx"
    generate_excel_report(report_path)

    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    report_to = os.getenv("REPORT_EMAIL_TO")

    if smtp_host and smtp_user and smtp_pass and report_to:
        subject = "OKX Bot Report"
        body = "Attached: periodic report from OKX Liquidity Bot"
        ok = send_email_with_attachment(smtp_host, smtp_port, smtp_user, smtp_pass, report_to, subject, body, report_path)
        if not ok:
            return None

    return report_path
