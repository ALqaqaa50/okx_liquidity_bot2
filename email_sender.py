"""
Email Sender Module
Sends trading reports via email
"""

import os
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger("EmailSender")


class EmailSender:
    """Send emails with attachments"""
    
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.sender_email = os.getenv("SENDER_EMAIL")
        self.sender_password = os.getenv("SENDER_PASSWORD")
        self.recipient_email = os.getenv("RECIPIENT_EMAIL")
        
        if not all([self.sender_email, self.sender_password, self.recipient_email]):
            logger.warning("Email credentials not fully configured in environment variables")
    
    def send_report(
        self,
        excel_file: str,
        subject: Optional[str] = None,
        body: Optional[str] = None,
        additional_files: List[str] = None
    ) -> bool:
        """
        Send trading report via email
        
        Args:
            excel_file: Path to Excel file to attach
            subject: Email subject (optional)
            body: Email body text (optional)
            additional_files: List of additional file paths to attach
        
        Returns:
            bool: True if email sent successfully
        """
        if not all([self.sender_email, self.sender_password, self.recipient_email]):
            logger.error("Email configuration is incomplete. Please set environment variables:")
            logger.error("SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            
            if subject is None:
                subject = f"Trading Bot Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            msg['Subject'] = subject
            
            if body is None:
                body = self._generate_default_body()
            
            msg.attach(MIMEText(body, 'html'))
            
            # Attach main Excel file
            self._attach_file(msg, excel_file)
            
            # Attach additional files if provided
            if additional_files:
                for file_path in additional_files:
                    self._attach_file(msg, file_path)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {self.recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def _attach_file(self, msg: MIMEMultipart, file_path: str):
        """Attach a file to the email message"""
        try:
            filepath = Path(file_path)
            if not filepath.exists():
                logger.warning(f"File not found: {file_path}")
                return
            
            with open(filepath, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {filepath.name}'
            )
            
            msg.attach(part)
            logger.debug(f"Attached file: {filepath.name}")
            
        except Exception as e:
            logger.error(f"Error attaching file {file_path}: {e}")
    
    def _generate_default_body(self) -> str:
        """Generate default email body in HTML format"""
        body = f"""
        <html>
            <body dir="rtl" style="font-family: Arial, sans-serif;">
                <h2>تقرير بوت التداول - OKX Liquidity Bot</h2>
                <p>مرحباً،</p>
                <p>يرجى الاطلاع على تقرير التداول المرفق. يحتوي التقرير على:</p>
                <ul>
                    <li>لقطات السوق (Market Snapshots)</li>
                    <li>الأوامر المنفذة (Orders)</li>
                    <li>مقاييس الأداء (API Metrics)</li>
                    <li>إحصائيات ملخصة (Summary)</li>
                    <li>تحليل الأداء (Performance)</li>
                </ul>
                <p>تاريخ ووقت التقرير: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <hr>
                <p style="color: gray; font-size: 12px;">
                    هذا التقرير تم إنشاؤه تلقائياً بواسطة نظام التداول الآلي.
                </p>
            </body>
        </html>
        """
        return body
    
    def test_connection(self) -> bool:
        """Test email connection and credentials"""
        if not all([self.sender_email, self.sender_password]):
            logger.error("Email credentials not configured")
            return False
        
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
            logger.info("Email connection test successful")
            return True
        except Exception as e:
            logger.error(f"Email connection test failed: {e}")
            return False
