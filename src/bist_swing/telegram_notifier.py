from __future__ import annotations

import os
import requests
from dotenv import load_dotenv

load_dotenv()

class TelegramNotifier:
    def __init__(self, token: str | None = None, chat_id: str | None = None, timeout: int = 15):
        self.token = token or os.getenv("TG_BOT_TOKEN", "").strip()
        self.chat_id = chat_id or os.getenv("TG_CHAT_ID", "").strip()
        self.timeout = timeout
        if not self.token or not self.chat_id:
            raise RuntimeError("Telegram credentials missing. Set TG_BOT_TOKEN and TG_CHAT_ID.")

    def send(self, text: str) -> None:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
        r = requests.post(url, json=payload, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"Telegram send failed: {r.status_code} {r.text}")
